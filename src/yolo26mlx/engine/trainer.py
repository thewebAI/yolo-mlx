# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 Trainer - Pure MLX Implementation

Training loop for YOLO26 models with mx.compile optimization.
Uses MLX v0.30.3 with proper state capture for compiled training graphs.
"""

import logging
import math
import re
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml
from mlx.optimizers import clip_grad_norm
from mlx.utils import tree_map

from yolo26mlx.data.coco_dataset import COCODataset
from yolo26mlx.optim.adamw import AdamW
from yolo26mlx.optim.musgd import MuSGD
from yolo26mlx.utils.coco_metrics import COCOMetrics
from yolo26mlx.utils.loss import E2ELoss, v8PoseLoss, v8SegmentationLoss
from yolo26mlx.utils.metrics import (
    OKS_SIGMA,
    PoseMetrics,
    SegmentationMetrics,
    gt_instance_masks_from_overlap,
    process_masks_at_proto,
)

logger = logging.getLogger(__name__)


class ModelEMA:
    """Exponential Moving Average of model weights.

    Port of PyTorch's ultralytics ModelEMA (torch_utils.py L606).
    Keeps a running average of model parameters for smoother validation.

    decay function: d = decay * (1 - exp(-updates / tau))
    - Starts near 0 (no averaging early on)
    - Ramps to `decay` (0.9999) as training progresses
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Initialize EMA with a deep copy of the model's parameters.

        Args:
            model: YOLO26 model instance to track.
            decay: Maximum decay rate (asymptotic target).
            tau: Time constant controlling the decay ramp-up speed.
            updates: Initial update counter (non-zero when resuming).
        """
        # Deep copy of model parameters (EMA shadow weights)
        self.ema_params = tree_map(lambda p: mx.array(p), model.parameters())
        self.updates = updates
        self.decay_max = decay
        self.tau = tau
        self.enabled = True

    def _decay(self):
        """Compute current decay rate (exponential ramp).

        Returns:
            Decay value as a float, ramping from ~0 toward decay_max.
        """
        return self.decay_max * (1 - math.exp(-self.updates / self.tau))

    def update(self, model):
        """Update EMA parameters from the model's current parameters.

        Args:
            model: YOLO26 model whose parameters to blend into the EMA.
        """
        if not self.enabled:
            return
        self.updates += 1
        d = self._decay()
        self.ema_params = tree_map(
            lambda e, m: d * e + (1.0 - d) * m, self.ema_params, model.parameters()
        )
        # Note: mx.eval deferred to caller's sync point to avoid extra GPU flush

    def apply(self, model):
        """Load EMA weights into the model for validation.

        Args:
            model: YOLO26 model to overwrite with EMA weights.

        Returns:
            Deep copy of the model's original (non-EMA) parameters for later restore.
        """
        original_params = tree_map(lambda p: mx.array(p), model.parameters())
        model.update(self.ema_params)
        return original_params

    def restore(self, model, original_params):
        """Restore original weights after EMA validation.

        Args:
            model: YOLO26 model to restore.
            original_params: Parameter tree previously returned by apply().
        """
        model.update(original_params)


class Trainer:
    """YOLO26 Training class - Pure MLX.

    Implements compiled training with proper MLX state management.
    Uses mx.compile with input/output capture for model and optimizer state.
    """

    def __init__(self, model: nn.Module, task: str = "detect"):
        """Initialize trainer.

        Args:
            model: YOLO26 model to train
            task: Task type - 'detect', 'segment', 'pose', or 'obb'
        """
        self.model = model
        self.task = task
        # Keypoint shape for pose training (K, dims). Read from the head when
        # available so the keypoint target tensor matches the model output.
        head = model.model[-1] if hasattr(model, "model") else None
        self._kpt_shape = tuple(getattr(head, "kpt_shape", (17, 3)))
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None

        # Training state
        self.epoch = 0
        self.best_fitness = 0.0
        self.ema = None  # Exponential moving average

        # Compiled step function (will be created during training)
        self._step_fn = None

        # Warmup tracking (per-iteration, matches PyTorch)
        self._warmup_nw = 0  # total warmup iterations
        self._nb = 0  # batches per epoch

        # BN-freeze flag (set by __call__). When True, ``_apply_bn_freeze``
        # walks the model after every ``model.train()`` and forces every
        # ``nn.BatchNorm`` back into eval mode so its running stats stay put.
        self._freeze_bn = False

    def _apply_bn_freeze(self) -> None:
        """Force every ``nn.BatchNorm`` into eval mode if BN is frozen.

        ``nn.Module.train()`` recursively flips ``_training`` to True on every
        submodule, including BatchNorm. This helper undoes that flip for BN
        layers only, so:

        * BN forward uses pretrained ``running_mean`` / ``running_var`` for
          normalization (no train/eval feature-distribution mismatch), and
        * BN ``running_mean`` / ``running_var`` are NOT mutated by the
          forward pass, eliminating the drift that otherwise tanks
          validation mAP on small batches.

        BN ``weight`` / ``bias`` remain trainable — gradients still flow.
        """
        if not self._freeze_bn:
            return

        def _bn_to_eval(_, module: nn.Module) -> None:
            if isinstance(module, nn.BatchNorm):
                module._set_training_mode(False)

        self.model.apply_to_modules(_bn_to_eval)

    def __call__(
        self,
        data: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        patience: int = 50,
        save_period: int = -1,
        project: str = "runs/train",
        name: str = "exp",
        exist_ok: bool = False,
        resume: bool = False,
        weight_decay: float = 0.0005,
        momentum: float = 0.937,
        lr: float | None = None,
        optimizer: str = "auto",
        freeze_bn: bool | None = None,
        val: bool = True,  # Enable/disable validation during training
        verbose: bool = True,  # Enable/disable progress printing
    ) -> dict[str, Any]:
        """Run training.

        Args:
            data: Path to data configuration file
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs (-1 to disable)
            project: Project directory
            name: Experiment name
            exist_ok: Overwrite existing experiment
            resume: Resume from last checkpoint
            weight_decay: Weight decay
            momentum: SGD momentum
            lr: Override the auto-computed base learning rate. If None, the
                LR is derived from ``MuSGD.auto_lr`` / ``AdamW.auto_lr``
                depending on ``optimizer`` (matches PyTorch Ultralytics
                ``optimizer='auto'``). Pass a float to fine-tune pretrained
                weights at a smaller LR.
            optimizer: Optimizer selection. ``"auto"`` (default) mirrors
                Ultralytics' ``optimizer='auto'``: AdamW for short runs
                (``iterations <= 10000``, the typical fine-tune regime) and
                MuSGD for longer runs. Use ``"adamw"`` or ``"musgd"`` to force
                a specific optimizer.
            freeze_bn: Freeze BatchNorm running statistics during training.
                When ``True`` every ``nn.BatchNorm`` is held in eval mode for
                the whole training loop, so its ``running_mean`` /
                ``running_var`` are never updated and the forward pass uses
                the (pretrained) running stats for normalization — only
                ``weight`` / ``bias`` are still trained. This is the standard
                fine-tuning fix for small batches on small datasets, where
                noisy 2- or 4-image batch statistics otherwise drift the BN
                running stats away from the pretrained feature distribution
                and tank validation mAP. ``None`` (default) auto-enables the
                freeze for typical fine-tune runs (``iterations <= 10000``,
                same threshold Ultralytics uses to auto-pick AdamW).
            val: Run validation after each epoch (default: True)
            verbose: Print progress (default: True)

        Returns:
            Training results dict
        """
        # Setup save directory
        save_dir = Path(project) / name
        save_dir.mkdir(parents=True, exist_ok=exist_ok or resume)

        # Load data config
        data_cfg = self._load_data_config(data)

        # Get number of classes from dataset config (important for custom datasets!)
        self._num_classes = data_cfg.get("nc", 80)
        if verbose:
            logger.info(f"  Dataset classes: {self._num_classes}")

        # Pre-load datasets once (not every epoch)
        self._train_dataset = None
        self._val_dataset = None
        self._preload_datasets(data_cfg, imgsz)

        # Setup warmup parameters (matches PyTorch: warmup_epochs=3.0)
        if self._train_dataset is not None:
            self._nb = len(
                list(self._train_dataset.get_dataloader(batch_size=batch, shuffle=False))
            )
        else:
            self._nb = 32  # fallback
        warmup_epochs_cfg = 3.0
        self._warmup_nw = (
            max(round(warmup_epochs_cfg * self._nb), 100) if warmup_epochs_cfg > 0 else -1
        )

        # Compute iterations for auto LR (matches PyTorch)
        nbs = 64
        accumulate_est = max(round(nbs / batch), 1)
        iterations_est = (
            math.ceil(len(self._train_dataset) / max(batch, nbs)) * epochs
            if self._train_dataset
            else 20
        )

        # Scale weight decay by effective batch ratio (matches PyTorch L344):
        #   weight_decay = args.weight_decay * batch_size * accumulate / nbs
        scaled_wd = weight_decay * batch * accumulate_est / nbs

        # Setup optimizer matching PyTorch Ultralytics ``optimizer='auto'``:
        # for short runs (iterations <= 10000) Ultralytics resolves ``auto`` to
        # **AdamW** with ``lr_fit = 0.002 * 5 / (4 + nc)``, NOT MuSGD. Using
        # MuSGD on a pretrained checkpoint at that LR perturbs weights
        # aggressively (Newton-Schulz orthogonalization) and degrades small-set
        # post-training mAP — which is why earlier MLX seg fine-tunes
        # underperformed PyTorch despite identical LR. ``optimizer="auto"``
        # below now picks AdamW or MuSGD with the same heuristic Ultralytics
        # does in ``ultralytics/engine/trainer.py:build_optimizer``.
        self._setup_optimizer(
            momentum,
            scaled_wd,
            iterations=iterations_est,
            lr_override=lr,
            optimizer_choice=optimizer,
        )
        self._total_epochs = epochs  # stored for warmup LR calculation + E2ELoss decay
        self._setup_loss()

        # Create compiled training step
        self._create_compiled_step()

        # Initialize EMA (matches PyTorch: ModelEMA(self.model))
        self.ema = ModelEMA(self.model)

        # Decide whether to freeze BN running stats. ``None`` mirrors the
        # auto-AdamW threshold above: short fine-tune runs (≤10k iterations)
        # see drifted BN running stats from noisy small-batch updates, and
        # validation mAP collapses even when weights barely change. Freezing
        # BN keeps the pretrained running_mean / running_var intact and only
        # trains BN weight / bias — the standard fine-tune recipe.
        if freeze_bn is None:
            self._freeze_bn = iterations_est <= 10000
        else:
            self._freeze_bn = bool(freeze_bn)

        # Set model to training mode
        self.model.train()
        self._apply_bn_freeze()

        if verbose:
            logger.info(f"\nTraining YOLO26 for {epochs} epochs")
            logger.info(f"  Data: {data}")
            logger.info(f"  Image size: {imgsz}")
            logger.info(f"  Batch size: {batch}")
            logger.info(f"  Learning rate: {self._lr0}")
            opt_name = getattr(self, "_optimizer_name", type(self.optimizer).__name__)
            if opt_name == "MuSGD":
                logger.info(
                    f"  Optimizer: MuSGD (muon={self.optimizer.muon_scale}, "
                    f"sgd={self.optimizer.sgd_scale})"
                )
            else:
                logger.info(
                    f"  Optimizer: {opt_name} (betas=({self.optimizer.beta1}, "
                    f"{self.optimizer.beta2}), wd={self.optimizer.weight_decay})"
                )
            logger.info(f"  Freeze BN running stats: {self._freeze_bn}")
            logger.info(f"  Save directory: {save_dir}")

        # Training loop
        results = {
            "epochs_completed": 0,
            "best_fitness": 0.0,
            "final_loss": 0.0,
        }

        no_improvement = 0

        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # Linear LR schedule (matches PyTorch ultralytics default)
            # lf(epoch) = max(1 - epoch/epochs, 0) * (1.0 - lrf) + lrf
            # where lrf=0.01 (final LR = lr0 * 0.01)
            lrf = 0.01
            lf = max(1 - epoch / epochs, 0) * (1.0 - lrf) + lrf
            self.optimizer.learning_rate = self._lr0 * lf

            # Training epoch
            if verbose:
                logger.info(
                    f"\n--- Epoch {epoch + 1}/{epochs} starting (lr={self.optimizer.learning_rate:.6f}) ---"
                )
            train_loss = self._train_epoch(batch_size=batch, imgsz=imgsz, verbose=verbose)

            # Update E2ELoss weights (decay one2many, increase one2one)
            if hasattr(self.loss_fn, "update"):
                self.loss_fn.update()

            # Validation (optional - for fair benchmarking, can be disabled)
            if val:
                val_metrics = self._validate(batch, imgsz)
                fitness = val_metrics.get("mAP50", 0.0)
            else:
                val_metrics = {"mAP50": 0.0, "mAP50-95": 0.0}
                fitness = 0.0

            # Check for improvement (only if validation enabled)
            if val and fitness > self.best_fitness:
                self.best_fitness = fitness
                no_improvement = 0
                # Save best model
                self._save_checkpoint(save_dir / "best.safetensors")
            else:
                no_improvement += 1

            # Early stopping (only if validation enabled)
            if val and patience > 0 and no_improvement >= patience:
                if verbose:
                    logger.info(f"\nEarly stopping at epoch {epoch + 1}")
                break

            # Periodic save
            if save_period > 0 and (epoch + 1) % save_period == 0:
                self._save_checkpoint(save_dir / f"epoch{epoch + 1}.safetensors")

            # Per-epoch GPU memory hygiene. Sustained training of larger
            # segmentation models (yolo26x-seg) on Apple Silicon can fragment
            # the Metal heap across epochs, surfacing as monotonically rising
            # epoch times and intermittent kIOGPUCommandBufferCallbackError
            # hangs. Releasing MLX's reusable buffer pool at the epoch boundary
            # (a known-safe sync point — model/optimizer/EMA state has just
            # been mx.eval'd) stops that drift without affecting correctness.
            mx.clear_cache()

            # Log progress
            epoch_time = time.time() - epoch_start
            if verbose:
                if val:
                    # Validation ran this epoch — emit the full metric line.
                    map50 = val_metrics.get("mAP50", 0.0)
                    map50_95 = val_metrics.get("mAP50-95", 0.0)
                    precision = val_metrics.get("precision", 0.0)
                    recall = val_metrics.get("recall", 0.0)
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs}: "
                        f"loss={train_loss:.4f}, "
                        f"mAP50={map50:.4f}, "
                        f"mAP50-95={map50_95:.4f}, "
                        f"P={precision:.4f}, "
                        f"R={recall:.4f}, "
                        f"time={epoch_time:.1f}s"
                    )
                else:
                    # Validation disabled (e.g. throughput benchmarks). Don't
                    # log mAP=0.0000 — that wasn't measured, it's the default
                    # placeholder, and printing it confuses readers into
                    # thinking training collapsed.
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs}: "
                        f"loss={train_loss:.4f}, "
                        f"time={epoch_time:.1f}s "
                        f"(val skipped)"
                    )

        # Save final model
        self._save_checkpoint(save_dir / "last.safetensors")

        results["epochs_completed"] = self.epoch + 1
        results["best_fitness"] = self.best_fitness
        results["final_loss"] = train_loss
        results["save_dir"] = str(save_dir)

        return results

    def _load_data_config(self, data: str) -> dict:
        """Load data configuration from YAML, searching package cfg/datasets if needed.

        Args:
            data: Path or filename of the YAML data configuration.

        Returns:
            Parsed YAML configuration as a dict.
        """
        data_path = Path(data)

        # If not absolute path, search in package's cfg/datasets directory
        if not data_path.exists():
            # Try package cfg/datasets directory
            package_dir = Path(__file__).parent.parent
            cfg_datasets_path = package_dir / "cfg" / "datasets" / data
            if cfg_datasets_path.exists():
                data_path = cfg_datasets_path
            else:
                # Also try with .yaml extension if not provided
                if not data.endswith(".yaml"):
                    cfg_datasets_path = package_dir / "cfg" / "datasets" / f"{data}.yaml"
                    if cfg_datasets_path.exists():
                        data_path = cfg_datasets_path

        if not data_path.exists():
            raise FileNotFoundError(f"Data config not found: {data}")

        with open(data_path) as f:
            cfg = yaml.safe_load(f)

        return cfg

    def _setup_optimizer(
        self,
        momentum: float,
        weight_decay: float,
        iterations: int = 20,
        lr_override: float | None = None,
        optimizer_choice: str = "auto",
    ):
        """Setup optimizer matching PyTorch Ultralytics ``optimizer='auto'``.

        Ultralytics' ``build_optimizer`` resolves ``optimizer='auto'`` based on
        the estimated iteration count (see
        ``ultralytics/engine/trainer.py:build_optimizer``):

        * ``iterations <= 10000`` → **AdamW** with
          ``lr_fit = round(0.002 * 5 / (4 + nc), 6)``, ``betas=(0.9, 0.999)``,
          and weight_decay only on the "weight" param group (zero on bias /
          norm). This is the path actual COCO128 / fine-tuning runs hit.
        * ``iterations > 10000`` → **MuSGD** at ``lr=0.01`` with
          ``muon=0.1, sgd=1.0``. This is the train-from-scratch path.

        For both branches:
        - Per-group weight decay (decay on weights, none on bias / BN).
        - Momentum 0.937 (the user-facing default) is reserved as the
          *warmup-end* target; the optimizer itself is built with momentum=0.9
          to match Ultralytics' auto override.

        Only the MuSGD branch additionally applies the fine-tuning regex
        ``(?=.*23)(?=.*cv3)|proto.semseg|flow_model`` with ``lr * 3``;
        ``ultralytics.engine.trainer.build_optimizer`` gates this 3× boost
        behind ``if use_muon:`` (L1037–1052) and AdamW gets a uniform LR
        across all groups. Applying the boost to AdamW too — as we did
        previously — drove the head LR to ``3 × 1.19e-4 ≈ 3.57e-4`` and was
        the dominant cause of the COCO128-Seg post-train mAP gap on the
        larger models (worst on yolo26x-seg).

        Args:
            momentum: Momentum target (warmup end), typically 0.937.
            weight_decay: Pre-scaled weight decay coefficient.
            iterations: Estimated total optimizer iterations.
            lr_override: Optional explicit base LR (overrides the auto value).
            optimizer_choice: ``"auto"`` (mirrors Ultralytics), ``"adamw"``,
                or ``"musgd"`` to force a specific optimizer.
        """
        nc = getattr(self, "_num_classes", 80)
        choice = (optimizer_choice or "auto").lower()
        if choice == "auto":
            choice = "adamw" if iterations <= 10000 else "musgd"

        # PyTorch auto mode overrides momentum 0.937 → 0.9 for both AdamW and
        # MuSGD (see Ultralytics build_optimizer L998). Store the user-facing
        # 0.937 as the *warmup-end* target; the optimizer runs at 0.9.
        self._args_momentum = momentum  # 0.937 — warmup-end target
        opt_momentum = 0.9

        # Weight decay is already scaled by caller:
        #   scaled_wd = weight_decay * batch_size * accumulate / nbs
        scaled_wd = weight_decay

        if choice == "adamw":
            auto_lr = AdamW.auto_lr(nc=nc) if lr_override is None else float(lr_override)
            self._lr0 = auto_lr
            self._optimizer_name = "AdamW"
            self.optimizer = AdamW(
                model=self.model,
                lr=auto_lr,
                betas=(opt_momentum, 0.999),
                eps=1e-8,
                weight_decay=scaled_wd,
                bias_correction=True,
            )
            # No per-param LR scale: PyTorch's ``build_optimizer`` only applies
            # the 3× fine-tune boost when ``use_muon`` is True (L1037), so
            # AdamW must run with a uniform LR across all groups.
        elif choice == "musgd":
            auto_lr_v, muon_scale, sgd_scale = MuSGD.auto_lr(nc=nc, iterations=iterations)
            auto_lr = float(lr_override) if lr_override is not None else auto_lr_v
            self._lr0 = auto_lr
            self._optimizer_name = "MuSGD"
            self.optimizer = MuSGD(
                model=self.model,
                lr=auto_lr,
                momentum=opt_momentum,
                weight_decay=scaled_wd,
                muon_scale=muon_scale,
                sgd_scale=sgd_scale,
                nesterov=True,
            )
            # MuSGD-only: 3× LR on the cv3 head + Proto26 semseg + flow heads
            # (matches ``build_optimizer`` lines 1037–1052).
            ft_pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg|flow_model")
            self.optimizer.set_lr_scale(self.model, ft_pattern, scale=3.0)
        else:
            raise ValueError(
                f"Unknown optimizer choice {optimizer_choice!r}; expected one of "
                "'auto', 'adamw', 'musgd'."
            )

    def _setup_loss(self):
        """Setup loss function based on task.

        Uses E2ELoss which trains both one2many and one2one detection heads.
        For segmentation, uses v8SegmentationLoss which adds mask loss to
        the one2many branch.
        """
        if self.task == "segment":
            self.loss_fn = E2ELoss(model=self.model, loss_fn=v8SegmentationLoss)
        elif self.task == "pose":
            self.loss_fn = E2ELoss(model=self.model, loss_fn=v8PoseLoss)
        else:
            self.loss_fn = E2ELoss(model=self.model)

        # Set actual epoch count for E2ELoss decay schedule.
        # Without this, _epochs defaults to 100, causing the o2m weight
        # to decay too slowly for short training runs (e.g. 10 epochs).
        if hasattr(self.loss_fn, "set_epochs"):
            self.loss_fn.set_epochs(self._total_epochs)

    # Fixed annotation count — targets are padded to this size so that
    # each batch has the same annotation tensor shapes, allowing MLX to
    # build a uniform computation graph.
    MAX_ANNOTATIONS = 200

    def _create_compiled_step(self):
        """Create compute_grad and step closures for the training loop."""

        def compute_grad(images, targets):
            """Compute forward pass loss and parameter gradients."""

            def loss_fn(model):
                """Forward pass through model and loss, returning scalar loss."""
                preds = model(images)
                loss, loss_items = self.loss_fn(preds, targets)
                return loss

            loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
            loss, grads = loss_and_grad_fn(self.model)
            return loss, grads

        self._compute_grad_fn = compute_grad

        def step(images, targets):
            """Single training step: compute gradients and update weights."""
            loss, grads = compute_grad(images, targets)
            self.optimizer.step(self.model, grads)
            return loss

        self._step_fn = step

    def _create_simple_compiled_step(self):
        """Create simplified mx.compile'd step for throughput benchmarking."""

        @mx.compile
        def step(images):
            """Fully compiled forward + backward + optimizer step on one batch."""

            def loss_fn(model):
                """Compute a simple mean-of-predictions loss for benchmarking."""
                preds = model(images)
                # Simple loss: mean of predictions
                if isinstance(preds, dict):
                    if "one2many" in preds:
                        preds = preds["one2many"]
                    total = mx.array(0.0)
                    for key in ["boxes", "scores"]:
                        if key in preds:
                            total = total + mx.mean(preds[key])
                    return total
                return mx.mean(preds)

            loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
            loss, grads = loss_and_grad_fn(self.model)
            self.optimizer.step(self.model, grads)
            return loss

        self._simple_step_fn = step

    # Known dataset download URLs (name → zip URL)
    _DATASET_URLS = {
        "coco128": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        "coco128-seg": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip",
        "coco8-pose": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip",
    }

    def _download_dataset(self, name: str, dest_dir: Path) -> Path | None:
        """Download and extract a known dataset.

        Args:
            name: Dataset name (e.g. "coco128").
            dest_dir: Directory to download and extract into.

        Returns:
            Path to the extracted dataset, or None on failure.
        """
        url = self._DATASET_URLS.get(name)
        if url is None:
            return None

        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dest_dir / f"{name}.zip"
        logger.info("Downloading %s dataset (~7 MB)...", name)
        try:
            result = subprocess.run(
                ["curl", "-L", "-f", "-o", str(zip_path), url],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0 or not zip_path.exists():
                raise RuntimeError(f"curl failed (code {result.returncode}): {result.stderr}")
            logger.info("Extracting %s...", name)
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(dest_dir))
            zip_path.unlink()
            dataset_path = dest_dir / name
            if not (dataset_path / "images").exists():
                raise RuntimeError(f"Extracted archive missing {name}/images/ directory")
            logger.info("Downloaded %s to: %s", name, dataset_path)
            return dataset_path
        except Exception as e:
            logger.error("Failed to download %s: %s", name, e)
            if zip_path.exists():
                zip_path.unlink()
            return None

    def _preload_datasets(self, data_cfg: dict, imgsz: int):
        """Pre-load training and validation datasets once.

        Args:
            data_cfg: Data configuration
            imgsz: Image size
        """
        dataset_path = data_cfg.get("path", "")
        train_path = data_cfg.get("train", "images/train2017")
        val_path = data_cfg.get("val", "images/train2017")

        # Resolve paths
        dataset_name = Path(dataset_path).name  # strip parent dirs for download lookup
        dataset_path = Path(dataset_path)
        if not dataset_path.is_absolute():
            possible_paths = [
                Path.home() / ".config" / "Ultralytics" / "datasets" / str(dataset_path),
                Path(__file__).parent.parent.parent.parent / "datasets" / str(dataset_path),
                Path.cwd() / "datasets" / str(dataset_path),
                dataset_path,
            ]
            resolved = False
            for p in possible_paths:
                if p.exists() and (p / train_path).exists():
                    dataset_path = p
                    resolved = True
                    break

            # Auto-download if dataset not found locally.
            # Resolve target relative to the user's CWD so the dataset lands at
            # ``./datasets/<name>/`` (matches the README) regardless of whether
            # yolo-mlx is installed editable from the repo or as a wheel from PyPI.
            if not resolved and dataset_name in self._DATASET_URLS:
                datasets_dir = Path.cwd() / "datasets"
                downloaded = self._download_dataset(dataset_name, datasets_dir)
                if downloaded is not None:
                    dataset_path = downloaded

        self._dataset_path = dataset_path

        # Pose-specific dataset args (keypoint shape + left/right flip map).
        pose_kwargs = {}
        if self.task == "pose":
            pose_kwargs = {
                "kpt_shape": self._kpt_shape,
                "flip_idx": data_cfg.get("flip_idx"),
            }

        # Load training dataset
        train_split = Path(train_path).name
        train_images_dir = dataset_path / train_path
        if train_images_dir.exists():
            self._train_dataset = COCODataset(
                root=str(dataset_path),
                split=train_split,
                img_size=imgsz,
                augment=True,
                task=self.task,
                **pose_kwargs,
            )

        # Load validation dataset
        val_split = Path(val_path).name
        val_images_dir = dataset_path / val_path
        if val_images_dir.exists():
            self._val_dataset = COCODataset(
                root=str(dataset_path),
                split=val_split,
                img_size=imgsz,
                task=self.task,
                **pose_kwargs,
            )

    def _train_epoch(self, batch_size: int, imgsz: int, verbose: bool = True) -> float:
        """Run one training epoch.

        For short training runs on small datasets, warmup and gradient
        accumulation are disabled since they reduce the effective number
        of optimizer steps too much (e.g., warmup_nw=100 iterations but
        only 320 total iterations leaves little room for actual training).

        For longer runs on larger datasets, warmup and accumulation can be
        re-enabled to match PyTorch's behavior.

        Args:
            batch_size: Batch size
            imgsz: Image size
            verbose: Print per-batch progress

        Returns:
            Average loss for epoch
        """
        # Use pre-loaded dataset (no reloading every epoch)
        dataset = self._train_dataset

        if dataset is None:
            logger.warning("  Warning: Training dataset not loaded")
            logger.info("  Using synthetic data for training (results may not be meaningful)")
            return self._train_epoch_synthetic(batch_size, imgsz, verbose=verbose)

        # Gradient accumulation + warmup — always enabled (matching PyTorch).
        # PyTorch NEVER conditionally skips warmup or accumulation.
        # accumulate = round(nbs / batch_size) ensures effective batch = nbs = 64.
        nbs = 64
        accumulate = max(round(nbs / batch_size), 1)

        total_loss = 0.0
        num_batches = 0
        accumulated_grads = None
        steps_since_update = 0

        # Get dataloader
        dataloader = dataset.get_dataloader(batch_size=batch_size, shuffle=True)
        batches = list(dataloader)  # materialize to know total count
        num_total = len(batches)
        nb = num_total  # batches per epoch

        # Progress logging: log every N batches so output stays readable
        log_interval = max(1, num_total // 10) if num_total > 10 else 1
        epoch_t0 = time.time()

        for batch_i, (batch_images, batch_annotations) in enumerate(batches):
            # --- Warmup (per-iteration, matches PyTorch) ---
            ni = batch_i + nb * self.epoch  # global iteration
            if self._warmup_nw > 0 and ni <= self._warmup_nw:
                # Ramp accumulate from 1 to nbs/batch_size during warmup
                accumulate = max(
                    1, int(round(np.interp(float(ni), [0, self._warmup_nw], [1, nbs / batch_size])))
                )
                # Ramp LR from 0 to target (scheduled) LR
                lrf = 0.01
                lf = max(1 - self.epoch / max(self._total_epochs, 1), 0) * (1.0 - lrf) + lrf
                target_lr = self._lr0 * lf
                warmup_lr = float(np.interp(float(ni), [0, self._warmup_nw], [0.0, target_lr]))
                self.optimizer.learning_rate = warmup_lr
                # Ramp momentum from warmup_momentum (0.8) to target (0.937)
                # Matches PyTorch: np.interp(ni, xi, [args.warmup_momentum, args.momentum])
                warmup_mom = float(
                    np.interp(float(ni), [0, self._warmup_nw], [0.8, self._args_momentum])
                )
                self.optimizer.momentum = warmup_mom
            # Convert annotations to target format for v8DetectionLoss
            # Collect all boxes and labels across the batch
            batch_idx_list = []
            cls_list = []
            bboxes_list = []
            kpts_list = []

            for img_idx, ann in enumerate(batch_annotations):
                boxes = ann["boxes"]  # (N, 4) in xyxy normalized format
                labels = ann["labels"]  # (N,)

                if len(boxes) > 0:
                    # Convert xyxy to xywh format (center x, center y, width, height)
                    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    xywh = np.stack([cx, cy, w, h], axis=-1)

                    batch_idx_list.extend([img_idx] * len(boxes))
                    cls_list.extend(labels.tolist())
                    bboxes_list.append(xywh)

                    if self.task == "pose":
                        kpts_list.append(ann["keypoints"])  # (N, K, 3) aligned with boxes

            # Handle case where no annotations in batch
            if len(bboxes_list) == 0:
                continue

            # Stack all annotations
            batch_idx = mx.array(batch_idx_list, dtype=mx.int32)
            cls = mx.array(cls_list, dtype=mx.int32)
            bboxes = mx.array(np.concatenate(bboxes_list, axis=0), dtype=mx.float32)

            # Pad targets to fixed size for mx.compile compatibility.
            # Padding entries use batch_idx=batch_size (out of range),
            # so they are ignored by the loss function's preprocess scatter.
            n_annot = len(batch_idx_list)
            pad_n = self.MAX_ANNOTATIONS - n_annot
            if pad_n > 0:
                batch_idx = mx.concatenate(
                    [batch_idx, mx.full((pad_n,), batch_size, dtype=mx.int32)]
                )
                cls = mx.concatenate([cls, mx.zeros((pad_n,), dtype=mx.int32)])
                bboxes = mx.concatenate([bboxes, mx.zeros((pad_n, 4), dtype=mx.float32)])
            elif pad_n < 0:
                # More annotations than MAX_ANNOTATIONS — truncate (rare)
                batch_idx = batch_idx[: self.MAX_ANNOTATIONS]
                cls = cls[: self.MAX_ANNOTATIONS]
                bboxes = bboxes[: self.MAX_ANNOTATIONS]

            targets = {
                "batch_idx": batch_idx,
                "cls": cls,
                "bboxes": bboxes,
            }

            # Include keypoints for pose training, aligned with the box order
            # and padded to MAX_ANNOTATIONS so the loss can reuse the box
            # preprocessing scatter (see v8PoseLoss._preprocess_keypoints).
            if self.task == "pose":
                kpts_np = (
                    np.concatenate(kpts_list, axis=0)
                    if kpts_list
                    else np.zeros((0, self._kpt_shape[0], 3), dtype=np.float32)
                )
                kdim0, kdim1 = self._kpt_shape[0], 3
                if pad_n > 0:
                    kpts_np = np.concatenate(
                        [kpts_np, np.zeros((pad_n, kdim0, kdim1), dtype=np.float32)], axis=0
                    )
                elif pad_n < 0:
                    kpts_np = kpts_np[: self.MAX_ANNOTATIONS]
                targets["keypoints"] = mx.array(kpts_np.astype(np.float32), dtype=mx.float32)

            # Include masks and sem_masks for segmentation training
            if self.task == "segment":
                mask_list = []
                sem_mask_list = []
                mask_h = self._train_dataset.img_size // self._train_dataset.mask_ratio
                for ann in batch_annotations:
                    if "masks" in ann:
                        mask_list.append(ann["masks"])
                    else:
                        mask_list.append(np.zeros((mask_h, mask_h), dtype=np.int32))
                    if "sem_masks" in ann:
                        sem_mask_list.append(ann["sem_masks"])
                    else:
                        sem_mask_list.append(np.zeros((mask_h, mask_h), dtype=np.int64))
                targets["masks"] = mx.array(np.stack(mask_list), dtype=mx.int32)
                targets["sem_masks"] = mx.array(np.stack(sem_mask_list), dtype=mx.int32)

            # Compute loss and gradients (no optimizer update yet)
            loss, grads = self._compute_grad_fn(batch_images, targets)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, grads)
            steps_since_update += 1

            # Evaluate every batch to keep the computation graph bounded
            # and prevent Metal memory from growing unbounded.
            #
            # NOTE: pass the gradient TREE (not a flat-unpacked list of leaves)
            # to ``mx.eval``. ``mx.eval(*flat_arrays)`` raises
            # ``[eval] Attempting to eval an array without a primitive`` on
            # mlx 0.31.x for the post-step state sync (user-reported, see
            # CHANGELOG). The tree form walks the same set of leaves and is
            # the documented MLX-recommended pattern (see ml-explore/mlx
            # discussion #2914 and the existing line in
            # ``_train_epoch_synthetic``: ``mx.eval(self.model.state, ...)``).
            mx.eval(loss, accumulated_grads)

            # Periodic mid-epoch GPU buffer-pool release. The per-batch
            # mx.eval above is already a sync point, so dropping MLX's
            # reusable buffer pool here is safe. Empirically, larger
            # segmentation models (yolo26x-seg) accumulate Metal heap
            # fragmentation within a single epoch — releasing every 8
            # batches keeps fragmentation bounded without measurable
            # throughput impact on smaller models.
            if (batch_i + 1) % 8 == 0:
                mx.clear_cache()

            # Step optimizer every `accumulate` batches, or on last batch
            is_last = batch_i == num_total - 1
            if steps_since_update >= accumulate or is_last:
                # PyTorch does NOT divide accumulated gradients by accumulate.
                # It accumulates raw gradient sums via .backward() and steps directly.
                # Clip gradient norm (matches PyTorch max_norm=10.0)
                accumulated_grads, _grad_norm = clip_grad_norm(accumulated_grads, max_norm=10.0)
                # Update model with clipped gradients (MuSGD handles weight decay)
                self.optimizer.step(self.model, accumulated_grads)
                # Update EMA after each optimizer step (matches PyTorch optimizer_step)
                if self.ema is not None:
                    self.ema.update(self.model)
                # Single GPU sync for model params + optimizer state + EMA params.
                # Pass trees (not flat-unpacked positional arrays) — matches the
                # pattern used at the synthetic-data sync site below and the
                # MLX-recommended idiom from ml-explore/mlx discussion #2914.
                # Avoids the ``[eval] Attempting to eval an array without a
                # primitive`` failure mode on mlx 0.31.x.
                sync_targets = [self.model.parameters(), self.optimizer.state]
                if self.ema is not None:
                    sync_targets.append(self.ema.ema_params)
                mx.eval(sync_targets)
                accumulated_grads = None
                steps_since_update = 0

            total_loss += float(loss)
            num_batches += 1

            if verbose and (batch_i % log_interval == 0 or batch_i == num_total - 1):
                avg_loss = total_loss / num_batches
                elapsed = time.time() - epoch_t0
                logger.info(
                    f"  Epoch {self.epoch + 1}/{self._total_epochs} "
                    f"batch {batch_i + 1}/{num_total} — "
                    f"loss: {avg_loss:.4f}, "
                    f"lr: {float(self.optimizer.learning_rate):.6f}, "
                    f"elapsed: {elapsed:.1f}s"
                )

        return total_loss / max(num_batches, 1)

    def _train_epoch_synthetic(
        self, batch_size: int, imgsz: int, num_batches: int = 100, verbose: bool = True
    ) -> float:
        """Fallback training with synthetic data when dataset is unavailable.

        Args:
            batch_size: Batch size
            imgsz: Image size
            num_batches: Number of synthetic batches
            verbose: Print per-batch progress

        Returns:
            Average loss for epoch
        """
        total_loss = 0.0
        log_interval = max(1, num_batches // 10) if num_batches > 10 else 1
        epoch_t0 = time.time()

        for i in range(num_batches):
            # Generate synthetic data
            images = mx.random.uniform(shape=(batch_size, imgsz, imgsz, 3))

            # Synthetic targets
            num_objects_per_image = 5
            total_objects = batch_size * num_objects_per_image

            batch_idx = mx.repeat(mx.arange(batch_size), num_objects_per_image)
            cls = mx.random.randint(0, 80, shape=(total_objects,))
            bboxes = mx.random.uniform(shape=(total_objects, 4)) * 0.5 + 0.1

            targets = {
                "batch_idx": batch_idx,
                "cls": cls,
                "bboxes": bboxes,
            }

            loss = self._step_fn(images, targets)
            mx.eval(self.model.state, self.optimizer.state)
            total_loss += float(loss)

            if verbose and (i % log_interval == 0 or i == num_batches - 1):
                avg_loss = total_loss / (i + 1)
                elapsed = time.time() - epoch_t0
                logger.info(
                    f"  Epoch {self.epoch + 1}/{self._total_epochs} "
                    f"batch {i + 1}/{num_batches} (synthetic) — "
                    f"loss: {avg_loss:.4f}, "
                    f"elapsed: {elapsed:.1f}s"
                )

        return total_loss / num_batches

    def _validate(self, batch_size: int, imgsz: int) -> dict[str, float]:
        """Run validation using official COCO metrics.

        Detection path uses ``COCOMetrics`` (box-only mAP). Segmentation path
        is dispatched to ``_validate_segment``, which evaluates both mask
        and box mAP at proto resolution to match Ultralytics' internal
        ``SegmentMetrics`` (so MLX numbers are directly comparable to
        ``ultralytics.YOLO.val()`` mask mAP after training).

        Args:
            batch_size: Batch size
            imgsz: Image size

        Returns:
            Validation metrics dict. For detection: ``mAP50``, ``mAP50-95``,
            ``precision``, ``recall``. For segmentation, additionally
            ``mAP50_mask``, ``mAP50-95_mask``, ``mAP50_box``, ``mAP50-95_box``.
        """
        if self.task == "segment":
            return self._validate_segment(batch_size, imgsz)
        if self.task == "pose":
            return self._validate_pose(batch_size, imgsz)

        # Set model to eval mode
        self.model.eval()

        # Use EMA weights for validation (matches PyTorch behavior)
        original_params = None
        if self.ema is not None and self.ema.enabled:
            original_params = self.ema.apply(self.model)

        metrics = {
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

        # Use pre-loaded validation dataset (no reloading every epoch)
        dataset = self._val_dataset

        if dataset is None:
            logger.warning("  Warning: Validation dataset not loaded")
            self.model.train()
            self._apply_bn_freeze()
            return metrics

        # Initialize COCO metrics calculator with correct number of classes
        # Use dataset's nc (not hardcoded 80) for proper evaluation
        num_classes = getattr(self, "_num_classes", 80)
        coco_metrics = COCOMetrics(num_classes=num_classes)

        dataloader = dataset.get_dataloader(batch_size=batch_size, shuffle=False)

        image_id = 0
        for batch_images, batch_annotations in dataloader:
            # Run inference (MLX doesn't require explicit no_grad context)
            preds = self.model(batch_images)
            if isinstance(preds, tuple):
                # Segmentation/pose heads return (detections, extra) tuples;
                # extract detections array for box-metric validation.
                mx.eval(*preds)
                preds = preds[0]
            else:
                mx.eval(preds)

            # Process each image in the batch
            batch_size_actual = batch_images.shape[0]

            # Handle different output formats:
            # 1. Training mode returns dict: {'one2one': {...}, 'one2many': {...}}
            # 2. Inference mode returns array: (B, max_det, 6[+nm]) with [x,y,w,h,conf,class_idx,...]

            if isinstance(preds, mx.array):
                # Inference mode: (B, max_det, 6) = [x, y, w, h, conf, class_idx]
                preds_np = np.array(preds)

                for b in range(batch_size_actual):
                    det = preds_np[b]  # (max_det, 6)

                    # Extract predictions
                    boxes_xywh = det[:, :4]  # (N, 4) in pixel coordinates
                    scores = det[:, 4]  # (N,) confidence scores
                    class_ids = det[:, 5].astype(np.int64)  # (N,) class indices

                    # Convert xywh to xyxy (still in pixel coords)
                    x, y, w, h = (
                        boxes_xywh[:, 0],
                        boxes_xywh[:, 1],
                        boxes_xywh[:, 2],
                        boxes_xywh[:, 3],
                    )
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)

                    # Normalize to [0, 1] to match GT format
                    boxes_xyxy[:, [0, 2]] /= imgsz  # x coords
                    boxes_xyxy[:, [1, 3]] /= imgsz  # y coords

                    # Filter by confidence threshold
                    conf_mask = scores > 0.001

                    predictions = {
                        "boxes": boxes_xyxy[conf_mask],
                        "scores": scores[conf_mask],
                        "labels": class_ids[conf_mask],
                    }

                    # Get ground truth for this image
                    if b < len(batch_annotations):
                        ann = batch_annotations[b]
                        gt_boxes = ann.get("boxes", np.zeros((0, 4)))
                        gt_labels = ann.get("labels", np.zeros(0, dtype=np.int64))
                        gt_iscrowd = ann.get("iscrowd", np.zeros(len(gt_labels), dtype=bool))

                        ground_truth = {
                            "boxes": gt_boxes,
                            "labels": gt_labels,
                            "iscrowd": gt_iscrowd,
                        }
                    else:
                        ground_truth = {
                            "boxes": np.zeros((0, 4)),
                            "labels": np.zeros(0, dtype=np.int64),
                            "iscrowd": np.zeros(0, dtype=bool),
                        }

                    # Update COCO metrics with this image
                    coco_metrics.update(predictions, ground_truth, image_id)
                    image_id += 1

            elif isinstance(preds, dict):
                # Training mode: dict with 'one2one' or 'one2many' keys
                if "one2one" in preds:
                    pred_dict = preds["one2one"]
                elif "one2many" in preds:
                    pred_dict = preds["one2many"]
                else:
                    pred_dict = preds

                for b in range(batch_size_actual):
                    if "boxes" in pred_dict and "scores" in pred_dict:
                        pred_boxes = np.array(pred_dict["boxes"][b])  # (N, 4)
                        pred_scores = np.array(pred_dict["scores"][b])  # (N, num_classes)

                        max_scores = np.max(pred_scores, axis=-1)
                        pred_labels = np.argmax(pred_scores, axis=-1)
                        conf_mask = max_scores > 0.001

                        predictions = {
                            "boxes": pred_boxes[conf_mask],
                            "scores": max_scores[conf_mask],
                            "labels": pred_labels[conf_mask],
                        }
                    else:
                        predictions = {
                            "boxes": np.zeros((0, 4)),
                            "scores": np.zeros(0),
                            "labels": np.zeros(0, dtype=np.int64),
                        }

                    if b < len(batch_annotations):
                        ann = batch_annotations[b]
                        gt_boxes = ann.get("boxes", np.zeros((0, 4)))
                        gt_labels = ann.get("labels", np.zeros(0, dtype=np.int64))
                        gt_iscrowd = ann.get("iscrowd", np.zeros(len(gt_labels), dtype=bool))

                        ground_truth = {
                            "boxes": gt_boxes,
                            "labels": gt_labels,
                            "iscrowd": gt_iscrowd,
                        }
                    else:
                        ground_truth = {
                            "boxes": np.zeros((0, 4)),
                            "labels": np.zeros(0, dtype=np.int64),
                            "iscrowd": np.zeros(0, dtype=bool),
                        }

                    coco_metrics.update(predictions, ground_truth, image_id)
                    image_id += 1

        # Compute final metrics using proper COCO protocol
        results = coco_metrics.compute()

        metrics["mAP50"] = round(results.get("mAP50", 0.0), 4)
        metrics["mAP50-95"] = round(results.get("mAP50-95", 0.0), 4)
        metrics["precision"] = round(results.get("precision", 0.0), 4)
        metrics["recall"] = round(results.get("recall", 0.0), 4)

        # Set model back to training mode
        # Restore original (non-EMA) weights for continued training
        if original_params is not None:
            self.ema.restore(self.model, original_params)
        self.model.train()
        self._apply_bn_freeze()

        return metrics

    def _validate_segment(self, batch_size: int, imgsz: int) -> dict[str, float]:
        """Segmentation validation: mask + box mAP at proto resolution.

        Mirrors Ultralytics' internal segmentation evaluator:
        - Inference produces ``(det, proto)``; mask coefficients are combined
          with the proto grid and cropped to the predicted box, then
          binarized — all at proto (160×160 for 640) resolution.
        - GT masks come from ``COCODataset.task='segment'`` (overlap map at
          ``img_size // mask_ratio = 160`` for our config), split into
          per-instance binary masks at the same resolution.
        - Both are accumulated by ``SegmentationMetrics``, which computes
          per-class 101-point AP with greedy one-to-one TP matching — the
          same recipe as ``ultralytics.utils.metrics.ap_per_class``.

        Args:
            batch_size: Validation batch size.
            imgsz: Input image size (used only as a sanity check; resolutions
                are derived from the proto grid and the dataloader's overlap
                map, which are intrinsically aligned).

        Returns:
            Dict with ``mAP50_mask``, ``mAP50-95_mask``, ``mAP50_box``,
            ``mAP50-95_box``, plus legacy ``mAP50``/``mAP50-95`` aliases set
            to the mask values for backward compatibility with callers that
            only read those keys.
        """
        # Default zero metrics — returned on early exits.
        metrics: dict[str, float] = {
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "mAP50_mask": 0.0,
            "mAP50-95_mask": 0.0,
            "mAP50_box": 0.0,
            "mAP50-95_box": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

        self.model.eval()

        # Use EMA weights for validation (matches PyTorch behaviour).
        original_params = None
        if self.ema is not None and self.ema.enabled:
            original_params = self.ema.apply(self.model)

        dataset = self._val_dataset
        if dataset is None:
            logger.warning("  Warning: Validation dataset not loaded")
            self.model.train()
            self._apply_bn_freeze()
            return metrics

        num_classes = getattr(self, "_num_classes", 80)
        seg_metrics = SegmentationMetrics(num_classes=num_classes)
        conf_thresh = 0.001

        try:
            for batch_images, batch_annotations in dataset.get_dataloader(
                batch_size=batch_size, shuffle=False
            ):
                outputs = self.model(batch_images)
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    mx.eval(outputs[0], outputs[1])
                    det_np = np.array(outputs[0])
                    proto_np = np.array(outputs[1])
                elif isinstance(outputs, mx.array):
                    # Inference path that did not return a proto branch — we
                    # cannot compute mask mAP, so fall back to box-only via
                    # the predicted detections (still useful as a sanity
                    # check, but mask metrics will stay at 0).
                    mx.eval(outputs)
                    det_np = np.array(outputs)
                    proto_np = None
                else:
                    # Training-mode dict (one2one/one2many) shouldn't appear
                    # in eval, but guard against it.
                    continue

                actual_batch = det_np.shape[0]
                for b in range(actual_batch):
                    pred_i = det_np[b]
                    proto_i = proto_np[b] if proto_np is not None else None

                    grid_masks, grid_boxes, grid_scores, grid_labels = process_masks_at_proto(
                        pred_i, proto_i, conf_thresh
                    )

                    if b < len(batch_annotations):
                        ann = batch_annotations[b]
                    else:
                        ann = {}

                    overlap = ann.get("masks")
                    overlap_arr = np.array(overlap) if overlap is not None else np.array([])
                    gt_stack, k_inst = gt_instance_masks_from_overlap(overlap_arr)

                    if k_inst > 0:
                        gt_boxes = ann["boxes"][:k_inst].astype(np.float32)
                        gt_labels = ann["labels"][:k_inst].astype(np.int64)
                        gt_masks_arg: np.ndarray | None = gt_stack
                    else:
                        # Either no instances or no rasterized masks for this
                        # image. Still feed any boxes/labels so the box-mAP
                        # part of SegmentationMetrics can count GTs correctly.
                        gt_boxes = np.asarray(ann.get("boxes", np.zeros((0, 4))), dtype=np.float32)
                        gt_labels = np.asarray(
                            ann.get("labels", np.zeros(0, dtype=np.int64)),
                            dtype=np.int64,
                        )
                        gt_masks_arg = None

                    seg_metrics.update(
                        grid_boxes,
                        grid_scores,
                        grid_labels,
                        grid_masks if grid_masks.size > 0 else None,
                        gt_boxes,
                        gt_labels,
                        gt_masks_arg,
                    )
        except Exception as e:  # noqa: BLE001 — keep training robust
            logger.warning("  Segmentation validation failed: %s", e)
            if original_params is not None:
                self.ema.restore(self.model, original_params)
            self.model.train()
            self._apply_bn_freeze()
            return metrics

        results = seg_metrics.compute()

        m50_mask = float(results.get("mAP50_mask", 0.0))
        m5095_mask = float(results.get("mAP50-95_mask", 0.0))
        m50_box = float(results.get("mAP50_box", 0.0))
        m5095_box = float(results.get("mAP50-95_box", 0.0))

        metrics.update(
            {
                "mAP50_mask": round(m50_mask, 4),
                "mAP50-95_mask": round(m5095_mask, 4),
                "mAP50_box": round(m50_box, 4),
                "mAP50-95_box": round(m5095_box, 4),
                # Legacy aliases — surface mask mAP under the unprefixed
                # keys so existing callers that only read mAP50/mAP50-95
                # see the segmentation-relevant number.
                "mAP50": round(m50_mask, 4),
                "mAP50-95": round(m5095_mask, 4),
                "precision": float(results.get("precision_mask", 0.0)),
                "recall": float(results.get("recall_mask", 0.0)),
            }
        )

        if original_params is not None:
            self.ema.restore(self.model, original_params)
        self.model.train()
        self._apply_bn_freeze()
        return metrics

    def _validate_pose(self, batch_size: int, imgsz: int) -> dict[str, float]:
        """Pose validation: keypoint (OKS) + box mAP.

        Mirrors Ultralytics ``PoseValidator``: the end-to-end pose head emits
        ``(B, max_det, 6 + K*3)`` detections with keypoints decoded into
        letterboxed pixel space. Predictions and GT are both normalized into
        the letterbox square (OKS is scale-invariant to the common scale) and
        accumulated by ``PoseMetrics``, which computes per-class 101-point AP
        with greedy one-to-one matching across IoU 0.5:0.05:0.95.

        Args:
            batch_size: Validation batch size.
            imgsz: Input image size (letterbox side length, used to normalize
                the predicted pixel-space boxes/keypoints to [0, 1]).

        Returns:
            Dict with ``mAP50_pose``, ``mAP50-95_pose``, ``mAP50_box``,
            ``mAP50-95_box``, plus legacy ``mAP50``/``mAP50-95`` aliases set to
            the pose values for callers that only read those keys.
        """
        metrics: dict[str, float] = {
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "mAP50_pose": 0.0,
            "mAP50-95_pose": 0.0,
            "mAP50_box": 0.0,
            "mAP50-95_box": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

        self.model.eval()

        original_params = None
        if self.ema is not None and self.ema.enabled:
            original_params = self.ema.apply(self.model)

        dataset = self._val_dataset
        if dataset is None:
            logger.warning("  Warning: Validation dataset not loaded")
            self.model.train()
            self._apply_bn_freeze()
            return metrics

        nkpt = self._kpt_shape[0]
        sigmas = OKS_SIGMA if tuple(self._kpt_shape) == (17, 3) else np.ones(nkpt) / nkpt
        pose_metrics = PoseMetrics(num_classes=getattr(self, "_num_classes", 1), sigmas=sigmas)
        conf_thresh = 0.001

        try:
            for batch_images, batch_annotations in dataset.get_dataloader(
                batch_size=batch_size, shuffle=False
            ):
                outputs = self.model(batch_images)
                if isinstance(outputs, tuple):
                    # Pose head returns the detection+keypoint array directly;
                    # any auxiliary tuple element is unused here.
                    mx.eval(*outputs)
                    det_np = np.array(outputs[0])
                elif isinstance(outputs, mx.array):
                    mx.eval(outputs)
                    det_np = np.array(outputs)
                else:
                    continue

                actual_batch = det_np.shape[0]
                for b in range(actual_batch):
                    pred_i = det_np[b]  # (max_det, 6 + K*3)
                    scores = pred_i[:, 4]
                    keep = scores > conf_thresh
                    pred_i = pred_i[keep]

                    if pred_i.shape[0] > 0 and pred_i.shape[1] >= 6 + nkpt * 3:
                        bx = pred_i[:, :4]
                        p_scores = pred_i[:, 4]
                        p_labels = pred_i[:, 5].astype(np.int64)
                        cx, cy, w, h = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
                        p_boxes = (
                            np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)
                            / imgsz
                        )
                        p_kpts = pred_i[:, 6 : 6 + nkpt * 3].reshape(-1, nkpt, 3).astype(np.float32)
                        p_kpts[..., 0] /= imgsz
                        p_kpts[..., 1] /= imgsz
                    else:
                        p_boxes = np.zeros((0, 4), dtype=np.float32)
                        p_scores = np.zeros(0, dtype=np.float32)
                        p_labels = np.zeros(0, dtype=np.int64)
                        p_kpts = np.zeros((0, nkpt, 3), dtype=np.float32)

                    ann = batch_annotations[b] if b < len(batch_annotations) else {}
                    gt_boxes = np.asarray(ann.get("boxes", np.zeros((0, 4))), dtype=np.float32)
                    gt_labels = np.asarray(
                        ann.get("labels", np.zeros(0, dtype=np.int64)), dtype=np.int64
                    )
                    gt_kpts = np.asarray(
                        ann.get("keypoints", np.zeros((0, nkpt, 3))), dtype=np.float32
                    )

                    pose_metrics.update(
                        p_boxes,
                        p_scores,
                        p_labels,
                        p_kpts if p_kpts.shape[0] > 0 else None,
                        gt_boxes,
                        gt_labels,
                        gt_kpts if gt_kpts.shape[0] > 0 else None,
                    )
        except Exception as e:  # noqa: BLE001 — keep training robust
            logger.warning("  Pose validation failed: %s", e)
            if original_params is not None:
                self.ema.restore(self.model, original_params)
            self.model.train()
            self._apply_bn_freeze()
            return metrics

        results = pose_metrics.compute()
        m50_pose = float(results.get("mAP50_pose", 0.0))
        m5095_pose = float(results.get("mAP50-95_pose", 0.0))

        metrics.update(
            {
                "mAP50_pose": round(m50_pose, 4),
                "mAP50-95_pose": round(m5095_pose, 4),
                "mAP50_box": round(float(results.get("mAP50_box", 0.0)), 4),
                "mAP50-95_box": round(float(results.get("mAP50-95_box", 0.0)), 4),
                # Legacy aliases — surface pose mAP under the unprefixed keys.
                "mAP50": round(m50_pose, 4),
                "mAP50-95": round(m5095_pose, 4),
                "precision": float(results.get("precision_pose", 0.0)),
                "recall": float(results.get("recall_pose", 0.0)),
            }
        )

        if original_params is not None:
            self.ema.restore(self.model, original_params)
        self.model.train()
        self._apply_bn_freeze()
        return metrics

    def _save_checkpoint(self, path: str | Path):
        """Save model checkpoint.

        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_weights(str(path))
        logger.info(f"Saved checkpoint to {path}")
