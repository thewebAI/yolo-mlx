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
from mlx.utils import tree_flatten, tree_map

from yolo26mlx.data.coco_dataset import COCODataset
from yolo26mlx.optim.musgd import MuSGD
from yolo26mlx.utils.coco_metrics import COCOMetrics
from yolo26mlx.utils.loss import E2ELoss

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

        # Setup training (auto LR + MuSGD, matching PyTorch optimizer='auto')
        self._setup_optimizer(momentum, scaled_wd, iterations=iterations_est)
        self._total_epochs = epochs  # stored for warmup LR calculation + E2ELoss decay
        self._setup_loss()

        # Create compiled training step
        self._create_compiled_step()

        # Initialize EMA (matches PyTorch: ModelEMA(self.model))
        self.ema = ModelEMA(self.model)

        # Set model to training mode
        self.model.train()

        if verbose:
            logger.info(f"\nTraining YOLO26 for {epochs} epochs")
            logger.info(f"  Data: {data}")
            logger.info(f"  Image size: {imgsz}")
            logger.info(f"  Batch size: {batch}")
            logger.info(f"  Learning rate: {self._lr0}")
            logger.info(
                f"  Optimizer: MuSGD (muon={self.optimizer.muon_scale}, sgd={self.optimizer.sgd_scale})"
            )
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

            # Log progress
            epoch_time = time.time() - epoch_start
            if verbose:
                # Extract all metrics
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

    def _setup_optimizer(self, momentum: float, weight_decay: float, iterations: int = 20):
        """Setup MuSGD optimizer matching PyTorch 'optimizer=auto'.

        PyTorch's ultralytics uses MuSGD when optimizer='auto'.
        It auto-computes LR from nc and uses:
        - Muon (Newton-Schulz orthogonalization) for 2D+ weight tensors
        - Nesterov SGD for all param groups
        - Weight decay only on conv weights (not BN/bias)

        Args:
            momentum: Momentum for SGD.
            weight_decay: Weight decay coefficient.
            iterations: Estimated total optimizer iterations.
        """
        # Auto-compute LR from nc (matches PyTorch optimizer='auto')
        nc = getattr(self, "_num_classes", 80)
        auto_lr, muon_scale, sgd_scale = MuSGD.auto_lr(nc=nc, iterations=iterations)

        # PyTorch auto mode overrides momentum from 0.937 → 0.9 for MuSGD:
        #   name, lr, momentum = ("MuSGD", ..., 0.9)  # trainer.py L951
        # Store original for warmup target, use 0.9 for optimizer init.
        self._args_momentum = momentum  # 0.937 — used as warmup end target
        momentum = 0.9  # auto override, matching PyTorch

        # Weight decay is already scaled by caller:
        #   scaled_wd = weight_decay * batch_size * accumulate / nbs
        # So we use it directly here.
        scaled_wd = weight_decay

        self._lr0 = auto_lr
        self.optimizer = MuSGD(
            model=self.model,
            lr=auto_lr,
            momentum=momentum,
            weight_decay=scaled_wd,
            muon_scale=muon_scale,
            sgd_scale=sgd_scale,
            nesterov=True,
        )

        # Apply fine-tuning LR boost (matches PyTorch trainer.py L993-1001):
        # Parameters matching the regex get lr * 3. In PyTorch this splits each
        # param group into two sub-groups; here we store per-path LR scales.
        ft_pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg|flow_model")
        self.optimizer.set_lr_scale(self.model, ft_pattern, scale=3.0)

    def _setup_loss(self):
        """Setup loss function based on task.

        Uses E2ELoss which trains both one2many and one2one detection heads.
        The one2one head is what's used at inference time, so it must receive
        training gradients (not just one2many).
        """
        # E2ELoss creates two v8DetectionLoss instances internally:
        #   - one2many (topk=10) for dense matching
        #   - one2one  (topk=7, topk2=1) for end-to-end matching
        # Each branch has its own TaskAlignedAssigner.
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
        dataset_name = dataset_path  # original name for download lookup
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

            # Auto-download if dataset not found locally
            if not resolved and dataset_name in self._DATASET_URLS:
                datasets_dir = Path(__file__).parent.parent.parent.parent / "datasets"
                downloaded = self._download_dataset(dataset_name, datasets_dir)
                if downloaded is not None:
                    dataset_path = downloaded

        self._dataset_path = dataset_path

        # Load training dataset
        train_split = Path(train_path).name
        train_images_dir = dataset_path / train_path
        if train_images_dir.exists():
            self._train_dataset = COCODataset(
                root=str(dataset_path),
                split=train_split,
                img_size=imgsz,
                augment=True,
            )

        # Load validation dataset
        val_split = Path(val_path).name
        val_images_dir = dataset_path / val_path
        if val_images_dir.exists():
            self._val_dataset = COCODataset(
                root=str(dataset_path),
                split=val_split,
                img_size=imgsz,
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
            flat_grads = tree_flatten(accumulated_grads)
            mx.eval(loss, *[v for _, v in flat_grads])

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
                # Single GPU sync for model params + optimizer state + EMA params
                # (consolidated from 4 separate mx.eval calls)
                eval_targets = [v for _, v in tree_flatten(self.model.parameters())]
                eval_targets.extend(self.optimizer.state)
                if self.ema is not None:
                    eval_targets.extend([v for _, v in tree_flatten(self.ema.ema_params)])
                mx.eval(*eval_targets)
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

        Uses COCOMetrics class for proper mAP calculation following the
        official COCO evaluation protocol (as documented in ISSUES_RESOLVED.md).

        Args:
            batch_size: Batch size
            imgsz: Image size

        Returns:
            Validation metrics dict with mAP50, mAP50-95, precision, recall
        """
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
            mx.eval(preds)  # Force evaluation

            # Process each image in the batch
            batch_size_actual = batch_images.shape[0]

            # Handle different output formats:
            # 1. Training mode returns dict: {'one2one': {...}, 'one2many': {...}}
            # 2. Inference mode returns array: (B, max_det, 6) with [x,y,w,h,conf,class_idx]

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
