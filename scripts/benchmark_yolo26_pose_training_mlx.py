#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 Pose Training Benchmark (Pure MLX)
=========================================
Measures training performance for YOLO26 pose models using native MLX on Apple Silicon.

This benchmark uses the pure MLX implementation of YOLO26 for training,
providing accurate measurements of MLX-native training performance.

Usage:
    python benchmark_yolo26_pose_training_mlx.py
    python benchmark_yolo26_pose_training_mlx.py --models n s      # Specific models only
    python benchmark_yolo26_pose_training_mlx.py --epochs 5        # Fewer epochs
    python benchmark_yolo26_pose_training_mlx.py --batch 2           # Smaller batch size
    python benchmark_yolo26_pose_training_mlx.py --output custom.json

Output:
    ../results/yolo26_pose_mlx_training_final.json (default, overridable via --output)
"""

import argparse
import gc
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from _runtime_dirs import ensure_runtime_dirs

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

EPOCHS = 10
BATCH_SIZE = 4
# Default LR matches Ultralytics' ``optimizer='auto'`` short-run choice for
# nc=1 (pose, person-only): ``round(0.002 * 5 / (4 + 1), 6) == 0.002``. With
# the default ``--optimizer auto`` this is consumed by AdamW (the optimizer
# Ultralytics itself picks for ``iterations <= 10000``); see
# ``Trainer._setup_optimizer``.
LEARNING_RATE = 0.002
DEFAULT_OPTIMIZER = "auto"
MODEL_SIZES = ["n", "s", "m", "l", "x"]

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR / ".."
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"
DATASETS_DIR = PROJECT_DIR / "datasets"

# COCO 17-keypoint left/right flip swap map (Ultralytics coco8-pose.yaml).
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


# =============================================================================
# Utility Functions
# =============================================================================


def _lr_source_label(lr_used: float) -> str:
    """Describe whether ``lr_used`` came from the auto formula or the user.

    Returns:
        ``"auto (Ultralytics build_optimizer, nc=1)"`` when ``lr_used`` matches
        the script default (which equals ``round(0.002 * 5 / (4 + 1), 6)``);
        otherwise ``"user-provided (--lr)"``.
    """
    return (
        "auto (Ultralytics build_optimizer, nc=1)"
        if lr_used == LEARNING_RATE
        else "user-provided (--lr)"
    )


def _optimizer_label(choice: str) -> str:
    """Pretty label for the optimizer ``--optimizer`` choice for benchmark JSON."""
    if choice == "auto":
        return "auto (AdamW for iter<=10000, MuSGD otherwise — matches Ultralytics)"
    if choice == "adamw":
        return "AdamW (forced)"
    if choice == "musgd":
        return "MuSGD (Muon + Nesterov SGD, forced)"
    return choice


def get_device_info() -> dict[str, Any]:
    """Get system and device information.

    Returns:
        Dict with platform, processor, Python version, and CPU name (macOS).
    """
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # Try to get chip name on macOS
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            info["cpu"] = result.stdout.strip()
    except Exception:
        pass

    return info


def clear_mlx_memory():
    """Run garbage collection, flush the MLX Metal cache, and reset peak memory tracking."""
    gc.collect()
    import mlx.core as mx

    mx.clear_cache()
    mx.reset_peak_memory()


def get_mlx_memory() -> tuple[float, float]:
    """Get MLX Metal memory usage in MB.

    Returns:
        Tuple of (active_memory_mb, peak_memory_mb)
    """
    import mlx.core as mx

    active = mx.get_active_memory() / 1024 / 1024
    peak = mx.get_peak_memory() / 1024 / 1024
    return active, peak


def setup_coco8_pose() -> Path:
    """Download and setup the COCO8-Pose dataset.

    Returns:
        Path to a local YAML config file describing the dataset (with the
        keypoint shape and left/right flip map required by the pose loader).
    """
    search_paths = [
        DATASETS_DIR / "coco8-pose",
        Path("datasets") / "coco8-pose",
        Path.cwd() / "coco8-pose",
    ]

    dataset_path = None
    for path in search_paths:
        if path.exists() and (path / "images").exists():
            dataset_path = path
            break

    if dataset_path is None:
        logger.info("  COCO8-Pose not found locally. Downloading...")
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DATASETS_DIR / "coco8-pose.zip"
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip"
        try:
            import zipfile

            result = subprocess.run(
                ["curl", "-L", "-f", "-o", str(zip_path), url],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0 or not zip_path.exists():
                raise RuntimeError(f"curl failed (code {result.returncode}): {result.stderr}")
            logger.info("  Extracting...")
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(DATASETS_DIR))
            zip_path.unlink()
            dataset_path = DATASETS_DIR / "coco8-pose"
            if not (dataset_path / "images").exists():
                raise RuntimeError("Extracted archive missing coco8-pose/images/ directory")
            logger.info(f"  Downloaded COCO8-Pose to: {dataset_path}")
        except Exception as e:
            logger.error(f"  ERROR: Failed to download COCO8-Pose: {e}")
            logger.warning("  Falling back to the packaged coco8-pose.yaml config.")
            if zip_path.exists():
                zip_path.unlink()
            return Path("coco8-pose.yaml")

    logger.info(f"  Found COCO8-Pose at: {dataset_path}")

    # coco8-pose ships images under images/train and images/val.
    train_dir = (
        "images/train" if (dataset_path / "images" / "train").exists() else "images/train2017"
    )
    val_dir = "images/val" if (dataset_path / "images" / "val").exists() else train_dir

    local_yaml = DATASETS_DIR / "coco8_pose_local.yaml"
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    config = f"""# COCO8-Pose Local Configuration
path: {dataset_path.absolute()}
train: {train_dir}
val: {val_dir}
test:

kpt_shape: [17, 3]
flip_idx: {FLIP_IDX}

names:
  0: person
nc: 1
"""

    with open(local_yaml, "w") as f:
        f.write(config)

    return local_yaml


def save_results(results: dict, output_path: Path, prefix: str = "") -> None:
    """Save results to JSON file.

    Args:
        results: Dict containing benchmark metadata and per-model result entries.
        output_path: Destination path for the JSON output file.
        prefix: Optional log-line prefix (e.g. emoji) prepended to the saved-path message.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    if prefix:
        logger.info(f"{prefix} Results saved to: {output_path}")
    else:
        logger.info(f"✅ Results saved to: {output_path}")


# =============================================================================
# MLX Training Benchmark
# =============================================================================


def _validation_map_from_metrics(val_metrics: dict[str, Any]) -> tuple[float, float]:
    """Prefer keypoint (pose) mAP when present; otherwise use box mAP."""
    if "mAP50_pose" in val_metrics or "mAP50-95_pose" in val_metrics:
        map50 = float(val_metrics.get("mAP50_pose", val_metrics.get("mAP50", 0.0)))
        map50_95 = float(val_metrics.get("mAP50-95_pose", val_metrics.get("mAP50-95", 0.0)))
    else:
        map50 = float(val_metrics.get("mAP50", 0.0))
        map50_95 = float(val_metrics.get("mAP50-95", 0.0))
    return map50, map50_95


def train_model_mlx(
    model_size: str,
    data_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer_choice: str = DEFAULT_OPTIMIZER,
) -> dict[str, Any] | None:
    """Train YOLO26 pose model using native MLX and measure time.

    Uses the pure MLX implementation of YOLO26 for training with the
    MLX-native trainer and data loader.

    Args:
        model_size: Model size (n, s, m, l, x)
        data_path: Path to dataset YAML config
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        optimizer_choice: ``"auto"`` (default; mirrors Ultralytics' AdamW vs
            MuSGD selection), ``"adamw"``, or ``"musgd"``.

    Returns:
        Training results dict or None if failed
    """

    model_name = f"yolo26{model_size}-pose"
    weights_file = MODELS_DIR / f"yolo26{model_size}-pose.npz"
    if not weights_file.exists():
        weights_file = MODELS_DIR / f"yolo26{model_size}-pose.safetensors"
    if not weights_file.exists():
        logger.warning(f"  ⚠️  MLX weights not found for {model_name} (tried .npz and .safetensors)")
        logger.warning(f"  Please run: python convert_weights.py --models {model_size}")
        return None

    try:
        from yolo26mlx import YOLO
        from yolo26mlx.engine.trainer import Trainer
    except ImportError as e:
        logger.warning(f"  ⚠️  YOLO26 MLX not available: {e}")
        return None

    logger.info(f"  Loading {model_name} from: {weights_file}")
    try:
        model = YOLO(str(weights_file), task="pose")
    except Exception as e:
        logger.error(f"  ⚠️  Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return None

    clear_mlx_memory()

    trainer = Trainer(model=model.model, task="pose")

    logger.info(
        f"  Training for {epochs} epochs (batch={batch_size}, lr={lr}, "
        f"optimizer={optimizer_choice})..."
    )
    logger.info("  Using pure MLX training with real COCO pose data")
    start_time = time.perf_counter()

    try:
        # ``val=False`` skips per-epoch validation so the benchmark
        # measures pure training throughput. Final mAP is computed via
        # the single ``trainer._validate(...)`` call after training.
        train_results = trainer(
            data=str(data_path),
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            patience=epochs + 1,
            save_period=-1,
            project=str(RESULTS_DIR / "mlx_runs"),
            name=model_name,
            exist_ok=True,
            lr=lr,
            optimizer=optimizer_choice,
            val=False,
            verbose=True,
        )
    except Exception as e:
        logger.error(f"  ⚠️  Training failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    training_time = time.perf_counter() - start_time

    _, peak_memory = get_mlx_memory()

    logger.info("  Running validation...")
    val_metrics: dict[str, Any] = {}
    try:
        model.model.eval()
        val_metrics = trainer._validate(batch_size, 640)
        map50, map50_95 = _validation_map_from_metrics(val_metrics)
    except Exception as e:
        logger.warning(f"  ⚠️  Validation failed: {e}")
        map50, map50_95 = 0.0, 0.0

    final_loss = train_results.get("final_loss", 0.0)

    map50_pose = float(val_metrics.get("mAP50_pose", 0.0))
    map5095_pose = float(val_metrics.get("mAP50-95_pose", 0.0))
    map50_box = float(val_metrics.get("mAP50_box", 0.0))
    map5095_box = float(val_metrics.get("mAP50-95_box", 0.0))

    result: dict[str, Any] = {
        "model": model_name,
        "task": "pose",
        "training_time_seconds": round(training_time, 2),
        "time_per_epoch_seconds": round(training_time / epochs, 2),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "final_loss": round(float(final_loss), 4),
        # Legacy headline values: keep ``mAP50``/``mAP50-95`` so older
        # downstream readers keep working. For pose these mirror the keypoint
        # values via ``Trainer._validate_pose``.
        "mAP50": round(float(map50), 4) if map50 else 0.0,
        "mAP50-95": round(float(map50_95), 4) if map50_95 else 0.0,
        "peak_memory_mb": round(peak_memory, 1),
        "framework": "MLX",
    }

    # Emit explicit pose/box keys when the pose validator returned them so the
    # collect-results script and chart generator can show apples-to-apples
    # keypoint mAP next to PyTorch MPS/CPU.
    if "mAP50_pose" in val_metrics or "mAP50-95_pose" in val_metrics:
        result["mAP50_pose"] = round(map50_pose, 4)
        result["mAP50-95_pose"] = round(map5095_pose, 4)
        result["mAP50_box"] = round(map50_box, 4)
        result["mAP50-95_box"] = round(map5095_box, 4)

    return result


# =============================================================================
# Main
# =============================================================================


def main():
    """Parse CLI args, set up dataset, train each model with native MLX, and save benchmark results."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="YOLO26 Pose MLX Training Benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_SIZES,
        choices=MODEL_SIZES,
        help="Model sizes to benchmark (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--optimizer",
        choices=["auto", "adamw", "musgd"],
        default=DEFAULT_OPTIMIZER,
        help=(
            "Optimizer choice. 'auto' (default) mirrors Ultralytics' "
            "optimizer='auto': AdamW for short fine-tune runs (iter <= 10000), "
            "MuSGD for long from-scratch runs."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "yolo26_pose_mlx_training_final.json",
        help="Output JSON path (default: results/yolo26_pose_mlx_training_final.json)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset YAML to train on (default: auto-setup COCO8-Pose). Use this to "
        "fine-tune on a larger pose subset and evaluate separately on COCO val2017.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the discarded 1-epoch warmup run (use for fine-tune accuracy runs, "
        "where throughput timing is not the goal).",
    )
    args = parser.parse_args()
    ensure_runtime_dirs(PROJECT_DIR)

    logger.info("=" * 70)
    logger.info("  YOLO26 Pose MLX Training Benchmark")
    logger.info("=" * 70)
    logger.info(f"  Models: {', '.join(args.models)}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info("=" * 70)
    logger.info("")

    try:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        logger.info(f"✅ MLX device: {mx.default_device()}")
    except ImportError:
        logger.error("❌ MLX not available. Please install: pip install mlx")
        sys.exit(1)

    device_info = get_device_info()
    logger.info(f"✅ Platform: {device_info.get('cpu', device_info.get('processor', 'Unknown'))}")

    if args.data:
        data_path = Path(args.data)
        logger.info("\n📦 Using dataset config: %s", data_path)
    else:
        logger.info("\n📦 Setting up COCO8-Pose dataset...")
        data_path = setup_coco8_pose()
    logger.info(f"✅ Dataset config: {data_path}")
    logger.info("")

    results = []

    progress_path = args.output.parent / (args.output.stem.replace("_final", "") + "_progress.json")

    # Warm up the backend once on a 1-epoch throwaway run so the one-time graph
    # build / kernel compilation cost is not charged to the first timed model.
    # COCO8-Pose is tiny (~20 iterations), so without this the first model's
    # per-epoch time is dominated by warmup and skews its speedup ratio.
    if args.no_warmup:
        logger.info("⏭️  Warmup skipped (--no-warmup).")
    else:
        logger.info("🔥 Warmup run (1 epoch, discarded)...")
        try:
            train_model_mlx(
                model_size=args.models[0],
                data_path=data_path,
                epochs=1,
                batch_size=args.batch,
                lr=args.lr,
                optimizer_choice=args.optimizer,
            )
        except Exception as e:
            logger.warning(f"  ⚠️  Warmup skipped: {e}")
    clear_mlx_memory()

    for i, size in enumerate(args.models):
        model_name = f"yolo26{size}-pose"
        logger.info(f"\n{'=' * 50}")
        logger.info(f"  [{i + 1}/{len(args.models)}] Training: {model_name}")
        logger.info(f"{'=' * 50}")

        clear_mlx_memory()

        result = train_model_mlx(
            model_size=size,
            data_path=data_path,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            optimizer_choice=args.optimizer,
        )

        if result:
            results.append(result)
            logger.info(f"\n  ✅ {model_name} completed:")
            logger.info(
                f"     Training time: {result['training_time_seconds']:.1f}s ({result['time_per_epoch_seconds']:.1f}s/epoch)"
            )
            logger.info(f"     mAP50: {result['mAP50']:.4f}")
            logger.info(f"     Peak memory: {result['peak_memory_mb']:.1f} MB")

            progress = {
                "benchmark": "YOLO26 Pose MLX Training (in progress)",
                "task": "pose",
                "timestamp": datetime.now().isoformat(),
                "device_info": device_info,
                "config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch,
                    "learning_rate": float(args.lr),
                    "learning_rate_source": _lr_source_label(float(args.lr)),
                    "optimizer": _optimizer_label(args.optimizer),
                    "dataset": str(data_path),
                },
                "results": results,
            }
            save_results(progress, progress_path, prefix="📝")
        else:
            logger.warning(f"\n  ❌ {model_name} failed")

    logger.info("\n" + "=" * 70)
    logger.info("  Training Benchmark Summary")
    logger.info("=" * 70)

    logger.info(
        f"\n{'Model':<18} {'Time (s)':<12} {'Time/Epoch':<12} {'mAP50':<10} {'Memory (MB)':<12}"
    )
    logger.info("-" * 64)

    results_by_model = {r["model"]: r for r in results}

    for size in args.models:
        model_name = f"yolo26{size}-pose"
        if model_name in results_by_model:
            r = results_by_model[model_name]
            logger.info(
                f"{model_name:<18} {r['training_time_seconds']:<12.1f} {r['time_per_epoch_seconds']:<12.1f} {r['mAP50']:<10.4f} {r['peak_memory_mb']:<12.1f}"
            )
        else:
            logger.warning(f"{model_name:<18} {'FAILED':<12} {'-':<12} {'-':<10} {'-':<12}")

    logger.info("-" * 64)
    logger.info("")

    lr_used = float(args.lr)

    final_output = {
        "benchmark": "YOLO26 Pose MLX Training (Pure MLX)",
        "task": "pose",
        "timestamp": datetime.now().isoformat(),
        "device_info": device_info,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": lr_used,
            "learning_rate_source": _lr_source_label(lr_used),
            "optimizer": _optimizer_label(args.optimizer),
            "dataset": str(data_path) if args.data else "COCO8-Pose",
            "framework": "MLX (native)",
        },
        "results": results,
    }

    save_results(final_output, args.output)

    logger.info("\n🎉 MLX Training benchmark complete!")
    logger.info(f"   Results: {args.output}")


if __name__ == "__main__":
    main()
