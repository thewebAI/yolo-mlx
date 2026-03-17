#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 Training Benchmark (Pure MLX)
=====================================
Measures training performance for YOLO26 models using native MLX on Apple Silicon.

This benchmark uses the pure MLX implementation of YOLO26 for training,
providing accurate measurements of MLX-native training performance.

Usage:
    python benchmark_yolo26_training_mlx.py
    python benchmark_yolo26_training_mlx.py --models n s      # Specific models only
    python benchmark_yolo26_training_mlx.py --epochs 5        # Fewer epochs
    python benchmark_yolo26_training_mlx.py --batch 2         # Smaller batch size
    python benchmark_yolo26_training_mlx.py --output custom.json

Output:
    ../results/yolo26_mlx_training_final.json (default, overridable via --output)
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
LEARNING_RATE = 0.000119  # auto-computed by MuSGD for nc=80
MODEL_SIZES = ["n", "s", "m", "l", "x"]

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR / ".."
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"
DATASETS_DIR = PROJECT_DIR / "datasets"


# =============================================================================
# Utility Functions
# =============================================================================


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


def setup_coco128() -> Path:
    """Download and setup COCO128 dataset.

    Returns:
        Path to local YAML config file
    """
    # COCO128 will be auto-downloaded by ultralytics when first used
    # Create a local config that points to the dataset

    # Check if ultralytics has already downloaded it
    ultralytics_datasets = Path.home() / ".config" / "Ultralytics" / "datasets"
    coco128_path = ultralytics_datasets / "coco128"

    # Also check common locations
    alt_paths = [
        DATASETS_DIR / "coco128",
        Path("datasets") / "coco128",
        Path.cwd() / "coco128",
    ]

    dataset_path = None
    for path in [coco128_path] + alt_paths:
        if path.exists() and (path / "images").exists():
            dataset_path = path
            break

    if dataset_path is None:
        # Download COCO128 automatically (~7 MB)
        logger.info("  COCO128 not found locally. Downloading...")
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DATASETS_DIR / "coco128.zip"
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
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
            dataset_path = DATASETS_DIR / "coco128"
            if not (dataset_path / "images").exists():
                raise RuntimeError("Extracted archive missing coco128/images/ directory")
            logger.info(f"  Downloaded COCO128 to: {dataset_path}")
        except Exception as e:
            logger.error(f"  ERROR: Failed to download COCO128: {e}")
            logger.warning("  Training will fall back to synthetic data.")
            if zip_path.exists():
                zip_path.unlink()
            return Path("coco128.yaml")

    logger.info(f"  Found COCO128 at: {dataset_path}")

    # Create local config with absolute paths
    local_yaml = DATASETS_DIR / "coco128_local.yaml"
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    config = f"""# COCO128 Local Configuration
path: {dataset_path.absolute()}
train: images/train2017
val: images/train2017
test:

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
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


def train_model_mlx(
    model_size: str,
    data_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any] | None:
    """Train YOLO26 model using native MLX and measure time.

    Uses the pure MLX implementation of YOLO26 for training with the
    MLX-native trainer and data loader.

    Args:
        model_size: Model size (n, s, m, l, x)
        data_path: Path to dataset YAML config
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate

    Returns:
        Training results dict or None if failed
    """

    model_name = f"yolo26{model_size}"
    weights_file = MODELS_DIR / f"{model_name}.npz"

    # Check for converted MLX weights
    if not weights_file.exists():
        logger.warning(f"  ⚠️  MLX weights not found: {weights_file}")
        logger.warning(f"  Please run: python convert_weights.py --models {model_size}")
        return None

    # Import YOLO26 MLX model
    try:
        from yolo26mlx import YOLO
        from yolo26mlx.engine.trainer import Trainer
    except ImportError as e:
        logger.warning(f"  ⚠️  YOLO26 MLX not available: {e}")
        return None

    # Load model
    logger.info(f"  Loading {model_name} from: {weights_file}")
    try:
        model = YOLO(str(weights_file))
    except Exception as e:
        logger.error(f"  ⚠️  Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return None

    # Reset memory stats before training
    clear_mlx_memory()

    # Create trainer
    trainer = Trainer(model=model.model, task="detect")

    # Train using MLX-native trainer
    logger.info(f"  Training for {epochs} epochs (batch={batch_size}, lr={lr})...")
    logger.info("  Using pure MLX training with real COCO data")
    start_time = time.perf_counter()

    try:
        train_results = trainer(
            data=str(data_path),
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            patience=epochs + 1,  # Disable early stopping
            save_period=-1,  # Don't save intermediate checkpoints
            project=str(RESULTS_DIR / "mlx_runs"),
            name=model_name,
            exist_ok=True,
            val=False,  # Disable validation during training (for fair timing comparison)
            verbose=True,  # Show per-batch training progress
        )
    except Exception as e:
        logger.error(f"  ⚠️  Training failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    training_time = time.perf_counter() - start_time

    # Get memory usage
    _, peak_memory = get_mlx_memory()

    # Run validation separately (same as MPS/CPU scripts)
    logger.info("  Running validation...")
    try:
        model.model.eval()
        val_metrics = trainer._validate(batch_size, 640)
        map50 = val_metrics.get("mAP50", 0.0)
        map50_95 = val_metrics.get("mAP50-95", 0.0)
    except Exception as e:
        logger.warning(f"  ⚠️  Validation failed: {e}")
        map50, map50_95 = 0.0, 0.0

    final_loss = train_results.get("final_loss", 0.0)

    return {
        "model": model_name,
        "training_time_seconds": round(training_time, 2),
        "time_per_epoch_seconds": round(training_time / epochs, 2),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "final_loss": round(float(final_loss), 4),
        "mAP50": round(float(map50), 4) if map50 else 0.0,
        "mAP50-95": round(float(map50_95), 4) if map50_95 else 0.0,
        "peak_memory_mb": round(peak_memory, 1),
        "framework": "MLX",
    }


# =============================================================================
# Main
# =============================================================================


def main():
    """Parse CLI args, set up dataset, train each model with native MLX, and save benchmark results."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="YOLO26 MLX Training Benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_SIZES,  # All models: n, s, m, l, x
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
        "--output",
        type=Path,
        default=RESULTS_DIR / "yolo26_mlx_training_final.json",
        help="Output JSON path (default: results/yolo26_mlx_training_final.json)",
    )
    args = parser.parse_args()
    ensure_runtime_dirs(PROJECT_DIR)

    logger.info("=" * 70)
    logger.info("  YOLO26 MLX Training Benchmark")
    logger.info("=" * 70)
    logger.info(f"  Models: {', '.join(args.models)}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info("=" * 70)
    logger.info("")

    # Verify MLX is available
    try:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        logger.info(f"✅ MLX device: {mx.default_device()}")
    except ImportError:
        logger.error("❌ MLX not available. Please install: pip install mlx")
        sys.exit(1)

    # Get device info
    device_info = get_device_info()
    logger.info(f"✅ Platform: {device_info.get('cpu', device_info.get('processor', 'Unknown'))}")

    # Setup dataset
    logger.info("\n📦 Setting up COCO128 dataset...")
    data_path = setup_coco128()
    logger.info(f"✅ Dataset config: {data_path}")
    logger.info("")

    # Results storage
    results = []

    # Progress file for intermediate saves (derived from output path)
    progress_path = args.output.parent / (args.output.stem.replace("_final", "") + "_progress.json")

    # Run benchmarks for each model size
    for i, size in enumerate(args.models):
        model_name = f"yolo26{size}"
        logger.info(f"\n{'='*50}")
        logger.info(f"  [{i+1}/{len(args.models)}] Training: {model_name}")
        logger.info(f"{'='*50}")

        # Clear memory before each run
        clear_mlx_memory()

        # Train and measure
        result = train_model_mlx(
            model_size=size,
            data_path=data_path,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
        )

        if result:
            results.append(result)
            logger.info(f"\n  ✅ {model_name} completed:")
            logger.info(
                f"     Training time: {result['training_time_seconds']:.1f}s ({result['time_per_epoch_seconds']:.1f}s/epoch)"
            )
            logger.info(f"     mAP50: {result['mAP50']:.4f}")
            logger.info(f"     Peak memory: {result['peak_memory_mb']:.1f} MB")

            # Save progress after each model
            progress = {
                "benchmark": "YOLO26 MLX Training (in progress)",
                "timestamp": datetime.now().isoformat(),
                "device_info": device_info,
                "config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch,
                    "learning_rate": args.lr,
                    "dataset": str(data_path),
                },
                "results": results,
            }
            save_results(progress, progress_path, prefix="📝")
        else:
            logger.warning(f"\n  ❌ {model_name} failed")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("  Training Benchmark Summary")
    logger.info("=" * 70)

    logger.info(
        f"\n{'Model':<12} {'Time (s)':<12} {'Time/Epoch':<12} {'mAP50':<10} {'Memory (MB)':<12}"
    )
    logger.info("-" * 58)

    # Create lookup dict from results list
    results_by_model = {r["model"]: r for r in results}

    for size in args.models:
        model_name = f"yolo26{size}"
        if model_name in results_by_model:
            r = results_by_model[model_name]
            logger.info(
                f"{model_name:<12} {r['training_time_seconds']:<12.1f} {r['time_per_epoch_seconds']:<12.1f} {r['mAP50']:<10.4f} {r['peak_memory_mb']:<12.1f}"
            )
        else:
            logger.warning(f"{model_name:<12} {'FAILED':<12} {'-':<12} {'-':<10} {'-':<12}")

    logger.info("-" * 58)
    logger.info("")

    # Save final results
    final_output = {
        "benchmark": "YOLO26 MLX Training (Pure MLX)",
        "timestamp": datetime.now().isoformat(),
        "device_info": device_info,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": "auto (MuSGD auto_lr)",
            "optimizer": "MuSGD (Muon + Nesterov SGD)",
            "dataset": "COCO128",
            "framework": "MLX (native)",
        },
        "results": results,
    }

    save_results(final_output, args.output)

    logger.info("\n🎉 MLX Training benchmark complete!")
    logger.info(f"   Results: {args.output}")


if __name__ == "__main__":
    main()
