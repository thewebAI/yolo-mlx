#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 PyTorch CPU Training Benchmark
======================================
Measures training performance using PyTorch CPU backend.

Usage:
    python benchmark_yolo26_training_cpu.py
    python benchmark_yolo26_training_cpu.py --models n s      # Specific models only
    python benchmark_yolo26_training_cpu.py --epochs 5        # Fewer epochs
    python benchmark_yolo26_training_cpu.py --batch 2         # Smaller batch size
    python benchmark_yolo26_training_cpu.py --output custom.json

Output:
    ../results/yolo26_cpu_training_final.json (default, overridable via --output)
"""

import argparse
import gc
import json
import logging
import platform
import resource
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
LEARNING_RATE = 0.00001
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
        Dict with platform, processor, Python version, CPU name, core count, and PyTorch version.
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

    # Try to get CPU core count
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass

    # Get PyTorch info
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["num_threads"] = torch.get_num_threads()
    except ImportError:
        info["torch_version"] = "not installed"

    return info


def clear_memory():
    """Force a garbage collection cycle to free unreferenced objects between benchmark runs."""
    gc.collect()


def get_process_memory() -> float:
    """Get peak process memory usage in MB.

    Returns:
        Peak RSS (Resident Set Size) memory in MB

    Note: ru_maxrss returns maximum RSS during process lifetime,
          which is effectively peak memory usage.
    """
    try:
        # Use resource module for cross-platform memory tracking
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in bytes on macOS, KB on Linux
        if sys.platform == "darwin":
            return usage.ru_maxrss / 1024 / 1024  # bytes to MB
        else:
            return usage.ru_maxrss / 1024  # KB to MB
    except Exception:
        return 0.0


def setup_coco128() -> Path:
    """Download and setup COCO128 dataset.

    Returns:
        Path to local YAML config file or standard coco128.yaml
    """
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


def save_progress(results: list[dict], output_file: Path):
    """Save intermediate progress to file.

    Args:
        results: List of benchmark result dicts accumulated so far.
        output_file: Path to the JSON file where progress is written.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def train_model_cpu(
    model_size: str,
    data_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any] | None:
    """Train single PyTorch model with CPU backend and measure time.

    Args:
        model_size: Model size (n, s, m, l, x)
        data_path: Path to dataset YAML config
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate

    Returns:
        Training results dict or None if failed
    """
    try:
        import torch

        # Explicitly set device to CPU
        device = "cpu"
        num_threads = torch.get_num_threads()
    except ImportError as e:
        logger.warning(f"  ⚠️  PyTorch not available: {e}")
        return None

    # Import ultralytics YOLO
    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.warning(f"  ⚠️  Ultralytics not available: {e}")
        return None

    # PyTorch ultralytics uses model names like "yolo26n.pt" which encode the scale
    # The model will be auto-downloaded if not present locally
    model_name = f"yolo26{model_size}"
    model_file = f"{model_name}.pt"

    # Check for local weights first
    local_weights = MODELS_DIR / model_file
    if local_weights.exists():
        model_source = str(local_weights)
        logger.info(f"  Loading {model_name} from local weights: {local_weights}")
    else:
        # Will trigger auto-download from ultralytics hub
        model_source = model_file
        logger.info(f"  Loading {model_name} (will download if not cached)...")

    try:
        model = YOLO(model_source)
    except Exception as e:
        logger.error(f"  ⚠️  Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return None

    # Clear memory before training
    clear_memory()

    # Train with CPU device
    logger.info(
        f"  Training for {epochs} epochs (batch={batch_size}, lr={lr}, device={device}, threads={num_threads})..."
    )
    start_time = time.perf_counter()

    try:
        model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            lr0=lr,  # PyTorch ultralytics uses lr0 for initial learning rate
            patience=epochs + 1,  # Disable early stopping
            save_period=-1,  # Don't save intermediate checkpoints
            workers=4,
            device=device,  # Use CPU
            project=str(RESULTS_DIR / "cpu_runs"),
            name=model_name,
            exist_ok=True,
            verbose=True,  # Show epoch-by-epoch progress
            val=False,  # Disable validation during training (for fair timing)
        )
    except Exception as e:
        logger.error(f"  ⚠️  Training failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    training_time = time.perf_counter() - start_time

    # Get peak memory usage (ru_maxrss is already peak)
    peak_memory = get_process_memory()

    # Run validation separately
    logger.info("  Running validation...")
    try:
        val_results = model.val(data=str(data_path), batch=batch_size, device=device)
        # PyTorch ultralytics validation returns metrics object
        # Access via val_results.box.map (mAP50-95) and val_results.box.map50 (mAP50)
        if hasattr(val_results, "box"):
            map50 = val_results.box.map50
            map50_95 = val_results.box.map
        else:
            # Fallback for different result formats
            map50 = getattr(val_results, "map50", 0)
            map50_95 = getattr(val_results, "map", 0)
    except Exception as e:
        logger.warning(f"  ⚠️  Validation failed: {e}")
        map50, map50_95 = 0.0, 0.0

    return {
        "model": model_name,
        "training_time_seconds": round(training_time, 2),
        "time_per_epoch_seconds": round(training_time / epochs, 2),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "mAP50": round(float(map50), 4) if map50 else 0.0,
        "mAP50-95": round(float(map50_95), 4) if map50_95 else 0.0,
        "peak_memory_mb": round(peak_memory, 1),
        "num_threads": num_threads,
        "device": "cpu",
    }


# =============================================================================
# Main
# =============================================================================


def main():
    """Parse CLI args, set up dataset, train each model on CPU, and save benchmark results."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="YOLO26 PyTorch CPU Training Benchmark")
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
        "--threads",
        type=int,
        default=None,
        help="Number of CPU threads (default: PyTorch default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "yolo26_cpu_training_final.json",
        help="Output JSON path (default: results/yolo26_cpu_training_final.json)",
    )
    args = parser.parse_args()
    ensure_runtime_dirs(PROJECT_DIR)

    logger.info("=" * 70)
    logger.info("YOLO26 PyTorch CPU Training Benchmark")
    logger.info("=" * 70)

    # Set thread count if specified
    if args.threads is not None:
        try:
            import torch

            torch.set_num_threads(args.threads)
            logger.info(f"\n🔧 Set PyTorch threads to: {args.threads}")
        except Exception as e:
            logger.warning(f"\n⚠️  Failed to set thread count: {e}")

    # Get device info
    logger.info("\n💻 Device Information:")
    device_info = get_device_info()
    for key, value in device_info.items():
        logger.info(f"   {key}: {value}")

    # Setup dataset
    logger.info("\n📦 Setting up COCO128 dataset...")
    data_path = setup_coco128()
    logger.info(f"   Using: {data_path}")

    # Setup results directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_file = args.output.parent / (args.output.stem.replace("_final", "") + "_progress.json")
    final_file = args.output

    # Run benchmarks
    logger.info(f"\n🏃 Running training benchmarks for models: {args.models}")
    logger.info(f"   Epochs: {args.epochs}, Batch: {args.batch}, LR: {args.lr}")
    logger.info("-" * 70)

    all_results = []

    for i, model_size in enumerate(args.models, 1):
        logger.info(f"\n[{i}/{len(args.models)}] Benchmarking yolo26{model_size}...")

        result = train_model_cpu(
            model_size=model_size,
            data_path=data_path,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
        )

        if result:
            all_results.append(result)
            logger.info(f"  ✅ Completed in {result['training_time_seconds']:.1f}s")
            logger.info(f"     mAP50: {result['mAP50']:.4f}, mAP50-95: {result['mAP50-95']:.4f}")

            # Save progress after each model
            save_progress(all_results, progress_file)
        else:
            logger.error("  ❌ Failed")

        # Clear memory between models
        clear_memory()

    # Save final results
    logger.info("\n" + "=" * 70)
    logger.info("📊 Final Results")
    logger.info("=" * 70)

    final_output = {
        "benchmark": "yolo26_cpu_training",
        "timestamp": datetime.now().isoformat(),
        "device_info": device_info,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": args.lr,
            "device": "cpu",
        },
        "results": all_results,
    }

    with open(final_file, "w") as f:
        json.dump(final_output, f, indent=2)

    logger.info(f"\n✅ Results saved to: {final_file}")

    # Print summary table
    if all_results:
        logger.info("\n" + "-" * 70)
        logger.info(
            f"{'Model':<12} {'Time (s)':<12} {'s/epoch':<12} {'mAP50':<12} {'mAP50-95':<12}"
        )
        logger.info("-" * 70)
        for r in all_results:
            logger.info(
                f"{r['model']:<12} "
                f"{r['training_time_seconds']:<12.1f} "
                f"{r['time_per_epoch_seconds']:<12.2f} "
                f"{r['mAP50']:<12.4f} "
                f"{r['mAP50-95']:<12.4f}"
            )
        logger.info("-" * 70)

    logger.info("\n✨ PyTorch CPU training benchmark complete!")


if __name__ == "__main__":
    main()
