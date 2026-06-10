#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 Pose PyTorch CPU Training Benchmark
==========================================
Measures pose training performance using PyTorch CPU backend.

Usage:
    python benchmark_yolo26_pose_training_cpu.py
    python benchmark_yolo26_pose_training_cpu.py --models n s      # Specific models only
    python benchmark_yolo26_pose_training_cpu.py --epochs 5        # Fewer epochs
    python benchmark_yolo26_pose_training_cpu.py --batch 2         # Smaller batch size
    python benchmark_yolo26_pose_training_cpu.py --output custom.json

Output:
    ../results/yolo26_pose_cpu_training_final.json (default, overridable via --output)
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

DATA_CONFIG = "coco8-pose.yaml"

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR / ".."
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"


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

    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass

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
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return usage.ru_maxrss / 1024 / 1024
        else:
            return usage.ru_maxrss / 1024
    except Exception:
        return 0.0


def save_progress(results: list[dict], output_file: Path):
    """Save intermediate progress to file.

    Args:
        results: List of benchmark result dicts accumulated so far.
        output_file: Path to the JSON file where progress is written.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def _extract_val_metrics(val_results: Any) -> tuple[float, float, float, float]:
    """Read keypoint and box mAP from validation results; keypoint zeros if pose unavailable."""
    map50_pose = map50_95_pose = 0.0
    map50_box = map50_95_box = 0.0
    try:
        if hasattr(val_results, "box"):
            map50_box = float(val_results.box.map50)
            map50_95_box = float(val_results.box.map)
        else:
            map50_box = float(getattr(val_results, "map50", 0) or 0)
            map50_95_box = float(getattr(val_results, "map", 0) or 0)
    except Exception:
        map50_box = map50_95_box = 0.0
    try:
        if hasattr(val_results, "pose"):
            map50_pose = float(val_results.pose.map50)
            map50_95_pose = float(val_results.pose.map)
    except Exception:
        pass
    return map50_pose, map50_95_pose, map50_box, map50_95_box


def train_model_cpu(
    model_size: str,
    data_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any] | None:
    """Train single pose PyTorch model with CPU backend and measure time.

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

        device = "cpu"
        num_threads = torch.get_num_threads()
    except ImportError as e:
        logger.warning(f"  ⚠️  PyTorch not available: {e}")
        return None

    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.warning(f"  ⚠️  Ultralytics not available: {e}")
        return None

    model_name = f"yolo26{model_size}-pose"
    model_file = f"yolo26{model_size}-pose.pt"

    local_weights = MODELS_DIR / model_file
    if local_weights.exists():
        model_source = str(local_weights)
        logger.info(f"  Loading {model_name} from local weights: {local_weights}")
    else:
        model_source = model_file
        logger.info(f"  Loading {model_name} (will download if not cached)...")

    try:
        model = YOLO(model_source)
    except Exception as e:
        logger.error(f"  ⚠️  Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return None

    clear_memory()

    logger.info(
        f"  Training for {epochs} epochs (batch={batch_size}, lr={lr}, device={device}, threads={num_threads})..."
    )
    start_time = time.perf_counter()

    # ``val=False`` skips per-epoch validation so the benchmark
    # measures pure training throughput. Final mAP is computed via
    # the explicit ``model.val(...)`` call after training.
    try:
        model.train(
            data=str(data_path),
            task="pose",
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            lr0=lr,
            patience=epochs + 1,
            save_period=-1,
            workers=4,
            device=device,
            project=str(RESULTS_DIR / "cpu_runs"),
            name=model_name,
            exist_ok=True,
            verbose=True,
            val=False,
        )
    except Exception as e:
        logger.error(f"  ⚠️  Training failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    training_time = time.perf_counter() - start_time

    peak_memory = get_process_memory()

    logger.info("  Running validation...")
    map50_pose = map50_95_pose = map50_box = map50_95_box = 0.0
    try:
        val_results = model.val(data=str(data_path), batch=batch_size, device=device)
        map50_pose, map50_95_pose, map50_box, map50_95_box = _extract_val_metrics(val_results)
    except Exception as e:
        logger.warning(f"  ⚠️  Validation failed: {e}")

    return {
        "model": model_name,
        "training_time_seconds": round(training_time, 2),
        "time_per_epoch_seconds": round(training_time / epochs, 2),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "mAP50_pose": round(map50_pose, 4) if map50_pose else 0.0,
        "mAP50-95_pose": round(map50_95_pose, 4) if map50_95_pose else 0.0,
        "mAP50_box": round(map50_box, 4) if map50_box else 0.0,
        "mAP50-95_box": round(map50_95_box, 4) if map50_95_box else 0.0,
        "mAP50": round(map50_pose, 4) if map50_pose else 0.0,
        "mAP50-95": round(map50_95_pose, 4) if map50_95_pose else 0.0,
        "peak_memory_mb": round(peak_memory, 1),
        "num_threads": num_threads,
        "device": "cpu",
    }


def main():
    """Parse CLI args, set up dataset, train each pose model on CPU, and save benchmark results."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="YOLO26 Pose PyTorch CPU Training Benchmark")
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
        "--threads",
        type=int,
        default=None,
        help="Number of CPU threads (default: PyTorch default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "yolo26_pose_cpu_training_final.json",
        help="Output JSON path (default: results/yolo26_pose_cpu_training_final.json)",
    )
    args = parser.parse_args()
    ensure_runtime_dirs(PROJECT_DIR)

    logger.info("=" * 70)
    logger.info("YOLO26 Pose PyTorch CPU Training Benchmark")
    logger.info("=" * 70)

    if args.threads is not None:
        try:
            import torch

            torch.set_num_threads(args.threads)
            logger.info(f"\n🔧 Set PyTorch threads to: {args.threads}")
        except Exception as e:
            logger.warning(f"\n⚠️  Failed to set thread count: {e}")

    logger.info("\n💻 Device Information:")
    device_info = get_device_info()
    for key, value in device_info.items():
        logger.info(f"   {key}: {value}")

    logger.info("\n📦 Using COCO8-Pose dataset...")
    data_path = DATA_CONFIG
    logger.info(f"   Using: {data_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_file = args.output.parent / (args.output.stem.replace("_final", "") + "_progress.json")
    final_file = args.output

    logger.info(f"\n🏃 Running training benchmarks for models: {args.models}")
    logger.info(f"   Epochs: {args.epochs}, Batch: {args.batch}, LR: {args.lr}")
    logger.info("-" * 70)

    all_results = []

    # Warm up the CPU backend once on a 1-epoch throwaway run so one-time
    # process/thread/dataloader initialization is not charged to the first
    # timed model. COCO8-Pose is tiny (~20 iterations), so without this the
    # first model's per-epoch time is dominated by warmup and skews its speedup.
    logger.info("\n🔥 Warmup run (1 epoch, discarded)...")
    try:
        train_model_cpu(
            model_size=args.models[0],
            data_path=data_path,
            epochs=1,
            batch_size=args.batch,
            lr=args.lr,
        )
    except Exception as e:
        logger.warning(f"  ⚠️  Warmup skipped: {e}")

    for i, model_size in enumerate(args.models, 1):
        logger.info(f"\n[{i}/{len(args.models)}] Benchmarking yolo26{model_size}-pose...")

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
            logger.info(
                f"     pose mAP50: {result['mAP50_pose']:.4f}, mAP50-95: {result['mAP50-95_pose']:.4f} | "
                f"box mAP50: {result['mAP50_box']:.4f}, mAP50-95: {result['mAP50-95_box']:.4f}"
            )

            save_progress(all_results, progress_file)
        else:
            logger.error("  ❌ Failed")

        clear_memory()

    logger.info("\n" + "=" * 70)
    logger.info("📊 Final Results")
    logger.info("=" * 70)

    final_output = {
        "benchmark": "yolo26_pose_cpu_training",
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

    if all_results:
        logger.info("\n" + "-" * 120)
        logger.info(
            f"{'Model':<16} {'Time (s)':<10} {'s/epoch':<10} "
            f"{'mAP50 pose':<12} {'mAP50-95 pose':<14} {'mAP50 box':<12} {'mAP50-95 box':<14}"
        )
        logger.info("-" * 120)
        for r in all_results:
            logger.info(
                f"{r['model']:<16} "
                f"{r['training_time_seconds']:<10.1f} "
                f"{r['time_per_epoch_seconds']:<10.2f} "
                f"{r['mAP50_pose']:<12.4f} "
                f"{r['mAP50-95_pose']:<14.4f} "
                f"{r['mAP50_box']:<12.4f} "
                f"{r['mAP50-95_box']:<14.4f}"
            )
        logger.info("-" * 120)

    logger.info("\n✨ PyTorch CPU pose training benchmark complete!")


if __name__ == "__main__":
    main()
