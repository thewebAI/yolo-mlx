#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 Benchmark Results Collector
===================================
Collects results from all benchmark runs and combines them into a unified format.

This script reads the output JSON files from:
- benchmark_yolo26_inference.py       -> yolo26_inference_three_way.json
- benchmark_yolo26_training_mlx.py    -> yolo26_mlx_training_final.json
- benchmark_yolo26_training_mps.py    -> yolo26_mps_training_final.json
- benchmark_yolo26_training_cpu.py    -> yolo26_cpu_training_final.json

And produces a combined report for analysis and chart generation.

Usage:
    python benchmark_yolo26_collect_results.py
    python benchmark_yolo26_collect_results.py --output custom_output.json

Output:
    ../results/yolo26_benchmark_combined.json
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from _runtime_dirs import ensure_runtime_dirs

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR / ".."
RESULTS_DIR = PROJECT_DIR / "results"

# Expected input files
INPUT_FILES = {
    "inference": RESULTS_DIR / "yolo26_inference_three_way.json",
    "training_mlx": RESULTS_DIR / "yolo26_mlx_training_final.json",
    "training_mps": RESULTS_DIR / "yolo26_mps_training_final.json",
    "training_cpu": RESULTS_DIR / "yolo26_cpu_training_final.json",
}

MODEL_SIZES = ["n", "s", "m", "l", "x"]


# =============================================================================
# Data Loading
# =============================================================================


def load_json_file(path: Path) -> dict | None:
    """Load JSON file if it exists.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data or None if file doesn't exist
    """
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"  ⚠️  Error loading {path.name}: {e}")
        return None


def load_all_results() -> dict[str, dict | None]:
    """Load all benchmark result files.

    Returns:
        Dictionary mapping benchmark name to loaded data (or None if missing)
    """
    results = {}
    for name, path in INPUT_FILES.items():
        data = load_json_file(path)
        if data:
            logger.info(f"  ✅ Loaded: {path.name}")
        else:
            logger.warning(f"  ⚠️  Missing: {path.name}")
        results[name] = data
    return results


# =============================================================================
# Data Processing
# =============================================================================


def extract_inference_data(data: dict | None) -> dict:
    """Extract inference benchmark data into normalized format.

    Args:
        data: Raw inference benchmark data

    Returns:
        Normalized inference data by model size and backend
    """
    if data is None:
        return {}

    result = {}
    raw_results = data.get("results", {})

    for backend in ["mlx", "pytorch_mps", "pytorch_cpu"]:
        backend_data = raw_results.get(backend, {})
        for size in MODEL_SIZES:
            model_key = f"yolo26{size}"
            if size in backend_data:
                if model_key not in result:
                    result[model_key] = {}

                size_data = backend_data[size]
                result[model_key][backend] = {
                    "mean_ms": size_data.get("mean_ms", 0),
                    "std_ms": size_data.get("std_ms", 0),
                    "fps": size_data.get("fps", 0),
                    "min_ms": size_data.get("min_ms", 0),
                    "max_ms": size_data.get("max_ms", 0),
                }

    return result


def extract_training_data(
    mlx_data: dict | None,
    mps_data: dict | None,
    cpu_data: dict | None,
) -> dict:
    """Extract training benchmark data into normalized format.

    Args:
        mlx_data: Raw MLX training benchmark data
        mps_data: Raw MPS training benchmark data
        cpu_data: Raw CPU training benchmark data

    Returns:
        Normalized training data by model size and backend
    """
    result = {}

    # Process each backend
    backends = [
        ("mlx", mlx_data),
        ("pytorch_mps", mps_data),
        ("pytorch_cpu", cpu_data),
    ]

    for backend_name, data in backends:
        if data is None:
            continue

        for item in data.get("results", []):
            model_name = item.get("model", "")
            if not model_name:
                continue

            if model_name not in result:
                result[model_name] = {}

            result[model_name][backend_name] = {
                "training_time_seconds": item.get("training_time_seconds", 0),
                "time_per_epoch_seconds": item.get("time_per_epoch_seconds", 0),
                "epochs": item.get("epochs", 0),
                "batch_size": item.get("batch_size", 0),
                "learning_rate": item.get("learning_rate", 0),
                "mAP50": item.get("mAP50", 0),
                "mAP50-95": item.get("mAP50-95", 0),
            }

            # Include memory info if available
            if "peak_memory_mb" in item:
                result[model_name][backend_name]["peak_memory_mb"] = item["peak_memory_mb"]
            if "current_memory_mb" in item:
                result[model_name][backend_name]["current_memory_mb"] = item["current_memory_mb"]
            if "driver_memory_mb" in item:
                result[model_name][backend_name]["driver_memory_mb"] = item["driver_memory_mb"]

    return result


def calculate_speedups(inference_data: dict, training_data: dict) -> dict:
    """Calculate speedup ratios between backends.

    Args:
        inference_data: Normalized inference data
        training_data: Normalized training data

    Returns:
        Speedup ratios for inference and training
    """
    speedups = {
        "inference": {},
        "training": {},
    }

    # Inference speedups
    for model_name, backends in inference_data.items():
        mlx_ms = backends.get("mlx", {}).get("mean_ms", 0)
        mps_ms = backends.get("pytorch_mps", {}).get("mean_ms", 0)
        cpu_ms = backends.get("pytorch_cpu", {}).get("mean_ms", 0)

        model_speedups = {}

        if mlx_ms > 0 and cpu_ms > 0:
            model_speedups["mlx_vs_cpu"] = round(cpu_ms / mlx_ms, 2)
        if mlx_ms > 0 and mps_ms > 0:
            model_speedups["mlx_vs_mps"] = round(mps_ms / mlx_ms, 2)
        if mps_ms > 0 and cpu_ms > 0:
            model_speedups["mps_vs_cpu"] = round(cpu_ms / mps_ms, 2)

        if model_speedups:
            speedups["inference"][model_name] = model_speedups

    # Training speedups
    for model_name, backends in training_data.items():
        mlx_time = backends.get("mlx", {}).get("training_time_seconds", 0)
        mps_time = backends.get("pytorch_mps", {}).get("training_time_seconds", 0)
        cpu_time = backends.get("pytorch_cpu", {}).get("training_time_seconds", 0)

        model_speedups = {}

        if mlx_time > 0 and cpu_time > 0:
            model_speedups["mlx_vs_cpu"] = round(cpu_time / mlx_time, 2)
        if mlx_time > 0 and mps_time > 0:
            model_speedups["mlx_vs_mps"] = round(mps_time / mlx_time, 2)
        if mps_time > 0 and cpu_time > 0:
            model_speedups["mps_vs_cpu"] = round(cpu_time / mps_time, 2)

        if model_speedups:
            speedups["training"][model_name] = model_speedups

    return speedups


def get_device_info(raw_data: dict[str, dict | None]) -> dict:
    """Extract device info from any available benchmark data.

    Args:
        raw_data: Dictionary of all loaded benchmark data

    Returns:
        Device info dictionary
    """
    # Try to get device info from any available source
    for data in raw_data.values():
        if data and "device_info" in data:
            return data["device_info"]
    return {}


def get_config_info(raw_data: dict[str, dict | None]) -> dict:
    """Extract configuration info from benchmark data.

    Args:
        raw_data: Dictionary of all loaded benchmark data

    Returns:
        Configuration info dictionary
    """
    config = {}

    # Inference config
    if raw_data.get("inference"):
        inf_config = raw_data["inference"].get("configuration", {})
        config["inference"] = {
            "warmup_runs": inf_config.get("warmup_runs", 0),
            "timed_runs": inf_config.get("timed_runs", 0),
            "image_size": inf_config.get("image_size", 0),
        }

    # Training config (should be same across all training benchmarks)
    for key in ["training_mlx", "training_mps", "training_cpu"]:
        if raw_data.get(key):
            train_config = raw_data[key].get("config", {})
            config["training"] = {
                "epochs": train_config.get("epochs", 0),
                "batch_size": train_config.get("batch_size", 0),
                "learning_rate": train_config.get("learning_rate", 0),
            }
            break

    return config


# =============================================================================
# Summary Generation
# =============================================================================


def print_inference_summary(inference_data: dict, speedups: dict) -> None:
    """Print inference benchmark summary table.

    Args:
        inference_data: Normalized inference data keyed by model name then backend.
        speedups: Dict of speedup ratios keyed by category then model name.
    """
    logger.info("\n" + "=" * 80)
    logger.info("  INFERENCE BENCHMARK SUMMARY")
    logger.info("=" * 80)

    if not inference_data:
        logger.info("  No inference data available.")
        return

    logger.info(
        f"\n{'Model':<12} {'MLX (ms)':<12} {'MPS (ms)':<12} {'CPU (ms)':<12} {'MLX vs CPU':<12} {'MLX vs MPS':<12}"
    )
    logger.info("-" * 80)

    for size in MODEL_SIZES:
        model_name = f"yolo26{size}"
        if model_name not in inference_data:
            continue

        backends = inference_data[model_name]
        mlx_ms = backends.get("mlx", {}).get("mean_ms", 0)
        mps_ms = backends.get("pytorch_mps", {}).get("mean_ms", 0)
        cpu_ms = backends.get("pytorch_cpu", {}).get("mean_ms", 0)

        model_speedups = speedups.get("inference", {}).get(model_name, {})
        mlx_vs_cpu = model_speedups.get("mlx_vs_cpu", 0)
        mlx_vs_mps = model_speedups.get("mlx_vs_mps", 0)

        # Format values with N/A fallback
        mlx_str = f"{mlx_ms:.2f}" if mlx_ms else "N/A"
        mps_str = f"{mps_ms:.2f}" if mps_ms else "N/A"
        cpu_str = f"{cpu_ms:.2f}" if cpu_ms else "N/A"
        mlx_cpu_str = f"{mlx_vs_cpu:.2f}x" if mlx_vs_cpu else "N/A"
        mlx_mps_str = f"{mlx_vs_mps:.2f}x" if mlx_vs_mps else "N/A"

        logger.info(
            f"{model_name:<12} "
            f"{mlx_str:<12} "
            f"{mps_str:<12} "
            f"{cpu_str:<12} "
            f"{mlx_cpu_str:<12} "
            f"{mlx_mps_str:<12}"
        )

    logger.info("-" * 80)


def print_training_summary(training_data: dict, speedups: dict) -> None:
    """Print training benchmark summary table.

    Args:
        training_data: Normalized training data keyed by model name then backend.
        speedups: Dict of speedup ratios keyed by category then model name.
    """
    logger.info("\n" + "=" * 80)
    logger.info("  TRAINING BENCHMARK SUMMARY")
    logger.info("=" * 80)

    if not training_data:
        logger.info("  No training data available.")
        return

    logger.info(
        f"\n{'Model':<12} {'MLX (s)':<12} {'MPS (s)':<12} {'CPU (s)':<12} {'MLX vs CPU':<12} {'MLX vs MPS':<12}"
    )
    logger.info("-" * 80)

    for size in MODEL_SIZES:
        model_name = f"yolo26{size}"
        if model_name not in training_data:
            continue

        backends = training_data[model_name]
        mlx_s = backends.get("mlx", {}).get("training_time_seconds", 0)
        mps_s = backends.get("pytorch_mps", {}).get("training_time_seconds", 0)
        cpu_s = backends.get("pytorch_cpu", {}).get("training_time_seconds", 0)

        model_speedups = speedups.get("training", {}).get(model_name, {})
        mlx_vs_cpu = model_speedups.get("mlx_vs_cpu", 0)
        mlx_vs_mps = model_speedups.get("mlx_vs_mps", 0)

        # Format values with N/A fallback
        mlx_str = f"{mlx_s:.1f}" if mlx_s else "N/A"
        mps_str = f"{mps_s:.1f}" if mps_s else "N/A"
        cpu_str = f"{cpu_s:.1f}" if cpu_s else "N/A"
        mlx_cpu_str = f"{mlx_vs_cpu:.2f}x" if mlx_vs_cpu else "N/A"
        mlx_mps_str = f"{mlx_vs_mps:.2f}x" if mlx_vs_mps else "N/A"

        logger.info(
            f"{model_name:<12} "
            f"{mlx_str:<12} "
            f"{mps_str:<12} "
            f"{cpu_str:<12} "
            f"{mlx_cpu_str:<12} "
            f"{mlx_mps_str:<12}"
        )

    logger.info("-" * 80)


def print_accuracy_summary(training_data: dict) -> None:
    """Print model accuracy (mAP) summary table.

    Args:
        training_data: Normalized training data keyed by model name then backend, containing mAP scores.
    """
    logger.info("\n" + "=" * 80)
    logger.info("  ACCURACY SUMMARY (mAP after training)")
    logger.info("=" * 80)

    if not training_data:
        logger.info("  No training data available.")
        return

    logger.info(f"\n{'Model':<12} {'MLX mAP50':<14} {'MPS mAP50':<14} {'CPU mAP50':<14}")
    logger.info("-" * 60)

    for size in MODEL_SIZES:
        model_name = f"yolo26{size}"
        if model_name not in training_data:
            continue

        backends = training_data[model_name]
        mlx_map = backends.get("mlx", {}).get("mAP50", 0)
        mps_map = backends.get("pytorch_mps", {}).get("mAP50", 0)
        cpu_map = backends.get("pytorch_cpu", {}).get("mAP50", 0)

        # Format values with N/A fallback
        mlx_str = f"{mlx_map:.4f}" if mlx_map else "N/A"
        mps_str = f"{mps_map:.4f}" if mps_map else "N/A"
        cpu_str = f"{cpu_map:.4f}" if cpu_map else "N/A"

        logger.info(f"{model_name:<12} " f"{mlx_str:<14} " f"{mps_str:<14} " f"{cpu_str:<14}")

    logger.info("-" * 60)


# =============================================================================
# Main
# =============================================================================


def main():
    """Collect all benchmark results, compute speedups, and save combined JSON."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="YOLO26 Benchmark Results Collector")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: ../results/yolo26_benchmark_combined.json)",
    )
    args = parser.parse_args()
    ensure_runtime_dirs(PROJECT_DIR)

    logger.info("=" * 80)
    logger.info("  YOLO26 Benchmark Results Collector")
    logger.info("=" * 80)

    # Load all result files
    logger.info("\n📂 Loading benchmark results...")
    raw_data = load_all_results()

    # Check if any data was loaded
    if not any(raw_data.values()):
        logger.error("\n❌ No benchmark results found.")
        logger.error("   Please run the individual benchmark scripts first:")
        logger.error("   - benchmark_yolo26_inference.py")
        logger.error("   - benchmark_yolo26_training_mlx.py")
        logger.error("   - benchmark_yolo26_training_mps.py")
        logger.error("   - benchmark_yolo26_training_cpu.py")
        return

    # Extract and normalize data
    logger.info("\n📊 Processing data...")

    inference_data = extract_inference_data(raw_data.get("inference"))
    training_data = extract_training_data(
        raw_data.get("training_mlx"),
        raw_data.get("training_mps"),
        raw_data.get("training_cpu"),
    )

    # Calculate speedups
    speedups = calculate_speedups(inference_data, training_data)

    # Get device and config info
    device_info = get_device_info(raw_data)
    config_info = get_config_info(raw_data)

    # Print summaries
    print_inference_summary(inference_data, speedups)
    print_training_summary(training_data, speedups)
    print_accuracy_summary(training_data)

    # Prepare combined output
    combined = {
        "benchmark": "YOLO26 Combined Benchmark Results",
        "timestamp": datetime.now().isoformat(),
        "device_info": device_info,
        "configuration": config_info,
        "inference": inference_data,
        "training": training_data,
        "speedups": speedups,
        "source_files": {
            name: str(path) for name, path in INPUT_FILES.items() if raw_data.get(name) is not None
        },
    }

    # Save combined results
    output_path = (
        Path(args.output) if args.output else RESULTS_DIR / "yolo26_benchmark_combined.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"\n✅ Combined results saved to: {output_path}")
    logger.info("\n✨ Results collection complete!")


if __name__ == "__main__":
    main()
