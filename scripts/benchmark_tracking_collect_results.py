#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 Tracking Benchmark Results Collector
=============================================
Collects tracking results from MOT17 evaluation runs and combines them into a
unified format for chart generation.

This script reads JSON files produced by:
- evaluate_mot17.py           -> results/tracking/{model}_{tracker}_mot17_results.json
- evaluate_mot17_pytorch.py   -> results/tracking/{model}_{tracker}_mot17_pytorch_{device}_results.json

And produces a combined report at:
    ../results/yolo26_tracking_benchmark_combined.json

Usage:
    python benchmark_tracking_collect_results.py
    python benchmark_tracking_collect_results.py --tracker botsort
    python benchmark_tracking_collect_results.py --output custom_output.json
"""

import argparse
import json
import logging
import re
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
TRACKING_DIR = RESULTS_DIR / "tracking"

MODEL_SIZES = ["n", "s", "m", "l", "x"]
MODEL_NAMES = [f"yolo26{s}" for s in MODEL_SIZES]

# Filename patterns produced by evaluate_mot17.py / evaluate_mot17_pytorch.py
MLX_PATTERN = re.compile(r"^(yolo26[nslmx])_(\w+)_mot17_results\.json$")
PYTORCH_PATTERN = re.compile(r"^(yolo26[nslmx])_(\w+)_mot17_pytorch_(mps|cpu)_results\.json$")


# =============================================================================
# Data Loading
# =============================================================================


def discover_tracking_files(tracking_dir: Path, tracker: str) -> dict[str, Path]:
    """Discover tracking result JSON files matching the given tracker.

    Args:
        tracking_dir: Directory containing per-model result JSON files.
        tracker: Tracker name to filter by (e.g. "bytetrack", "botsort").

    Returns:
        Mapping from descriptive key to file path for each discovered file.
    """
    found: dict[str, Path] = {}
    if not tracking_dir.is_dir():
        return found

    for path in sorted(tracking_dir.glob("*.json")):
        m_mlx = MLX_PATTERN.match(path.name)
        if m_mlx and m_mlx.group(2) == tracker:
            model = m_mlx.group(1)
            found[f"{model}_mlx"] = path
            continue

        m_pt = PYTORCH_PATTERN.match(path.name)
        if m_pt and m_pt.group(2) == tracker:
            model, device = m_pt.group(1), m_pt.group(3)
            found[f"{model}_pytorch_{device}"] = path

    return found


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


def load_tracking_results(tracking_dir: Path, tracker: str) -> dict[str, dict | None]:
    """Load all tracking result files for the given tracker.

    Args:
        tracking_dir: Directory containing per-model result JSON files.
        tracker: Tracker name to filter by.

    Returns:
        Mapping from descriptive key to loaded data (or None if load failed).
    """
    files = discover_tracking_files(tracking_dir, tracker)
    results: dict[str, dict | None] = {}

    if not files:
        logger.warning(f"  ⚠️  No tracking result files found for tracker={tracker}")
        return results

    for key, path in files.items():
        data = load_json_file(path)
        if data:
            logger.info(f"  ✅ Loaded: {path.name}")
        else:
            logger.warning(f"  ⚠️  Failed: {path.name}")
        results[key] = data

    return results


# =============================================================================
# Data Processing
# =============================================================================


def extract_tracking_data(raw: dict[str, dict | None]) -> dict:
    """Normalize per-file tracking results into a model-keyed structure.

    Args:
        raw: Mapping from descriptive key to raw JSON data.

    Returns:
        Dict keyed by model name, each containing backend sub-dicts with
        metrics and speed data.
    """
    result: dict[str, dict] = {}

    for _key, data in raw.items():
        if data is None:
            continue

        model = data.get("model", "")
        framework = data.get("framework", "")
        if not model or not framework:
            continue

        # Normalize framework name to match detection convention
        backend = framework  # "mlx", "pytorch_mps", "pytorch_cpu"

        if model not in result:
            result[model] = {}

        metrics = data.get("metrics", {})
        speed = data.get("speed", {})

        result[model][backend] = {
            "MOTA": metrics.get("MOTA", 0),
            "IDF1": metrics.get("IDF1", 0),
            "MT": metrics.get("MT", 0),
            "ML": metrics.get("ML", 0),
            "FP": metrics.get("FP", 0),
            "FN": metrics.get("FN", 0),
            "IDSW": metrics.get("IDSW", 0),
            "Frag": metrics.get("Frag", 0),
            "fps": speed.get("fps", 0),
            "compute_fps": speed.get("compute_fps", speed.get("fps", 0)),
            "total_ms": speed.get("total_ms", 0),
            "detection_ms": speed.get("detection_ms", 0),
            "tracking_ms": speed.get("tracking_ms", 0),
            "io_ms": speed.get("io_ms", 0),
        }

    return result


def calculate_speedups(tracking_data: dict) -> dict:
    """Calculate FPS speedup ratios between backends.

    Args:
        tracking_data: Normalized tracking data keyed by model then backend.

    Returns:
        Speedup ratios keyed by model name.
    """
    speedups: dict[str, dict] = {}

    for model_name, backends in tracking_data.items():
        mlx_fps = backends.get("mlx", {}).get("fps", 0)
        mps_fps = backends.get("pytorch_mps", {}).get("fps", 0)
        cpu_fps = backends.get("pytorch_cpu", {}).get("fps", 0)

        model_speedups: dict[str, float] = {}

        if mlx_fps > 0 and cpu_fps > 0:
            model_speedups["mlx_vs_cpu"] = round(mlx_fps / cpu_fps, 2)
        if mlx_fps > 0 and mps_fps > 0:
            model_speedups["mlx_vs_mps"] = round(mlx_fps / mps_fps, 2)
        if mps_fps > 0 and cpu_fps > 0:
            model_speedups["mps_vs_cpu"] = round(mps_fps / cpu_fps, 2)

        if model_speedups:
            speedups[model_name] = model_speedups

    return speedups


def get_common_metadata(raw: dict[str, dict | None]) -> dict:
    """Extract shared metadata (dataset, tracker, imgsz) from any result file.

    Args:
        raw: Mapping from descriptive key to raw JSON data.

    Returns:
        Dict of common metadata fields.
    """
    for data in raw.values():
        if data is not None:
            return {
                "tracker": data.get("tracker", ""),
                "dataset": data.get("dataset", ""),
                "imgsz": data.get("imgsz", 0),
                "conf_thresh": data.get("conf_thresh", 0),
            }
    return {}


# =============================================================================
# Summary
# =============================================================================


def print_tracking_summary(tracking_data: dict, speedups: dict) -> None:
    """Print tracking benchmark summary table.

    Args:
        tracking_data: Normalized tracking data keyed by model then backend.
        speedups: Speedup ratios keyed by model name.
    """
    logger.info("\n" + "=" * 95)
    logger.info("  TRACKING BENCHMARK SUMMARY")
    logger.info("=" * 95)

    if not tracking_data:
        logger.info("  No tracking data available.")
        return

    header = (
        f"{'Model':<12} "
        f"{'MLX MOTA':>9} {'MPS MOTA':>9} {'CPU MOTA':>9} "
        f"{'MLX FPS':>8} {'MPS FPS':>8} {'CPU FPS':>8} "
        f"{'vs CPU':>7}"
    )
    logger.info(f"\n{header}")
    logger.info("-" * 95)

    for size in MODEL_SIZES:
        model_name = f"yolo26{size}"
        if model_name not in tracking_data:
            continue

        b = tracking_data[model_name]
        mlx = b.get("mlx", {})
        mps = b.get("pytorch_mps", {})
        cpu = b.get("pytorch_cpu", {})

        sp = speedups.get(model_name, {})

        def fmt(v, prec=1):
            return f"{v:.{prec}f}" if v else "N/A"

        logger.info(
            f"{model_name:<12} "
            f"{fmt(mlx.get('MOTA')):>9} {fmt(mps.get('MOTA')):>9} {fmt(cpu.get('MOTA')):>9} "
            f"{fmt(mlx.get('fps')):>8} {fmt(mps.get('fps')):>8} {fmt(cpu.get('fps')):>8} "
            f"{fmt(sp.get('mlx_vs_cpu'), 2) + 'x' if sp.get('mlx_vs_cpu') else 'N/A':>7}"
        )

    logger.info("-" * 95)


# =============================================================================
# Main
# =============================================================================


def main():
    """Collect tracking results, compute speedups, and save combined JSON."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="YOLO26 Tracking Benchmark Results Collector")
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        help="Tracker name to collect results for (default: bytetrack)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"Input directory with tracking JSON files (default: {TRACKING_DIR})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: ../results/yolo26_tracking_benchmark_combined.json)",
    )
    args = parser.parse_args()
    ensure_runtime_dirs(PROJECT_DIR)

    logger.info("=" * 80)
    logger.info("  YOLO26 Tracking Benchmark Results Collector")
    logger.info("=" * 80)

    tracking_dir = args.input or TRACKING_DIR

    # Load tracking result files
    logger.info(f"\n📂 Loading tracking results from: {tracking_dir}")
    logger.info(f"   Tracker: {args.tracker}")
    raw_data = load_tracking_results(tracking_dir, args.tracker)

    if not any(raw_data.values()):
        logger.error("\n❌ No tracking results found.")
        logger.error("   Run the evaluation scripts first:")
        logger.error("   - python scripts/evaluate_mot17.py --model all")
        logger.error("   - python scripts/evaluate_mot17_pytorch.py --model all --device mps")
        logger.error("   - python scripts/evaluate_mot17_pytorch.py --model all --device cpu")
        return

    # Extract and normalize
    logger.info("\n📊 Processing data...")
    tracking_data = extract_tracking_data(raw_data)
    speedups = calculate_speedups(tracking_data)
    metadata = get_common_metadata(raw_data)

    # Print summary
    print_tracking_summary(tracking_data, speedups)

    # Prepare combined output
    combined = {
        "benchmark": "YOLO26 Tracking Benchmark Results",
        "timestamp": datetime.now().isoformat(),
        "tracker": metadata.get("tracker", args.tracker),
        "dataset": metadata.get("dataset", "MOT17-train"),
        "imgsz": metadata.get("imgsz", 1440),
        "conf_thresh": metadata.get("conf_thresh", 0.25),
        "tracking": tracking_data,
        "speedups": speedups,
        "source_files": {
            key: str(path)
            for key, path in discover_tracking_files(tracking_dir, args.tracker).items()
        },
    }

    output_path = (
        Path(args.output)
        if args.output
        else RESULTS_DIR / "yolo26_tracking_benchmark_combined.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"\n✅ Combined results saved to: {output_path}")
    logger.info("\n✨ Tracking results collection complete!")


if __name__ == "__main__":
    main()
