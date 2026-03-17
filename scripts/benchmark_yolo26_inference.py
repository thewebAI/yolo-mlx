#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 Inference Benchmark - Three-Way Comparison
==================================================
Compares inference performance across MLX GPU, PyTorch MPS, and PyTorch CPU backends.

Usage:
    python benchmark_yolo26_inference.py
    python benchmark_yolo26_inference.py --models n s      # Specific models only
    python benchmark_yolo26_inference.py --runs 20         # More timed runs
    python benchmark_yolo26_inference.py --output custom.json

Output:
    ../results/yolo26_inference_three_way.json (default, overridable via --output)
"""

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from _runtime_dirs import ensure_runtime_dirs
from PIL import Image

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

WARMUP_RUNS = 3
TIMED_RUNS = 10
IMAGE_SIZE = 640
MODEL_SIZES = ["n", "s", "m", "l", "x"]

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR / ".."
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"
IMAGES_DIR = PROJECT_DIR / "images"
TEST_IMAGE = IMAGES_DIR / "bus.jpg"


# =============================================================================
# Utility Functions
# =============================================================================


def get_device_info() -> dict[str, Any]:
    """Get system and device information.

    Returns:
        Dict with platform, processor, python_version, and macOS-specific cpu/model fields.
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

    # Try to get Mac model
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            info["mac_model"] = result.stdout.strip()
    except Exception:
        pass

    return info


def calculate_stats(times_ms: list[float]) -> dict[str, float]:
    """Calculate statistics from timing results.

    Args:
        times_ms: List of latency measurements in milliseconds.

    Returns:
        Dict with mean, std, min, max, median (all in ms) and fps.
    """
    arr = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "median_ms": float(np.median(arr)),
        "fps": float(1000.0 / np.mean(arr)),
    }


def load_test_image() -> Image.Image:
    """Load and verify test image exists.

    Returns:
        PIL Image loaded from the configured TEST_IMAGE path.
    """
    if not TEST_IMAGE.exists():
        raise FileNotFoundError(
            f"Test image not found: {TEST_IMAGE}\n"
            "Run: curl -L -o ../images/bus.jpg https://ultralytics.com/images/bus.jpg"
        )
    return Image.open(TEST_IMAGE)


# =============================================================================
# MLX Inference Benchmark
# =============================================================================


def test_mlx_inference(
    model_size: str,
    warmup: int,
    runs: int,
    allow_random_mlx: bool = False,
) -> dict[str, Any] | None:
    """Test MLX GPU inference timing.

    Args:
        model_size: YOLO26 scale variant letter (e.g. "n", "s", "m", "l", "x").
        warmup: Number of warmup iterations before timing begins.
        runs: Number of timed inference iterations to measure.

    Returns:
        Dict of timing stats and forward-pass metrics, or None if MLX is unavailable.
    """
    try:
        import mlx.core as mx
        import yaml

        # Set MLX to use GPU
        mx.set_default_device(mx.gpu)

        from yolo26mlx import YOLO
        from yolo26mlx.nn.tasks import build_model
    except ImportError as e:
        logger.warning(f"  ⚠️  MLX not available: {e}")
        return None

    model_name = f"yolo26{model_size}"

    # Try different model sources in order of preference:
    # 1. NPZ weights (converted from PyTorch)
    # 2. Safetensors weights
    # 3. YAML config with scale (random weights - optional timing-only fallback)
    npz_path = MODELS_DIR / f"{model_name}.npz"
    safetensors_path = MODELS_DIR / f"{model_name}.safetensors"
    yaml_path = None

    # Find YAML config in the package (yolo26.yaml, not yolo26n.yaml)
    try:
        import yolo26mlx

        pkg_dir = Path(yolo26mlx.__file__).parent
        yaml_candidates = [
            pkg_dir / "cfg" / "models" / "26" / "yolo26.yaml",
            pkg_dir / "cfg" / "models" / "yolo26.yaml",
        ]
        for candidate in yaml_candidates:
            if candidate.exists():
                yaml_path = candidate
                break
    except Exception:
        pass

    # Load model
    model_source = None
    try:
        if npz_path.exists():
            logger.info(f"  Loading {model_name} from npz weights...")
            model = YOLO(str(npz_path))
            model_source = "npz"
        elif safetensors_path.exists():
            logger.info(f"  Loading {model_name} from safetensors...")
            model = YOLO(str(safetensors_path))
            model_source = "safetensors"
        elif yaml_path and yaml_path.exists():
            if not allow_random_mlx:
                logger.warning(
                    f"  ⚠️  Skipping {model_name} MLX benchmark: converted weights not found"
                )
                logger.warning(f"      Expected: {npz_path} or {safetensors_path}")
                logger.warning(
                    "      Refusing YAML random-weights fallback for fair MLX vs MPS comparison."
                )
                logger.warning(
                    "      Use --allow-random-mlx to benchmark random-weight YAML models."
                )
                return None
            # Load YAML config and set scale for proper model sizing
            logger.info(f"  Loading {model_name} from YAML (random weights, scale={model_size})...")
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            cfg["scale"] = model_size  # Set the model scale (n, s, m, l, x)

            # Build model with correct scale
            model = YOLO(str(yaml_path))
            model.model = build_model(cfg, verbose=True)
            model._setup_metadata()
            model_source = "yaml"
        else:
            logger.warning(f"  ⚠️  No MLX model found for {model_name}")
            logger.warning(f"      Expected: {npz_path} or {safetensors_path}")
            logger.warning(f"      Or YAML: {yaml_path}")
            return None
    except Exception as e:
        logger.warning(f"  ⚠️  Failed to load MLX model: {e}")
        return None

    # Load test image
    image = load_test_image()

    # Warmup runs (end-to-end predict)
    logger.info(f"  Warmup ({warmup} runs)...")
    for _ in range(warmup):
        # `model.predict()` already synchronizes MLX work in predictor via `mx.eval(preds)`.
        _ = model.predict(image, imgsz=IMAGE_SIZE)
    logger.info("done")

    # Timed runs (end-to-end predict)
    logger.info(f"  Timing ({runs} runs)...")
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        # Keep end-to-end timing at predictor level; pass arrays to `mx.eval(...)`
        # when explicit synchronization is needed.
        _ = model.predict(image, imgsz=IMAGE_SIZE)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    logger.info("done")

    stats = calculate_stats(times)
    stats["model_source"] = model_source if model_source else "unknown"

    # Forward-pass-only timing (isolates mx.compile benefit)
    logger.info(f"  Forward-pass-only ({runs} runs)...")
    img_np = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0
    x = mx.expand_dims(mx.array(img_np), axis=0)
    mx.eval(x)

    # Ensure model is compiled
    det_model = model.model
    if hasattr(det_model, "compile_for_inference"):
        det_model.compile_for_inference()

    for _ in range(warmup):
        out = det_model(x)
        mx.eval(out)

    fwd_times = []
    for _ in range(runs):
        start = time.perf_counter()
        out = det_model(x)
        mx.eval(out)
        elapsed = (time.perf_counter() - start) * 1000
        fwd_times.append(elapsed)
    logger.info("done")

    fwd_stats = calculate_stats(fwd_times)
    stats["forward_only_mean_ms"] = fwd_stats["mean_ms"]
    stats["forward_only_fps"] = fwd_stats["fps"]

    # Get memory info if available (MLX 0.30.3+ API)
    try:
        if hasattr(mx, "get_peak_memory"):
            stats["peak_memory_mb"] = mx.get_peak_memory() / 1024 / 1024
        elif hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            stats["peak_memory_mb"] = mx.metal.get_peak_memory() / 1024 / 1024
    except Exception:
        pass

    return stats


# =============================================================================
# PyTorch MPS Inference Benchmark
# =============================================================================


def test_pytorch_mps_inference(model_size: str, warmup: int, runs: int) -> dict[str, Any] | None:
    """Test PyTorch MPS inference timing.

    Args:
        model_size: YOLO26 scale variant letter (e.g. "n", "s", "m", "l", "x").
        warmup: Number of warmup iterations before timing begins.
        runs: Number of timed inference iterations to measure.

    Returns:
        Dict of timing stats and forward-pass metrics, or None if MPS is unavailable.
    """
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        logger.warning(f"  ⚠️  PyTorch/Ultralytics not available: {e}")
        return None

    # Check MPS availability
    if not torch.backends.mps.is_available():
        logger.warning("  ⚠️  MPS not available on this system")
        return None

    model_name = f"yolo26{model_size}"
    pt_path = MODELS_DIR / f"{model_name}.pt"

    # Load model (auto-downloads if not present)
    try:
        if pt_path.exists():
            logger.info(f"  Loading {model_name} from local .pt file...")
            model = YOLO(str(pt_path))
        else:
            logger.info(f"  Loading {model_name} (will auto-download)...")
            model = YOLO(f"{model_name}.pt")
    except Exception as e:
        logger.warning(f"  ⚠️  Failed to load PyTorch model: {e}")
        return None

    # Load test image path
    image_path = str(TEST_IMAGE)

    # Warmup runs
    logger.info(f"  Warmup ({warmup} runs)...")
    for _ in range(warmup):
        _ = model.predict(image_path, imgsz=IMAGE_SIZE, device="mps", verbose=False)
    logger.info("done")

    # Timed runs
    logger.info(f"  Timing ({runs} runs)...")
    times = []
    for _ in range(runs):
        # Sync MPS before timing
        torch.mps.synchronize()
        start = time.perf_counter()
        _ = model.predict(image_path, imgsz=IMAGE_SIZE, device="mps", verbose=False)
        torch.mps.synchronize()  # Wait for GPU to finish
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    logger.info("done")

    stats = calculate_stats(times)

    # Forward-pass-only timing (isolates GPU compute)
    logger.info(f"  Forward-pass-only ({runs} runs)...")
    torch_model = model.model
    torch_model.to("mps")
    torch_model.eval()
    x_pt = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device="mps")

    with torch.no_grad():
        for _ in range(warmup):
            _ = torch_model(x_pt)
            torch.mps.synchronize()

        fwd_times = []
        for _ in range(runs):
            torch.mps.synchronize()
            start = time.perf_counter()
            _ = torch_model(x_pt)
            torch.mps.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            fwd_times.append(elapsed)
    logger.info("done")

    fwd_stats = calculate_stats(fwd_times)
    stats["forward_only_mean_ms"] = fwd_stats["mean_ms"]
    stats["forward_only_fps"] = fwd_stats["fps"]

    return stats


# =============================================================================
# PyTorch CPU Inference Benchmark
# =============================================================================


def test_pytorch_cpu_inference(model_size: str, warmup: int, runs: int) -> dict[str, Any] | None:
    """Test PyTorch CPU inference timing.

    Args:
        model_size: YOLO26 scale variant letter (e.g. "n", "s", "m", "l", "x").
        warmup: Number of warmup iterations before timing begins.
        runs: Number of timed inference iterations to measure.

    Returns:
        Dict of timing stats, or None if PyTorch/Ultralytics is unavailable.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.warning(f"  ⚠️  Ultralytics not available: {e}")
        return None

    model_name = f"yolo26{model_size}"
    pt_path = MODELS_DIR / f"{model_name}.pt"

    # Load model
    try:
        if pt_path.exists():
            logger.info(f"  Loading {model_name} from local .pt file...")
            model = YOLO(str(pt_path))
        else:
            logger.info(f"  Loading {model_name} (will auto-download)...")
            model = YOLO(f"{model_name}.pt")
    except Exception as e:
        logger.warning(f"  ⚠️  Failed to load PyTorch model: {e}")
        return None

    # Load test image path
    image_path = str(TEST_IMAGE)

    # Warmup runs
    logger.info(f"  Warmup ({warmup} runs)...")
    for _ in range(warmup):
        _ = model.predict(image_path, imgsz=IMAGE_SIZE, device="cpu", verbose=False)
    logger.info("done")

    # Timed runs
    logger.info(f"  Timing ({runs} runs)...")
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model.predict(image_path, imgsz=IMAGE_SIZE, device="cpu", verbose=False)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    logger.info("done")

    return calculate_stats(times)


# =============================================================================
# Results Processing
# =============================================================================


def calculate_speedups(results: dict[str, dict]) -> dict[str, dict]:
    """Calculate speedup ratios between backends.

    Args:
        results: Nested dict keyed by backend name then model size, each containing timing stats.

    Returns:
        Dict with mlx_vs_cpu, mps_vs_cpu, and mlx_vs_mps speedup ratios per model size.
    """
    speedups = {
        "mlx_vs_cpu": {},
        "mps_vs_cpu": {},
        "mlx_vs_mps": {},
    }

    for size in MODEL_SIZES:
        mlx_time = results.get("mlx", {}).get(size, {}).get("mean_ms")
        mps_time = results.get("pytorch_mps", {}).get(size, {}).get("mean_ms")
        cpu_time = results.get("pytorch_cpu", {}).get(size, {}).get("mean_ms")

        if mlx_time and cpu_time:
            speedups["mlx_vs_cpu"][size] = round(cpu_time / mlx_time, 2)
        if mps_time and cpu_time:
            speedups["mps_vs_cpu"][size] = round(cpu_time / mps_time, 2)
        if mlx_time and mps_time:
            speedups["mlx_vs_mps"][size] = round(mps_time / mlx_time, 2)

    return speedups


def print_summary(results: dict[str, dict], speedups: dict[str, dict]) -> None:
    """Print a summary table of results.

    Args:
        results: Nested dict keyed by backend name then model size with timing stats.
        speedups: Dict of speedup ratios between backend pairs per model size.
    """
    logger.info("\n" + "=" * 80)
    logger.info("  YOLO26 Inference Benchmark Summary")
    logger.info("=" * 80)

    # Header
    logger.info(
        f"\n{'Model':<10} {'MLX (ms)':<12} {'MPS (ms)':<12} {'CPU (ms)':<12} {'MLX vs MPS':<12} {'MLX vs CPU':<12}"
    )
    logger.info("-" * 70)

    for size in MODEL_SIZES:
        mlx = results.get("mlx", {}).get(size, {})
        mps = results.get("pytorch_mps", {}).get(size, {})
        cpu = results.get("pytorch_cpu", {}).get(size, {})

        mlx_str = f"{mlx.get('mean_ms', 0):.1f}" if mlx else "N/A"
        mps_str = f"{mps.get('mean_ms', 0):.1f}" if mps else "N/A"
        cpu_str = f"{cpu.get('mean_ms', 0):.1f}" if cpu else "N/A"

        speedup_mps = speedups.get("mlx_vs_mps", {}).get(size, 0)
        speedup_mps_str = f"{speedup_mps:.2f}x" if speedup_mps else "N/A"

        speedup_cpu = speedups.get("mlx_vs_cpu", {}).get(size, 0)
        speedup_cpu_str = f"{speedup_cpu:.1f}x" if speedup_cpu else "N/A"

        logger.info(
            f"yolo26{size:<4} {mlx_str:<12} {mps_str:<12} {cpu_str:<12} {speedup_mps_str:<12} {speedup_cpu_str:<12}"
        )

    logger.info("-" * 70)

    # Print average speedups
    mlx_vs_mps_values = [v for v in speedups.get("mlx_vs_mps", {}).values() if v]
    mlx_vs_cpu_values = [v for v in speedups.get("mlx_vs_cpu", {}).values() if v]

    if mlx_vs_mps_values or mlx_vs_cpu_values:
        logger.info("\n  Average Speedups (end-to-end predict):")
        if mlx_vs_mps_values:
            avg_mps = sum(mlx_vs_mps_values) / len(mlx_vs_mps_values)
            logger.info(
                f"    MLX vs MPS: {avg_mps:.2f}x {'(MLX faster)' if avg_mps > 1 else '(MPS faster)'}"
            )
        if mlx_vs_cpu_values:
            avg_cpu = sum(mlx_vs_cpu_values) / len(mlx_vs_cpu_values)
            logger.info(f"    MLX vs CPU: {avg_cpu:.1f}x")

    # Forward-pass-only summary
    logger.info(f"\n{'Model':<10} {'MLX fwd':<12} {'MPS fwd':<12} {'MLX vs MPS':<12}")
    logger.info("-" * 46)

    for size in MODEL_SIZES:
        mlx = results.get("mlx", {}).get(size, {})
        mps = results.get("pytorch_mps", {}).get(size, {})

        mlx_fwd = mlx.get("forward_only_mean_ms", 0)
        mps_fwd = mps.get("forward_only_mean_ms", 0)

        mlx_str = f"{mlx_fwd:.1f}" if mlx_fwd else "N/A"
        mps_str = f"{mps_fwd:.1f}" if mps_fwd else "N/A"

        if mlx_fwd > 0 and mps_fwd > 0:
            ratio = mps_fwd / mlx_fwd
            ratio_str = f"{ratio:.2f}x {'(MLX!)' if ratio > 1 else '(MPS)'}"
        else:
            ratio_str = "N/A"

        logger.info(f"yolo26{size:<4} {mlx_str:<12} {mps_str:<12} {ratio_str:<12}")

    logger.info("-" * 46)
    logger.info("")


def save_results(results: dict, output_path: Path) -> None:
    """Save results to JSON file.

    Args:
        results: Benchmark results dictionary to serialize.
        output_path: Destination file path for the JSON output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"✅ Results saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run the three-way inference benchmark, print results, and save to JSON."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="YOLO26 Inference Benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_SIZES,
        choices=MODEL_SIZES,
        help="Model sizes to benchmark (default: all)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_RUNS,
        help=f"Number of warmup runs (default: {WARMUP_RUNS})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=TIMED_RUNS,
        help=f"Number of timed runs (default: {TIMED_RUNS})",
    )
    parser.add_argument(
        "--skip-mlx",
        action="store_true",
        help="Skip MLX benchmark",
    )
    parser.add_argument(
        "--skip-mps",
        action="store_true",
        help="Skip MPS benchmark",
    )
    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        help="Skip CPU benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "yolo26_inference_three_way.json",
        help="Output JSON path (default: results/yolo26_inference_three_way.json)",
    )
    parser.add_argument(
        "--allow-random-mlx",
        action="store_true",
        help="Allow MLX YAML random-weights fallback when converted weights are missing",
    )
    args = parser.parse_args()
    ensure_runtime_dirs(PROJECT_DIR)

    logger.info("=" * 70)
    logger.info("  YOLO26 Inference Benchmark - Three-Way Comparison")
    logger.info("=" * 70)
    logger.info(f"  Models: {', '.join(args.models)}")
    logger.info(f"  Warmup runs: {args.warmup}")
    logger.info(f"  Timed runs: {args.runs}")
    logger.info(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    logger.info("=" * 70)
    logger.info("")

    # Verify test image exists
    try:
        load_test_image()
        logger.info(f"✅ Test image: {TEST_IMAGE}")
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)

    # Get device info
    device_info = get_device_info()
    logger.info(f"✅ Platform: {device_info.get('cpu', device_info.get('processor', 'Unknown'))}")
    logger.info("")

    # Results storage
    results = {
        "mlx": {},
        "pytorch_mps": {},
        "pytorch_cpu": {},
    }

    # Run benchmarks for each model size
    for size in args.models:
        model_name = f"yolo26{size}"
        logger.info(f"\n{'=' * 50}")
        logger.info(f"  Benchmarking: {model_name}")
        logger.info(f"{'=' * 50}")

        # MLX benchmark
        if not args.skip_mlx:
            logger.info("\n[MLX GPU]")
            mlx_result = test_mlx_inference(
                size,
                args.warmup,
                args.runs,
                allow_random_mlx=args.allow_random_mlx,
            )
            if mlx_result:
                results["mlx"][size] = mlx_result
                logger.info(
                    f"  → Mean: {mlx_result['mean_ms']:.2f}ms ({mlx_result['fps']:.1f} FPS)"
                )

        # MPS benchmark
        if not args.skip_mps:
            logger.info("\n[PyTorch MPS]")
            mps_result = test_pytorch_mps_inference(size, args.warmup, args.runs)
            if mps_result:
                results["pytorch_mps"][size] = mps_result
                logger.info(
                    f"  → Mean: {mps_result['mean_ms']:.2f}ms ({mps_result['fps']:.1f} FPS)"
                )

        # CPU benchmark
        if not args.skip_cpu:
            logger.info("\n[PyTorch CPU]")
            cpu_result = test_pytorch_cpu_inference(size, args.warmup, args.runs)
            if cpu_result:
                results["pytorch_cpu"][size] = cpu_result
                logger.info(
                    f"  → Mean: {cpu_result['mean_ms']:.2f}ms ({cpu_result['fps']:.1f} FPS)"
                )

    # Calculate speedups
    speedups = calculate_speedups(results)

    # Print summary
    print_summary(results, speedups)

    # Prepare final output
    output = {
        "benchmark": "YOLO26 Inference - Three-Way Comparison",
        "timestamp": datetime.now().isoformat(),
        "device_info": device_info,
        "configuration": {
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "image_size": IMAGE_SIZE,
            "model_sizes": args.models,
            "allow_random_mlx": args.allow_random_mlx,
        },
        "results": results,
        "speedups": speedups,
    }

    # Save results
    save_results(output, args.output)


if __name__ == "__main__":
    main()
