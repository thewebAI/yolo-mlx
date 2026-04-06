#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 Tracking Benchmark Chart Generator
==========================================
Generates visualization charts from tracking benchmark results.

This script reads the combined tracking results from:
    ../results/yolo26_tracking_benchmark_combined.json

And generates charts in:
    ../results/charts/

Charts generated:
- Tracking accuracy (MOTA) comparison
- Tracking accuracy (IDF1) comparison
- Tracking throughput (FPS) comparison
- Tracking speedup comparison (MLX vs CPU, MLX vs MPS)
- Tracking overhead breakdown (detection vs tracking time)
- Tracking summary dashboard (2x2 grid)

Usage:
    python benchmark_tracking_generate_charts.py
    python benchmark_tracking_generate_charts.py --input custom_results.json
    python benchmark_tracking_generate_charts.py --format pdf
    python benchmark_tracking_generate_charts.py --output custom_charts/
"""

import argparse
import json
import logging
from pathlib import Path

from _runtime_dirs import ensure_runtime_dirs

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR / ".."
RESULTS_DIR = PROJECT_DIR / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
DEFAULT_INPUT = RESULTS_DIR / "yolo26_tracking_benchmark_combined.json"

MODEL_SIZES = ["n", "s", "m", "l", "x"]
MODEL_LABELS = ["YOLO26n", "YOLO26s", "YOLO26m", "YOLO26l", "YOLO26x"]

# Colorblind-friendly colors (IBM Design Library) — matches detection charts
COLORS = {
    "mlx": "#648FFF",
    "pytorch_mps": "#785EF0",
    "pytorch_cpu": "#DC267F",
}

BACKEND_LABELS = {
    "mlx": "MLX (Apple GPU)",
    "pytorch_mps": "PyTorch MPS",
    "pytorch_cpu": "PyTorch CPU",
}


# =============================================================================
# Data Loading
# =============================================================================


def load_results(path: Path) -> dict | None:
    """Load tracking benchmark results from JSON file.

    Args:
        path: Path to combined tracking results JSON file

    Returns:
        Parsed JSON data or None if file doesn't exist
    """
    if not path.exists():
        logger.error(f"❌ Results file not found: {path}")
        logger.error("   Run benchmark_tracking_collect_results.py first.")
        return None

    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"❌ Error loading {path}: {e}")
        return None


def _extract_models(tracking_data: dict) -> list[tuple[str, str, str]]:
    """Build ordered list of (size, label, model_key) tuples for available models.

    Args:
        tracking_data: Tracking data dict keyed by model name.

    Returns:
        List of (size_letter, display_label, model_key) tuples.
    """
    models = []
    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = f"yolo26{size}"
        if model_key in tracking_data:
            models.append((size, label, model_key))
    return models


# =============================================================================
# Helpers
# =============================================================================


def _add_bar_labels(ax, bars, fmt: str = "{:.1f}", fontsize: int = 8) -> None:
    """Annotate each bar with its numeric value."""
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                fmt.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )


# =============================================================================
# Chart Generation Functions
# =============================================================================


def create_tracking_mota_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> bool:
    """Create MOTA comparison bar chart across backends.

    Args:
        data: Combined tracking benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution.

    Returns:
        True if chart was created successfully.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("  ⚠️  matplotlib not available, skipping chart")
        return False

    tracking = data.get("tracking", {})
    models = _extract_models(tracking)
    if not models:
        logger.warning("  ⚠️  No tracking data available")
        return False

    mlx_vals = [tracking[m[2]].get("mlx", {}).get("MOTA", 0) for m in models]
    mps_vals = [tracking[m[2]].get("pytorch_mps", {}).get("MOTA", 0) for m in models]
    cpu_vals = [tracking[m[2]].get("pytorch_cpu", {}).get("MOTA", 0) for m in models]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(models))
    width = 0.25

    b1 = ax.bar(
        x - width,
        mlx_vals,
        width,
        label=BACKEND_LABELS["mlx"],
        color=COLORS["mlx"],
        edgecolor="white",
        linewidth=0.5,
    )
    b2 = ax.bar(
        x,
        mps_vals,
        width,
        label=BACKEND_LABELS["pytorch_mps"],
        color=COLORS["pytorch_mps"],
        edgecolor="white",
        linewidth=0.5,
    )
    b3 = ax.bar(
        x + width,
        cpu_vals,
        width,
        label=BACKEND_LABELS["pytorch_cpu"],
        color=COLORS["pytorch_cpu"],
        edgecolor="white",
        linewidth=0.5,
    )

    _add_bar_labels(ax, b1)
    _add_bar_labels(ax, b2)
    _add_bar_labels(ax, b3)

    tracker = data.get("tracker", "bytetrack").replace("_", " ").title()
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("MOTA (%)", fontsize=12)
    ax.set_title(f"YOLO26 Tracking MOTA Comparison ({tracker})", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in models])
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def create_tracking_idf1_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> bool:
    """Create IDF1 comparison bar chart across backends.

    Args:
        data: Combined tracking benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution.

    Returns:
        True if chart was created successfully.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    tracking = data.get("tracking", {})
    models = _extract_models(tracking)
    if not models:
        return False

    mlx_vals = [tracking[m[2]].get("mlx", {}).get("IDF1", 0) for m in models]
    mps_vals = [tracking[m[2]].get("pytorch_mps", {}).get("IDF1", 0) for m in models]
    cpu_vals = [tracking[m[2]].get("pytorch_cpu", {}).get("IDF1", 0) for m in models]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(models))
    width = 0.25

    b1 = ax.bar(
        x - width,
        mlx_vals,
        width,
        label=BACKEND_LABELS["mlx"],
        color=COLORS["mlx"],
        edgecolor="white",
        linewidth=0.5,
    )
    b2 = ax.bar(
        x,
        mps_vals,
        width,
        label=BACKEND_LABELS["pytorch_mps"],
        color=COLORS["pytorch_mps"],
        edgecolor="white",
        linewidth=0.5,
    )
    b3 = ax.bar(
        x + width,
        cpu_vals,
        width,
        label=BACKEND_LABELS["pytorch_cpu"],
        color=COLORS["pytorch_cpu"],
        edgecolor="white",
        linewidth=0.5,
    )

    _add_bar_labels(ax, b1)
    _add_bar_labels(ax, b2)
    _add_bar_labels(ax, b3)

    tracker = data.get("tracker", "bytetrack").replace("_", " ").title()
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("IDF1 (%)", fontsize=12)
    ax.set_title(f"YOLO26 Tracking IDF1 Comparison ({tracker})", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in models])
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def create_tracking_fps_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> bool:
    """Create tracking FPS comparison bar chart across backends.

    Args:
        data: Combined tracking benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution.

    Returns:
        True if chart was created successfully.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    tracking = data.get("tracking", {})
    models = _extract_models(tracking)
    if not models:
        return False

    mlx_fps = [tracking[m[2]].get("mlx", {}).get("fps", 0) for m in models]
    mps_fps = [tracking[m[2]].get("pytorch_mps", {}).get("fps", 0) for m in models]
    cpu_fps = [tracking[m[2]].get("pytorch_cpu", {}).get("fps", 0) for m in models]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(models))
    width = 0.25

    b1 = ax.bar(
        x - width,
        mlx_fps,
        width,
        label=BACKEND_LABELS["mlx"],
        color=COLORS["mlx"],
        edgecolor="white",
        linewidth=0.5,
    )
    b2 = ax.bar(
        x,
        mps_fps,
        width,
        label=BACKEND_LABELS["pytorch_mps"],
        color=COLORS["pytorch_mps"],
        edgecolor="white",
        linewidth=0.5,
    )
    b3 = ax.bar(
        x + width,
        cpu_fps,
        width,
        label=BACKEND_LABELS["pytorch_cpu"],
        color=COLORS["pytorch_cpu"],
        edgecolor="white",
        linewidth=0.5,
    )

    _add_bar_labels(ax, b1)
    _add_bar_labels(ax, b2)
    _add_bar_labels(ax, b3)

    tracker = data.get("tracker", "bytetrack").replace("_", " ").title()
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Throughput (FPS)", fontsize=12)
    ax.set_title(
        f"YOLO26 Tracking Throughput Comparison ({tracker})", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in models])
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def create_tracking_speedup_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> bool:
    """Create tracking speedup comparison bar chart (MLX vs CPU / MPS).

    Args:
        data: Combined tracking benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution.

    Returns:
        True if chart was created successfully.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    speedups = data.get("speedups", {})
    tracking = data.get("tracking", {})
    models = _extract_models(tracking)
    if not models or not speedups:
        logger.warning("  ⚠️  No speedup data available")
        return False

    mlx_vs_cpu = [speedups.get(m[2], {}).get("mlx_vs_cpu", 0) for m in models]
    mlx_vs_mps = [speedups.get(m[2], {}).get("mlx_vs_mps", 0) for m in models]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(models))
    width = 0.35

    b1 = ax.bar(
        x - width / 2,
        mlx_vs_cpu,
        width,
        label="MLX vs CPU",
        color="#2E86AB",
        edgecolor="white",
        linewidth=0.5,
    )
    b2 = ax.bar(
        x + width / 2,
        mlx_vs_mps,
        width,
        label="MLX vs MPS",
        color="#A23B72",
        edgecolor="white",
        linewidth=0.5,
    )

    _add_bar_labels(ax, b1, fmt="{:.1f}x")
    _add_bar_labels(ax, b2, fmt="{:.1f}x")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    tracker = data.get("tracker", "bytetrack").replace("_", " ").title()
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Speedup Factor", fontsize=12)
    ax.set_title(f"YOLO26 Tracking Speedup ({tracker}, MOT17)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in models])
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def create_tracking_overhead_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> bool:
    """Create stacked bar chart showing detection vs tracking overhead (MLX only).

    Args:
        data: Combined tracking benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution.

    Returns:
        True if chart was created successfully.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    tracking = data.get("tracking", {})
    models = _extract_models(tracking)
    if not models:
        return False

    det_ms = []
    trk_ms = []
    io_ms_vals = []
    labels = []

    for _, label, key in models:
        mlx = tracking[key].get("mlx", {})
        d = mlx.get("detection_ms", 0)
        t = mlx.get("tracking_ms", 0)
        io = mlx.get("io_ms", 0)
        if d > 0 or t > 0:
            det_ms.append(d)
            trk_ms.append(t)
            io_ms_vals.append(io)
            labels.append(label)

    if not labels:
        logger.warning("  ⚠️  No per-component timing data available")
        return False

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    width = 0.5

    b1 = ax.bar(
        x, det_ms, width, label="Detection", color=COLORS["mlx"], edgecolor="white", linewidth=0.5
    )
    b2 = ax.bar(
        x,
        trk_ms,
        width,
        bottom=det_ms,
        label="Tracking (Kalman + Matching)",
        color="#FE6100",
        edgecolor="white",
        linewidth=0.5,
    )
    if any(v > 0 for v in io_ms_vals):
        bottoms = [d + t for d, t in zip(det_ms, trk_ms, strict=True)]
        ax.bar(
            x,
            io_ms_vals,
            width,
            bottom=bottoms,
            label="I/O",
            color="#FFB000",
            edgecolor="white",
            linewidth=0.5,
        )

    for bar_d, bar_t in zip(b1, b2, strict=True):
        d_h = bar_d.get_height()
        t_h = bar_t.get_height()
        if d_h > 0:
            ax.text(
                bar_d.get_x() + bar_d.get_width() / 2,
                d_h / 2,
                f"{d_h:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        if t_h > 0:
            ax.text(
                bar_t.get_x() + bar_t.get_width() / 2,
                d_h + t_h / 2,
                f"{t_h:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

    tracker = data.get("tracker", "bytetrack").replace("_", " ").title()
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Time per Frame (ms)", fontsize=12)
    ax.set_title(
        f"YOLO26 Tracking Overhead Breakdown — MLX ({tracker})", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def create_tracking_summary_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (14, 10),
    dpi: int = 150,
) -> bool:
    """Create a 2x2 tracking summary dashboard.

    Args:
        data: Combined tracking benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution.

    Returns:
        True if chart was created successfully.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    tracking = data.get("tracking", {})
    speedups = data.get("speedups", {})
    models = _extract_models(tracking)
    if not models:
        return False

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    x = np.arange(len(models))
    width = 0.25

    def _get(metric: str):
        return (
            [tracking[m[2]].get("mlx", {}).get(metric, 0) for m in models],
            [tracking[m[2]].get("pytorch_mps", {}).get(metric, 0) for m in models],
            [tracking[m[2]].get("pytorch_cpu", {}).get(metric, 0) for m in models],
        )

    xlabels = [m[1] for m in models]

    # 1. MOTA (top-left)
    ax = axes[0, 0]
    mlx_v, mps_v, cpu_v = _get("MOTA")
    ax.bar(x - width, mlx_v, width, label="MLX", color=COLORS["mlx"])
    ax.bar(x, mps_v, width, label="MPS", color=COLORS["pytorch_mps"])
    ax.bar(x + width, cpu_v, width, label="CPU", color=COLORS["pytorch_cpu"])
    ax.set_ylabel("MOTA (%)")
    ax.set_title("Tracking MOTA", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # 2. IDF1 (top-right)
    ax = axes[0, 1]
    mlx_v, mps_v, cpu_v = _get("IDF1")
    ax.bar(x - width, mlx_v, width, label="MLX", color=COLORS["mlx"])
    ax.bar(x, mps_v, width, label="MPS", color=COLORS["pytorch_mps"])
    ax.bar(x + width, cpu_v, width, label="CPU", color=COLORS["pytorch_cpu"])
    ax.set_ylabel("IDF1 (%)")
    ax.set_title("Tracking IDF1", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # 3. FPS (bottom-left)
    ax = axes[1, 0]
    mlx_v, mps_v, cpu_v = _get("fps")
    ax.bar(x - width, mlx_v, width, label="MLX", color=COLORS["mlx"])
    ax.bar(x, mps_v, width, label="MPS", color=COLORS["pytorch_mps"])
    ax.bar(x + width, cpu_v, width, label="CPU", color=COLORS["pytorch_cpu"])
    ax.set_ylabel("FPS")
    ax.set_title("Tracking Throughput", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # 4. Speedup (bottom-right)
    ax = axes[1, 1]
    vs_cpu = [speedups.get(m[2], {}).get("mlx_vs_cpu", 0) for m in models]
    vs_mps = [speedups.get(m[2], {}).get("mlx_vs_mps", 0) for m in models]
    ax.bar(x - width / 2, vs_cpu, width, label="vs CPU", color="#2E86AB")
    ax.bar(x + width / 2, vs_mps, width, label="vs MPS", color="#A23B72")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Speedup")
    ax.set_title("MLX Tracking Speedup", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    tracker = data.get("tracker", "bytetrack").replace("_", " ").title()
    fig.suptitle(
        f"YOLO26 Tracking Benchmark Summary ({tracker}, MOT17)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


# =============================================================================
# Main
# =============================================================================


def main():
    """Generate tracking benchmark charts from combined results JSON."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="YOLO26 Tracking Benchmark Chart Generator")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=f"Input JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI for raster formats (default: 150)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CHARTS_DIR,
        help=f"Output directory for charts (default: {CHARTS_DIR.relative_to(PROJECT_DIR)})",
    )
    args = parser.parse_args()
    ensure_runtime_dirs(PROJECT_DIR)

    logger.info("=" * 70)
    logger.info("  YOLO26 Tracking Benchmark Chart Generator")
    logger.info("=" * 70)

    # Check for matplotlib
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401

        logger.info(f"\n✅ matplotlib version: {matplotlib.__version__}")
    except ImportError:
        logger.error("\n❌ matplotlib is required for chart generation.")
        logger.error("   Install with: pip install matplotlib")
        return

    # Load results
    input_path = Path(args.input) if args.input else DEFAULT_INPUT
    logger.info(f"\n📂 Loading results from: {input_path}")

    data = load_results(input_path)
    if data is None:
        return

    # Create output directory
    charts_dir = args.output
    charts_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {charts_dir}")

    # Generate charts
    logger.info("\n📊 Generating tracking charts...")
    ext = args.format
    dpi = args.dpi

    charts = [
        ("tracking_mota", create_tracking_mota_chart, "Tracking MOTA comparison"),
        ("tracking_idf1", create_tracking_idf1_chart, "Tracking IDF1 comparison"),
        ("tracking_fps", create_tracking_fps_chart, "Tracking throughput (FPS)"),
        ("tracking_speedup", create_tracking_speedup_chart, "Tracking speedup comparison"),
        ("tracking_overhead", create_tracking_overhead_chart, "Tracking overhead breakdown"),
        ("tracking_summary", create_tracking_summary_chart, "Tracking summary dashboard"),
    ]

    created = 0
    for name, func, description in charts:
        output_path = charts_dir / f"yolo26_{name}.{ext}"
        logger.info(f"  • {description}...")

        if func(data, output_path, dpi=dpi):
            logger.info(f"    ✅ {output_path.name}")
            created += 1
        else:
            logger.info("    ⏭️  skipped (no data)")

    logger.info(f"\n✅ Generated {created}/{len(charts)} charts")
    logger.info(f"📁 Charts saved to: {charts_dir}")
    logger.info("\n✨ Tracking chart generation complete!")


if __name__ == "__main__":
    main()
