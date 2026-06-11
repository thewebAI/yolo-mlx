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

Empty-chart policy:
    A chart is only written when it has real data to show. Series for a
    backend that was not benchmarked (e.g. PyTorch MPS/CPU when only MLX was
    run) are omitted instead of drawn as empty bars, and any chart or
    sub-panel that would end up with no data is skipped entirely.

Usage:
    python benchmark_tracking_generate_charts.py
    python benchmark_tracking_generate_charts.py --input custom_results.json
    python benchmark_tracking_generate_charts.py --format pdf
    python benchmark_tracking_generate_charts.py --output custom_charts/
"""

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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

BACKEND_ORDER = ("mlx", "pytorch_mps", "pytorch_cpu")

SPEEDUP_COLORS = {"mlx_vs_cpu": "#2E86AB", "mlx_vs_mps": "#A23B72"}
SPEEDUP_LABELS = {"mlx_vs_cpu": "MLX vs CPU", "mlx_vs_mps": "MLX vs MPS"}


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


def _has_values(values: Iterable) -> bool:
    """True if at least one value in the iterable is a positive number."""
    return any((v or 0) > 0 for v in values)


def _present_series(values_by_backend: dict) -> list:
    """Return [(label, color, values), ...] for backends that have real data."""
    series = []
    for key in BACKEND_ORDER:
        vals = values_by_backend.get(key, [])
        if _has_values(vals):
            series.append((BACKEND_LABELS[key], COLORS[key], vals))
    return series


def _grouped_bar(
    ax: Any,
    model_labels: list,
    series: list,
    *,
    value_fmt: str = "{:.1f}",
    annotate: bool = True,
) -> Any:
    """Draw a grouped bar chart for the given (label, color, values) series.

    Args:
        ax: Matplotlib axis to draw on.
        model_labels: X-axis category labels.
        series: List of (label, color, values) tuples to plot.
        value_fmt: Format string used when annotating bar heights.
        annotate: Whether to draw value labels above bars.

    Returns:
        The x-position array, or None if there is no series to draw.
    """
    import numpy as np

    if not series:
        return None

    x = np.arange(len(model_labels))
    n = len(series)
    width = 0.8 / n

    for i, (label, color, values) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        if annotate:
            for bar in bars:
                height = bar.get_height()
                if height and height > 0:
                    ax.annotate(
                        value_fmt.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    return x


def _collect_metric(tracking: dict, models: list, metric: str) -> tuple[list, dict]:
    """Collect per-model, per-backend values for a tracking metric.

    Only keeps models that have at least one backend with data for the metric.

    Args:
        tracking: The "tracking" section of the combined results.
        models: List of (size, label, model_key) tuples to consider.
        metric: Metric key to extract per backend (e.g. "MOTA", "fps").

    Returns:
        (model_labels, values_by_backend) for models that have data.
    """
    kept_labels = []
    vals = {k: [] for k in BACKEND_ORDER}
    for _, label, key in models:
        backends = tracking.get(key, {})
        row = {k: (backends.get(k, {}).get(metric, 0) or 0) for k in BACKEND_ORDER}
        if not _has_values(row.values()):
            continue
        kept_labels.append(label)
        for k in BACKEND_ORDER:
            vals[k].append(row[k])
    return kept_labels, vals


def _collect_speedups(speedups: dict, models: list) -> tuple[list, list]:
    """Collect per-model MLX-vs-CPU/MPS speedups, keeping only models with data.

    Args:
        speedups: The "speedups" section of the combined results.
        models: List of (size, label, model_key) tuples to consider.

    Returns:
        (model_labels, series) where series is a list of (label, color, values)
        for the speedup comparisons that actually have data.
    """
    kept_labels = []
    vs_cpu = []
    vs_mps = []
    for _, label, key in models:
        entry = speedups.get(key, {})
        cpu = entry.get("mlx_vs_cpu", 0) or 0
        mps = entry.get("mlx_vs_mps", 0) or 0
        if cpu <= 0 and mps <= 0:
            continue
        kept_labels.append(label)
        vs_cpu.append(cpu)
        vs_mps.append(mps)
    series = [
        (SPEEDUP_LABELS["mlx_vs_cpu"], SPEEDUP_COLORS["mlx_vs_cpu"], vs_cpu),
        (SPEEDUP_LABELS["mlx_vs_mps"], SPEEDUP_COLORS["mlx_vs_mps"], vs_mps),
    ]
    series = [s for s in series if _has_values(s[2])]
    return kept_labels, series


# =============================================================================
# Chart Generation Functions
# =============================================================================


def _metric_bar_chart(
    data: dict,
    output_path: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
    legend_loc: str,
    figsize: tuple,
    dpi: int,
) -> bool:
    """Shared grouped-bar chart for a single tracking metric (skips empty)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  ⚠️  matplotlib not available, skipping chart")
        return False

    tracking = data.get("tracking", {})
    models = _extract_models(tracking)
    if not models:
        logger.warning("  ⚠️  No tracking data available")
        return False

    kept, vals = _collect_metric(tracking, models, metric)
    series = _present_series(vals)
    if not kept or not series:
        return False

    fig, ax = plt.subplots(figsize=figsize)
    _grouped_bar(ax, kept, series, value_fmt="{:.1f}")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc=legend_loc)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def _tracker_name(data: dict) -> str:
    return data.get("tracker", "bytetrack").replace("_", " ").title()


def create_tracking_mota_chart(
    data: dict, output_path: Path, figsize: tuple = (12, 6), dpi: int = 150
) -> bool:
    """Create MOTA comparison bar chart across backends.

    Args:
        data: Combined tracking benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution (dots per inch).

    Returns:
        True if the chart was written, False if skipped for lack of data.
    """
    return _metric_bar_chart(
        data,
        output_path,
        metric="MOTA",
        ylabel="MOTA (%)",
        title=f"YOLO26 Tracking MOTA Comparison ({_tracker_name(data)})",
        legend_loc="lower right",
        figsize=figsize,
        dpi=dpi,
    )


def create_tracking_idf1_chart(
    data: dict, output_path: Path, figsize: tuple = (12, 6), dpi: int = 150
) -> bool:
    """Create IDF1 comparison bar chart across backends.

    Args:
        data: Combined tracking benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution (dots per inch).

    Returns:
        True if the chart was written, False if skipped for lack of data.
    """
    return _metric_bar_chart(
        data,
        output_path,
        metric="IDF1",
        ylabel="IDF1 (%)",
        title=f"YOLO26 Tracking IDF1 Comparison ({_tracker_name(data)})",
        legend_loc="lower right",
        figsize=figsize,
        dpi=dpi,
    )


def create_tracking_fps_chart(
    data: dict, output_path: Path, figsize: tuple = (12, 6), dpi: int = 150
) -> bool:
    """Create tracking FPS comparison bar chart across backends.

    Args:
        data: Combined tracking benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution (dots per inch).

    Returns:
        True if the chart was written, False if skipped for lack of data.
    """
    return _metric_bar_chart(
        data,
        output_path,
        metric="fps",
        ylabel="Throughput (FPS)",
        title=f"YOLO26 Tracking Throughput Comparison ({_tracker_name(data)})",
        legend_loc="upper right",
        figsize=figsize,
        dpi=dpi,
    )


def create_tracking_speedup_chart(
    data: dict, output_path: Path, figsize: tuple = (12, 6), dpi: int = 150
) -> bool:
    """Create tracking speedup comparison bar chart (MLX vs CPU / MPS).

    Args:
        data: Combined tracking benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution (dots per inch).

    Returns:
        True if the chart was written, False if skipped for lack of data.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    speedups = data.get("speedups", {})
    tracking = data.get("tracking", {})
    models = _extract_models(tracking)
    if not models or not speedups:
        logger.warning("  ⚠️  No speedup data available")
        return False

    kept, series = _collect_speedups(speedups, models)
    if not kept or not series:
        logger.warning("  ⚠️  No speedup data available")
        return False

    fig, ax = plt.subplots(figsize=figsize)
    _grouped_bar(ax, kept, series, value_fmt="{:.1f}x")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Speedup Factor", fontsize=12)
    ax.set_title(
        f"YOLO26 Tracking Speedup ({_tracker_name(data)}, MOT17)", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def create_tracking_overhead_chart(
    data: dict, output_path: Path, figsize: tuple = (12, 6), dpi: int = 150
) -> bool:
    """Create stacked bar chart showing detection vs tracking overhead (MLX only).

    Args:
        data: Combined tracking benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution (dots per inch).

    Returns:
        True if the chart was written, False if skipped for lack of data.
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

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Time per Frame (ms)", fontsize=12)
    ax.set_title(
        f"YOLO26 Tracking Overhead Breakdown — MLX ({_tracker_name(data)})",
        fontsize=14,
        fontweight="bold",
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
    data: dict, output_path: Path, figsize: tuple = (14, 10), dpi: int = 150
) -> bool:
    """Create a tracking summary dashboard, including only panels that have data.

    Args:
        data: Combined tracking benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution (dots per inch).

    Returns:
        True if the chart was written, False if skipped for lack of data.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    tracking = data.get("tracking", {})
    speedups = data.get("speedups", {})
    models = _extract_models(tracking)
    if not models:
        return False

    def metric_panel(metric: str, ylabel: str, title: str) -> Any:
        def _panel(ax: Any) -> bool:
            kept, vals = _collect_metric(tracking, models, metric)
            series = _present_series(vals)
            if not kept or not series:
                return False
            _grouped_bar(ax, kept, series, annotate=False)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold")
            ax.set_xticklabels(kept, fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)
            return True

        return _panel

    def speedup_panel(ax: Any) -> bool:
        kept, series = _collect_speedups(speedups, models)
        if not kept or not series:
            return False
        _grouped_bar(ax, kept, series, annotate=False)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Speedup")
        ax.set_title("MLX Tracking Speedup", fontweight="bold")
        ax.set_xticklabels(kept, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        return True

    candidate_panels = [
        metric_panel("MOTA", "MOTA (%)", "Tracking MOTA"),
        metric_panel("IDF1", "IDF1 (%)", "Tracking IDF1"),
        metric_panel("fps", "FPS", "Tracking Throughput"),
        speedup_panel,
    ]

    available = []
    for panel in candidate_panels:
        probe_fig, probe_ax = plt.subplots()
        if panel(probe_ax):
            available.append(panel)
        plt.close(probe_fig)

    if not available:
        return False

    ncols = 2 if len(available) > 1 else 1
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.flatten()
    for ax, panel in zip(flat, available, strict=False):
        panel(ax)
    for ax in flat[len(available) :]:
        ax.axis("off")

    fig.suptitle(
        f"YOLO26 Tracking Benchmark Summary ({_tracker_name(data)}, MOT17)",
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


def main() -> None:
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
    skipped = 0
    for name, func, description in charts:
        output_path = charts_dir / f"yolo26_{name}.{ext}"
        logger.info(f"  • {description}...")

        if func(data, output_path, dpi=dpi):
            logger.info(f"    ✅ {output_path.name}")
            created += 1
        else:
            logger.info("    ⏭️  skipped (no data for this chart)")
            skipped += 1

    logger.info(
        f"\n✅ Generated {created}/{len(charts)} charts ({skipped} skipped for lack of data)"
    )
    logger.info(f"📁 Charts saved to: {charts_dir}")
    logger.info("\n✨ Tracking chart generation complete!")


if __name__ == "__main__":
    main()
