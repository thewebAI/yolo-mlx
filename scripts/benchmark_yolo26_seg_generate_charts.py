#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 Segmentation Benchmark Chart Generator
=============================================
Generates visualization charts from benchmark results.

This script reads the combined benchmark results from:
    ../results/yolo26_seg_benchmark_combined.json

And generates charts in:
    ../results/charts/

Charts generated:
- Inference latency comparison (bar chart)
- Inference throughput (FPS) comparison
- Training time comparison
- Speedup comparison (MLX vs CPU, MLX vs MPS)
- Memory usage comparison
- Accuracy (mask mAP when available, else box mAP) comparison

Empty-chart policy:
    A chart is only written when it has real data to show. Series for a
    backend that was not benchmarked (e.g. PyTorch MPS/CPU when only MLX was
    run) are omitted instead of drawn as empty bars, and any chart or
    sub-panel that would end up with no data is skipped entirely.

Usage:
    python benchmark_yolo26_seg_generate_charts.py
    python benchmark_yolo26_seg_generate_charts.py --input custom_results.json
    python benchmark_yolo26_seg_generate_charts.py --format pdf  # For publications
    python benchmark_yolo26_seg_generate_charts.py --output custom_charts/

Output:
    ../results/charts/*.png (default, overridable via --output)
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
DEFAULT_INPUT = RESULTS_DIR / "yolo26_seg_benchmark_combined.json"

MODEL_SIZES = ["n", "s", "m", "l", "x"]
MODEL_LABELS = ["YOLO26n-seg", "YOLO26s-seg", "YOLO26m-seg", "YOLO26l-seg", "YOLO26x-seg"]

# Colorblind-friendly colors (IBM Design Library)
COLORS = {
    "mlx": "#648FFF",  # Blue
    "pytorch_mps": "#785EF0",  # Purple
    "pytorch_cpu": "#DC267F",  # Magenta
}

BACKEND_LABELS = {
    "mlx": "MLX (Apple GPU)",
    "pytorch_mps": "PyTorch MPS",
    "pytorch_cpu": "PyTorch CPU",
}

BACKEND_ORDER = ("mlx", "pytorch_mps", "pytorch_cpu")

SPEEDUP_COLORS = {"mlx_vs_cpu": "#2E86AB", "mlx_vs_mps": "#A23B72"}
SPEEDUP_LABELS = {"mlx_vs_cpu": "MLX vs CPU", "mlx_vs_mps": "MLX vs MPS"}


def _model_key(size: str) -> str:
    return f"yolo26{size}-seg"


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


def _training_has_mask_map_keys(training_data: dict) -> bool:
    for size in MODEL_SIZES:
        mk = _model_key(size)
        backends = training_data.get(mk, {})
        for b in BACKEND_ORDER:
            if "mAP50_mask" in backends.get(b, {}):
                return True
    return False


def _map50_for_backend(backends: dict, backend_name: str, use_mask: bool) -> float:
    sub = backends.get(backend_name, {})
    if use_mask and "mAP50_mask" in sub:
        return float(sub.get("mAP50_mask", 0) or 0)
    return float(sub.get("mAP50", 0) or 0)


def _map95_for_backend(backends: dict, backend_name: str, use_mask: bool) -> float:
    sub = backends.get(backend_name, {})
    if use_mask and "mAP50-95_mask" in sub:
        return float(sub.get("mAP50-95_mask", 0) or 0)
    return float(sub.get("mAP50-95", 0) or 0)


def _speedup_present(speedup_section: dict) -> bool:
    """True if any model has a positive speedup ratio in this section."""
    for entry in speedup_section.values():
        if (entry.get("mlx_vs_cpu", 0) or 0) > 0 or (entry.get("mlx_vs_mps", 0) or 0) > 0:
            return True
    return False


# =============================================================================
# Data Loading
# =============================================================================


def load_results(path: Path) -> dict | None:
    """Load benchmark results from JSON file.

    Args:
        path: Path to combined results JSON file

    Returns:
        Parsed JSON data or None if file doesn't exist
    """
    if not path.exists():
        logger.error(f"❌ Results file not found: {path}")
        logger.error("   Run benchmark_yolo26_seg_collect_results.py first.")
        return None

    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"❌ Error loading {path}: {e}")
        return None


# =============================================================================
# Chart Generation Functions
# =============================================================================


def _collect_inference_metric(inference_data: dict, metric: str) -> tuple[list, dict]:
    """Collect per-model, per-backend values for an inference metric.

    Args:
        inference_data: The "inference" section of the combined results.
        metric: Metric key to extract per backend (e.g. "mean_ms", "fps").

    Returns:
        (model_labels, values_by_backend) including only models that have at
        least one backend with data for the metric.
    """
    models = []
    vals = {k: [] for k in BACKEND_ORDER}

    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = _model_key(size)
        if model_key not in inference_data:
            continue
        backends = inference_data[model_key]
        row = {k: (backends.get(k, {}).get(metric, 0) or 0) for k in BACKEND_ORDER}
        if not _has_values(row.values()):
            continue
        models.append(label)
        for k in BACKEND_ORDER:
            vals[k].append(row[k])

    return models, vals


def create_inference_latency_chart(
    data: dict, output_path: Path, figsize: tuple = (12, 6), dpi: int = 150
) -> bool:
    """Create inference latency comparison bar chart.

    Args:
        data: Combined benchmark data.
        output_path: Path to save the chart.
        figsize: Figure size (width, height).
        dpi: Output resolution (dots per inch).

    Returns:
        True if the chart was written, False if skipped for lack of data.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  ⚠️  matplotlib not available, skipping chart")
        return False

    inference_data = data.get("inference", {})
    if not inference_data:
        logger.warning("  ⚠️  No inference data available")
        return False

    models, vals = _collect_inference_metric(inference_data, "mean_ms")
    series = _present_series(vals)
    if not models or not series:
        return False

    fig, ax = plt.subplots(figsize=figsize)
    _grouped_bar(ax, models, series, value_fmt="{:.1f}")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Inference Latency (ms)", fontsize=12)
    ax.set_title("YOLO26 Segmentation Inference Latency Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def create_inference_fps_chart(
    data: dict, output_path: Path, figsize: tuple = (12, 6), dpi: int = 150
) -> bool:
    """Create inference throughput (FPS) comparison bar chart.

    Args:
        data: Combined benchmark data.
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

    inference_data = data.get("inference", {})
    if not inference_data:
        return False

    models, vals = _collect_inference_metric(inference_data, "fps")
    series = _present_series(vals)
    if not models or not series:
        return False

    fig, ax = plt.subplots(figsize=figsize)
    _grouped_bar(ax, models, series, value_fmt="{:.1f}")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Throughput (FPS)", fontsize=12)
    ax.set_title(
        "YOLO26 Segmentation Inference Throughput Comparison", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def create_training_time_chart(
    data: dict, output_path: Path, figsize: tuple = (12, 6), dpi: int = 150
) -> bool:
    """Create training time comparison bar chart.

    Args:
        data: Combined benchmark data.
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

    training_data = data.get("training", {})
    if not training_data:
        logger.warning("  ⚠️  No training data available")
        return False

    models = []
    vals = {k: [] for k in BACKEND_ORDER}
    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = _model_key(size)
        if model_key not in training_data:
            continue
        backends = training_data[model_key]
        row = {k: (backends.get(k, {}).get("training_time_seconds", 0) or 0) for k in BACKEND_ORDER}
        if not _has_values(row.values()):
            continue
        models.append(label)
        for k in BACKEND_ORDER:
            vals[k].append(row[k])

    series = _present_series(vals)
    if not models or not series:
        return False

    fig, ax = plt.subplots(figsize=figsize)
    _grouped_bar(ax, models, series, value_fmt="{:.0f}")

    config = data.get("configuration", {}).get("training", {})
    epochs = config.get("epochs", "?")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Training Time (seconds)", fontsize=12)
    ax.set_title(
        f"YOLO26 Segmentation Training Time Comparison ({epochs} epochs)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def _draw_speedup_panel(ax: Any, speedup_section: dict, title: str) -> bool:
    """Draw a single speedup panel; return False if it has no data."""
    import numpy as np

    models = []
    ratios = {"mlx_vs_cpu": [], "mlx_vs_mps": []}
    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = _model_key(size)
        if model_key not in speedup_section:
            continue
        entry = speedup_section[model_key]
        cpu = entry.get("mlx_vs_cpu", 0) or 0
        mps = entry.get("mlx_vs_mps", 0) or 0
        if cpu <= 0 and mps <= 0:
            continue
        models.append(label)
        ratios["mlx_vs_cpu"].append(cpu)
        ratios["mlx_vs_mps"].append(mps)

    series = [
        (SPEEDUP_LABELS[k], SPEEDUP_COLORS[k], ratios[k])
        for k in ("mlx_vs_cpu", "mlx_vs_mps")
        if _has_values(ratios[k])
    ]
    if not models or not series:
        return False

    x = np.arange(len(models))
    width = 0.8 / len(series)
    for i, (label, color, values) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width, label=label, color=color, edgecolor="white", linewidth=0.5
        )
        for bar in bars:
            height = bar.get_height()
            if height and height > 0:
                ax.annotate(
                    f"{height:.1f}x",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Speedup Factor", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    return True


def create_speedup_chart(
    data: dict, output_path: Path, figsize: tuple = (12, 6), dpi: int = 150
) -> bool:
    """Create speedup comparison chart with only the panels that have data.

    Args:
        data: Combined benchmark data.
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
    inference_speedups = speedups.get("inference", {})
    training_speedups = speedups.get("training", {})

    panels = []
    if _speedup_present(inference_speedups):
        panels.append(("Inference Speedup", inference_speedups))
    if _speedup_present(training_speedups):
        panels.append(("Training Speedup", training_speedups))

    if not panels:
        logger.warning("  ⚠️  No speedup data available")
        return False

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), figsize[1]), squeeze=False)
    drawn = False
    for ax, (title, section) in zip(axes[0], panels, strict=True):
        drawn = _draw_speedup_panel(ax, section, title) or drawn

    if not drawn:
        plt.close()
        return False

    fig.suptitle(
        "YOLO26 Segmentation MLX Speedup Comparison", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def create_accuracy_chart(
    data: dict, output_path: Path, figsize: tuple = (12, 6), dpi: int = 150
) -> bool:
    """Create accuracy (mask or box mAP) comparison chart.

    Prefers mAP50_mask and mAP50-95_mask when present; otherwise uses box mAP.

    Args:
        data: Combined benchmark data.
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

    training_data = data.get("training", {})
    if not training_data:
        return False

    use_mask = _training_has_mask_map_keys(training_data)

    models = []
    map50 = {k: [] for k in BACKEND_ORDER}
    map95 = {k: [] for k in BACKEND_ORDER}
    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = _model_key(size)
        if model_key not in training_data:
            continue
        backends = training_data[model_key]
        row50 = {k: _map50_for_backend(backends, k, use_mask) for k in BACKEND_ORDER}
        row95 = {k: _map95_for_backend(backends, k, use_mask) for k in BACKEND_ORDER}
        if not _has_values(row50.values()) and not _has_values(row95.values()):
            continue
        models.append(label)
        for k in BACKEND_ORDER:
            map50[k].append(row50[k])
            map95[k].append(row95[k])

    series50 = _present_series(map50)
    series95 = _present_series(map95)
    if not models or (not series50 and not series95):
        logger.warning("  ⚠️  No accuracy data available")
        return False

    if use_mask:
        ylabel50, title50 = "mAP50 (mask)", "Mask mAP@IoU=0.50"
        ylabel95, title95 = "mAP50-95 (mask)", "Mask mAP@IoU=0.50:0.95"
    else:
        ylabel50, title50 = "mAP50", "mAP@IoU=0.50"
        ylabel95, title95 = "mAP50-95", "mAP@IoU=0.50:0.95"

    panels = []
    if series50:
        panels.append((series50, ylabel50, title50))
    if series95:
        panels.append((series95, ylabel95, title95))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), figsize[1]), squeeze=False)
    for ax, (series, ylabel, title) in zip(axes[0], panels, strict=True):
        _grouped_bar(ax, models, series, annotate=False)
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticklabels(models, fontsize=9)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.0)

    fig.suptitle(
        "YOLO26 Segmentation Accuracy Comparison (After Training)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def create_memory_chart(
    data: dict, output_path: Path, figsize: tuple = (12, 6), dpi: int = 150
) -> bool:
    """Create memory usage comparison bar chart.

    Args:
        data: Combined benchmark data.
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

    training_data = data.get("training", {})
    if not training_data:
        return False

    models = []
    vals = {k: [] for k in BACKEND_ORDER}
    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = _model_key(size)
        if model_key not in training_data:
            continue
        backends = training_data[model_key]
        row = {
            "mlx": backends.get("mlx", {}).get("peak_memory_mb", 0) or 0,
            "pytorch_mps": (
                backends.get("pytorch_mps", {}).get(
                    "driver_memory_mb", backends.get("pytorch_mps", {}).get("current_memory_mb", 0)
                )
                or 0
            ),
            "pytorch_cpu": backends.get("pytorch_cpu", {}).get("peak_memory_mb", 0) or 0,
        }
        if not _has_values(row.values()):
            continue
        models.append(label)
        for k in BACKEND_ORDER:
            vals[k].append(row[k])

    series = _present_series(vals)
    if not models or not series:
        logger.warning("  ⚠️  No memory data available")
        return False

    fig, ax = plt.subplots(figsize=figsize)
    _grouped_bar(ax, models, series, value_fmt="{:.0f}")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax.set_title("YOLO26 Segmentation Training Memory Usage", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def create_summary_chart(
    data: dict, output_path: Path, figsize: tuple = (14, 10), dpi: int = 150
) -> bool:
    """Create a summary dashboard, including only panels that have data.

    Args:
        data: Combined benchmark data.
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

    inference_data = data.get("inference", {})
    training_data = data.get("training", {})
    speedups = data.get("speedups", {})
    inf_speed = speedups.get("inference", {})
    train_speed = speedups.get("training", {})

    model_list = [
        (s, label, _model_key(s))
        for s, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True)
        if _model_key(s) in inference_data or _model_key(s) in training_data
    ]
    if not model_list:
        return False

    labels = [m[1] for m in model_list]

    def _backend_values(source: dict, model_field: str) -> dict:
        return {
            k: [source.get(m[2], {}).get(k, {}).get(model_field, 0) or 0 for m in model_list]
            for k in BACKEND_ORDER
        }

    def latency_panel(ax: Any) -> bool:
        series = _present_series(_backend_values(inference_data, "mean_ms"))
        if not series:
            return False
        _grouped_bar(ax, labels, series, annotate=False)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Inference Latency", fontweight="bold")
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        return True

    def training_panel(ax: Any) -> bool:
        series = _present_series(_backend_values(training_data, "training_time_seconds"))
        if not series:
            return False
        _grouped_bar(ax, labels, series, annotate=False)
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Training Time", fontweight="bold")
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        return True

    def _speedup_panel(ax: Any, section: dict, title: str) -> bool:
        x = np.arange(len(model_list))
        vs_cpu = [section.get(m[2], {}).get("mlx_vs_cpu", 0) or 0 for m in model_list]
        vs_mps = [section.get(m[2], {}).get("mlx_vs_mps", 0) or 0 for m in model_list]
        series = []
        if _has_values(vs_cpu):
            series.append(("vs CPU", SPEEDUP_COLORS["mlx_vs_cpu"], vs_cpu))
        if _has_values(vs_mps):
            series.append(("vs MPS", SPEEDUP_COLORS["mlx_vs_mps"], vs_mps))
        if not series:
            return False
        n = len(series)
        bar_w = 0.8 / n
        for i, (label, color, values) in enumerate(series):
            offset = (i - (n - 1) / 2) * bar_w
            ax.bar(x + offset, values, bar_w, label=label, color=color)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Speedup")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        return True

    def inf_speedup_panel(ax: Any) -> bool:
        return _speedup_panel(ax, inf_speed, "MLX Inference Speedup")

    def train_speedup_panel(ax: Any) -> bool:
        return _speedup_panel(ax, train_speed, "MLX Training Speedup")

    candidate_panels = [latency_panel, training_panel, inf_speedup_panel, train_speedup_panel]

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

    device_info = data.get("device_info", {})
    device_str = device_info.get("cpu", device_info.get("processor", ""))

    fig.suptitle("YOLO26 Segmentation Benchmark Summary", fontsize=16, fontweight="bold", y=1.02)
    if device_str:
        fig.text(0.5, 0.98, f"Device: {device_str}", ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Generate benchmark visualization charts from combined results JSON."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="YOLO26 Segmentation Benchmark Chart Generator")
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
    logger.info("  YOLO26 Segmentation Benchmark Chart Generator")
    logger.info("=" * 70)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401

        logger.info(f"\n✅ matplotlib version: {matplotlib.__version__}")
    except ImportError:
        logger.error("\n❌ matplotlib is required for chart generation.")
        logger.error("   Install with: pip install matplotlib")
        return

    input_path = Path(args.input) if args.input else DEFAULT_INPUT
    logger.info(f"\n📂 Loading results from: {input_path}")

    data = load_results(input_path)
    if data is None:
        return

    charts_dir = args.output
    charts_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {charts_dir}")

    logger.info("\n📊 Generating charts...")
    ext = args.format
    dpi = args.dpi

    charts = [
        ("inference_latency", create_inference_latency_chart, "Inference latency comparison"),
        ("inference_fps", create_inference_fps_chart, "Inference throughput (FPS)"),
        ("training_time", create_training_time_chart, "Training time comparison"),
        ("speedup", create_speedup_chart, "Speedup comparison"),
        ("accuracy", create_accuracy_chart, "Accuracy (mask/box mAP) comparison"),
        ("memory", create_memory_chart, "Memory usage comparison"),
        ("summary", create_summary_chart, "Summary dashboard"),
    ]

    created = 0
    skipped = 0
    for name, func, description in charts:
        output_path = charts_dir / f"yolo26_seg_{name}.{ext}"
        logger.info(f"  • {description}...")

        if func(data, output_path, dpi=dpi):
            logger.info(f"✅ {output_path.name}")
            created += 1
        else:
            logger.info("⏭️  skipped (no data for this chart)")
            skipped += 1

    logger.info(
        f"\n✅ Generated {created}/{len(charts)} charts ({skipped} skipped for lack of data)"
    )
    logger.info(f"📁 Charts saved to: {charts_dir}")
    logger.info("\n✨ Chart generation complete!")


if __name__ == "__main__":
    main()
