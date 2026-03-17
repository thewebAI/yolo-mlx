#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 Benchmark Chart Generator
=================================
Generates visualization charts from benchmark results.

This script reads the combined benchmark results from:
    ../results/yolo26_benchmark_combined.json

And generates charts in:
    ../results/charts/

Charts generated:
- Inference latency comparison (bar chart)
- Inference throughput (FPS) comparison
- Training time comparison
- Speedup comparison (MLX vs CPU, MLX vs MPS)
- Memory usage comparison
- Accuracy (mAP) comparison

Usage:
    python benchmark_yolo26_generate_charts.py
    python benchmark_yolo26_generate_charts.py --input custom_results.json
    python benchmark_yolo26_generate_charts.py --format pdf  # For publications
    python benchmark_yolo26_generate_charts.py --output custom_charts/

Output:
    ../results/charts/*.png (default, overridable via --output)
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
DEFAULT_INPUT = RESULTS_DIR / "yolo26_benchmark_combined.json"

MODEL_SIZES = ["n", "s", "m", "l", "x"]
MODEL_LABELS = ["YOLO26n", "YOLO26s", "YOLO26m", "YOLO26l", "YOLO26x"]

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
        logger.error("   Run benchmark_yolo26_collect_results.py first.")
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


def create_inference_latency_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> bool:
    """Create inference latency comparison bar chart.

    Args:
        data: Combined benchmark data
        output_path: Path to save the chart
        figsize: Figure size (width, height)
        dpi: Output resolution (dots per inch)

    Returns:
        True if chart was created successfully
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("  ⚠️  matplotlib not available, skipping chart")
        return False

    inference_data = data.get("inference", {})
    if not inference_data:
        logger.warning("  ⚠️  No inference data available")
        return False

    # Extract data for each backend
    models = []
    mlx_times = []
    mps_times = []
    cpu_times = []

    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = f"yolo26{size}"
        if model_key in inference_data:
            models.append(label)
            backends = inference_data[model_key]
            mlx_times.append(backends.get("mlx", {}).get("mean_ms", 0))
            mps_times.append(backends.get("pytorch_mps", {}).get("mean_ms", 0))
            cpu_times.append(backends.get("pytorch_cpu", {}).get("mean_ms", 0))

    if not models:
        logger.warning("  ⚠️  No models found in inference data")
        return False

    # Create chart
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        mlx_times,
        width,
        label=BACKEND_LABELS["mlx"],
        color=COLORS["mlx"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x,
        mps_times,
        width,
        label=BACKEND_LABELS["pytorch_mps"],
        color=COLORS["pytorch_mps"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars3 = ax.bar(
        x + width,
        cpu_times,
        width,
        label=BACKEND_LABELS["pytorch_cpu"],
        color=COLORS["pytorch_cpu"],
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels on bars
    def add_labels(bars):
        """Annotate each bar with its numeric value."""
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Inference Latency (ms)", fontsize=12)
    ax.set_title("YOLO26 Inference Latency Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return True


def create_inference_fps_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> bool:
    """Create inference throughput (FPS) comparison bar chart.

    Args:
        data: Combined benchmark data
        output_path: Path to save the chart
        figsize: Figure size (width, height)
        dpi: Output resolution (dots per inch)

    Returns:
        True if chart was created successfully
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    inference_data = data.get("inference", {})
    if not inference_data:
        return False

    # Extract data
    models = []
    mlx_fps = []
    mps_fps = []
    cpu_fps = []

    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = f"yolo26{size}"
        if model_key in inference_data:
            models.append(label)
            backends = inference_data[model_key]
            mlx_fps.append(backends.get("mlx", {}).get("fps", 0))
            mps_fps.append(backends.get("pytorch_mps", {}).get("fps", 0))
            cpu_fps.append(backends.get("pytorch_cpu", {}).get("fps", 0))

    if not models:
        return False

    # Create chart
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        mlx_fps,
        width,
        label=BACKEND_LABELS["mlx"],
        color=COLORS["mlx"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x,
        mps_fps,
        width,
        label=BACKEND_LABELS["pytorch_mps"],
        color=COLORS["pytorch_mps"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars3 = ax.bar(
        x + width,
        cpu_fps,
        width,
        label=BACKEND_LABELS["pytorch_cpu"],
        color=COLORS["pytorch_cpu"],
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels
    def add_labels(bars):
        """Annotate each bar with its numeric value."""
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Throughput (FPS)", fontsize=12)
    ax.set_title("YOLO26 Inference Throughput Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return True


def create_training_time_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> bool:
    """Create training time comparison bar chart.

    Args:
        data: Combined benchmark data
        output_path: Path to save the chart
        figsize: Figure size (width, height)
        dpi: Output resolution (dots per inch)

    Returns:
        True if chart was created successfully
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    training_data = data.get("training", {})
    if not training_data:
        logger.warning("  ⚠️  No training data available")
        return False

    # Extract data
    models = []
    mlx_times = []
    mps_times = []
    cpu_times = []

    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = f"yolo26{size}"
        if model_key in training_data:
            models.append(label)
            backends = training_data[model_key]
            mlx_times.append(backends.get("mlx", {}).get("training_time_seconds", 0))
            mps_times.append(backends.get("pytorch_mps", {}).get("training_time_seconds", 0))
            cpu_times.append(backends.get("pytorch_cpu", {}).get("training_time_seconds", 0))

    if not models:
        return False

    # Create chart
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        mlx_times,
        width,
        label=BACKEND_LABELS["mlx"],
        color=COLORS["mlx"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x,
        mps_times,
        width,
        label=BACKEND_LABELS["pytorch_mps"],
        color=COLORS["pytorch_mps"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars3 = ax.bar(
        x + width,
        cpu_times,
        width,
        label=BACKEND_LABELS["pytorch_cpu"],
        color=COLORS["pytorch_cpu"],
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels
    def add_labels(bars):
        """Annotate each bar with its rounded numeric value."""
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    # Get epochs from config for title
    config = data.get("configuration", {}).get("training", {})
    epochs = config.get("epochs", "?")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Training Time (seconds)", fontsize=12)
    ax.set_title(
        f"YOLO26 Training Time Comparison ({epochs} epochs)", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return True


def create_speedup_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> bool:
    """Create speedup comparison bar chart.

    Args:
        data: Combined benchmark data
        output_path: Path to save the chart
        figsize: Figure size (width, height)
        dpi: Output resolution (dots per inch)

    Returns:
        True if chart was created successfully
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    speedups = data.get("speedups", {})
    inference_speedups = speedups.get("inference", {})
    training_speedups = speedups.get("training", {})

    if not inference_speedups and not training_speedups:
        logger.warning("  ⚠️  No speedup data available")
        return False

    # Extract data
    models = []
    inf_mlx_vs_cpu = []
    inf_mlx_vs_mps = []
    train_mlx_vs_cpu = []
    train_mlx_vs_mps = []

    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = f"yolo26{size}"

        has_inf = model_key in inference_speedups
        has_train = model_key in training_speedups

        if has_inf or has_train:
            models.append(label)

            if has_inf:
                inf_mlx_vs_cpu.append(inference_speedups[model_key].get("mlx_vs_cpu", 0))
                inf_mlx_vs_mps.append(inference_speedups[model_key].get("mlx_vs_mps", 0))
            else:
                inf_mlx_vs_cpu.append(0)
                inf_mlx_vs_mps.append(0)

            if has_train:
                train_mlx_vs_cpu.append(training_speedups[model_key].get("mlx_vs_cpu", 0))
                train_mlx_vs_mps.append(training_speedups[model_key].get("mlx_vs_mps", 0))
            else:
                train_mlx_vs_cpu.append(0)
                train_mlx_vs_mps.append(0)

    if not models:
        return False

    # Create chart with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    x = np.arange(len(models))
    width = 0.35

    # Inference speedups
    bars1 = ax1.bar(
        x - width / 2,
        inf_mlx_vs_cpu,
        width,
        label="MLX vs CPU",
        color="#2E86AB",
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax1.bar(
        x + width / 2,
        inf_mlx_vs_mps,
        width,
        label="MLX vs MPS",
        color="#A23B72",
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(
                f"{height:.1f}x",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(
                f"{height:.1f}x",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Model", fontsize=11)
    ax1.set_ylabel("Speedup Factor", fontsize=11)
    ax1.set_title("Inference Speedup", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.set_ylim(bottom=0)

    # Training speedups
    bars3 = ax2.bar(
        x - width / 2,
        train_mlx_vs_cpu,
        width,
        label="MLX vs CPU",
        color="#2E86AB",
        edgecolor="white",
        linewidth=0.5,
    )
    bars4 = ax2.bar(
        x + width / 2,
        train_mlx_vs_mps,
        width,
        label="MLX vs MPS",
        color="#A23B72",
        edgecolor="white",
        linewidth=0.5,
    )

    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(
                f"{height:.1f}x",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for bar in bars4:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(
                f"{height:.1f}x",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Model", fontsize=11)
    ax2.set_ylabel("Speedup Factor", fontsize=11)
    ax2.set_title("Training Speedup", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.set_ylim(bottom=0)

    fig.suptitle("YOLO26 MLX Speedup Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return True


def create_accuracy_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> bool:
    """Create accuracy (mAP) comparison bar chart.

    Args:
        data: Combined benchmark data
        output_path: Path to save the chart
        figsize: Figure size (width, height)
        dpi: Output resolution (dots per inch)

    Returns:
        True if chart was created successfully
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    training_data = data.get("training", {})
    if not training_data:
        return False

    # Extract data
    models = []
    mlx_map50 = []
    mps_map50 = []
    cpu_map50 = []
    mlx_map = []
    mps_map = []
    cpu_map = []

    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = f"yolo26{size}"
        if model_key in training_data:
            backends = training_data[model_key]
            # Only add if at least one backend has data
            if any(
                backends.get(b, {}).get("mAP50", 0) > 0
                for b in ["mlx", "pytorch_mps", "pytorch_cpu"]
            ):
                models.append(label)
                mlx_map50.append(backends.get("mlx", {}).get("mAP50", 0))
                mps_map50.append(backends.get("pytorch_mps", {}).get("mAP50", 0))
                cpu_map50.append(backends.get("pytorch_cpu", {}).get("mAP50", 0))
                mlx_map.append(backends.get("mlx", {}).get("mAP50-95", 0))
                mps_map.append(backends.get("pytorch_mps", {}).get("mAP50-95", 0))
                cpu_map.append(backends.get("pytorch_cpu", {}).get("mAP50-95", 0))

    if not models:
        logger.warning("  ⚠️  No accuracy data available")
        return False

    # Create chart with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    x = np.arange(len(models))
    width = 0.25

    # mAP50 chart
    ax1.bar(
        x - width,
        mlx_map50,
        width,
        label=BACKEND_LABELS["mlx"],
        color=COLORS["mlx"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.bar(
        x,
        mps_map50,
        width,
        label=BACKEND_LABELS["pytorch_mps"],
        color=COLORS["pytorch_mps"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.bar(
        x + width,
        cpu_map50,
        width,
        label=BACKEND_LABELS["pytorch_cpu"],
        color=COLORS["pytorch_cpu"],
        edgecolor="white",
        linewidth=0.5,
    )

    ax1.set_xlabel("Model", fontsize=11)
    ax1.set_ylabel("mAP50", fontsize=11)
    ax1.set_title("mAP@IoU=0.50", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, 1.0)

    # mAP50-95 chart
    ax2.bar(
        x - width,
        mlx_map,
        width,
        label=BACKEND_LABELS["mlx"],
        color=COLORS["mlx"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.bar(
        x,
        mps_map,
        width,
        label=BACKEND_LABELS["pytorch_mps"],
        color=COLORS["pytorch_mps"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.bar(
        x + width,
        cpu_map,
        width,
        label=BACKEND_LABELS["pytorch_cpu"],
        color=COLORS["pytorch_cpu"],
        edgecolor="white",
        linewidth=0.5,
    )

    ax2.set_xlabel("Model", fontsize=11)
    ax2.set_ylabel("mAP50-95", fontsize=11)
    ax2.set_title("mAP@IoU=0.50:0.95", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 1.0)

    fig.suptitle(
        "YOLO26 Accuracy Comparison (After Training)", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return True


def create_memory_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> bool:
    """Create memory usage comparison bar chart.

    Args:
        data: Combined benchmark data
        output_path: Path to save the chart
        figsize: Figure size (width, height)
        dpi: Output resolution (dots per inch)

    Returns:
        True if chart was created successfully
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    training_data = data.get("training", {})
    if not training_data:
        return False

    # Extract data - check for peak_memory_mb
    models = []
    mlx_mem = []
    mps_mem = []
    cpu_mem = []

    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = f"yolo26{size}"
        if model_key in training_data:
            backends = training_data[model_key]

            # Get memory values (different scripts use different keys)
            mlx_memory = backends.get("mlx", {}).get("peak_memory_mb", 0)
            mps_memory = backends.get("pytorch_mps", {}).get(
                "driver_memory_mb", backends.get("pytorch_mps", {}).get("current_memory_mb", 0)
            )
            cpu_memory = backends.get("pytorch_cpu", {}).get("peak_memory_mb", 0)

            if mlx_memory > 0 or mps_memory > 0 or cpu_memory > 0:
                models.append(label)
                mlx_mem.append(mlx_memory)
                mps_mem.append(mps_memory)
                cpu_mem.append(cpu_memory)

    if not models:
        logger.warning("  ⚠️  No memory data available")
        return False

    # Create chart
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        mlx_mem,
        width,
        label=BACKEND_LABELS["mlx"],
        color=COLORS["mlx"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x,
        mps_mem,
        width,
        label=BACKEND_LABELS["pytorch_mps"],
        color=COLORS["pytorch_mps"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars3 = ax.bar(
        x + width,
        cpu_mem,
        width,
        label=BACKEND_LABELS["pytorch_cpu"],
        color=COLORS["pytorch_cpu"],
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels
    def add_labels(bars):
        """Annotate each bar with its rounded numeric value."""
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax.set_title("YOLO26 Training Memory Usage", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return True


def create_summary_chart(
    data: dict,
    output_path: Path,
    figsize: tuple = (14, 10),
    dpi: int = 150,
) -> bool:
    """Create a summary dashboard with all key metrics.

    Args:
        data: Combined benchmark data
        output_path: Path to save the chart
        figsize: Figure size (width, height)
        dpi: Output resolution (dots per inch)

    Returns:
        True if chart was created successfully
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    inference_data = data.get("inference", {})
    training_data = data.get("training", {})
    speedups = data.get("speedups", {})

    if not inference_data and not training_data:
        return False

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract model data
    models = []
    for size, label in zip(MODEL_SIZES, MODEL_LABELS, strict=True):
        model_key = f"yolo26{size}"
        if model_key in inference_data or model_key in training_data:
            models.append((size, label, model_key))

    if not models:
        plt.close()
        return False

    x = np.arange(len(models))
    width = 0.25

    # 1. Inference latency (top-left)
    ax1 = axes[0, 0]
    mlx_times = [inference_data.get(m[2], {}).get("mlx", {}).get("mean_ms", 0) for m in models]
    mps_times = [
        inference_data.get(m[2], {}).get("pytorch_mps", {}).get("mean_ms", 0) for m in models
    ]
    cpu_times = [
        inference_data.get(m[2], {}).get("pytorch_cpu", {}).get("mean_ms", 0) for m in models
    ]

    ax1.bar(x - width, mlx_times, width, label="MLX", color=COLORS["mlx"])
    ax1.bar(x, mps_times, width, label="MPS", color=COLORS["pytorch_mps"])
    ax1.bar(x + width, cpu_times, width, label="CPU", color=COLORS["pytorch_cpu"])
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Inference Latency", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[1] for m in models], fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(bottom=0)

    # 2. Training time (top-right)
    ax2 = axes[0, 1]
    mlx_train = [
        training_data.get(m[2], {}).get("mlx", {}).get("training_time_seconds", 0) for m in models
    ]
    mps_train = [
        training_data.get(m[2], {}).get("pytorch_mps", {}).get("training_time_seconds", 0)
        for m in models
    ]
    cpu_train = [
        training_data.get(m[2], {}).get("pytorch_cpu", {}).get("training_time_seconds", 0)
        for m in models
    ]

    ax2.bar(x - width, mlx_train, width, label="MLX", color=COLORS["mlx"])
    ax2.bar(x, mps_train, width, label="MPS", color=COLORS["pytorch_mps"])
    ax2.bar(x + width, cpu_train, width, label="CPU", color=COLORS["pytorch_cpu"])
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Training Time", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[1] for m in models], fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(bottom=0)

    # 3. Inference speedups (bottom-left)
    ax3 = axes[1, 0]
    inf_speedups = speedups.get("inference", {})
    mlx_vs_cpu = [inf_speedups.get(m[2], {}).get("mlx_vs_cpu", 0) for m in models]
    mlx_vs_mps = [inf_speedups.get(m[2], {}).get("mlx_vs_mps", 0) for m in models]

    ax3.bar(x - width / 2, mlx_vs_cpu, width, label="vs CPU", color="#2E86AB")
    ax3.bar(x + width / 2, mlx_vs_mps, width, label="vs MPS", color="#A23B72")
    ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
    ax3.set_ylabel("Speedup")
    ax3.set_title("MLX Inference Speedup", fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([m[1] for m in models], fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_ylim(bottom=0)

    # 4. Training speedups (bottom-right)
    ax4 = axes[1, 1]
    train_speedups = speedups.get("training", {})
    train_mlx_vs_cpu = [train_speedups.get(m[2], {}).get("mlx_vs_cpu", 0) for m in models]
    train_mlx_vs_mps = [train_speedups.get(m[2], {}).get("mlx_vs_mps", 0) for m in models]

    ax4.bar(x - width / 2, train_mlx_vs_cpu, width, label="vs CPU", color="#2E86AB")
    ax4.bar(x + width / 2, train_mlx_vs_mps, width, label="vs MPS", color="#A23B72")
    ax4.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
    ax4.set_ylabel("Speedup")
    ax4.set_title("MLX Training Speedup", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels([m[1] for m in models], fontsize=9)
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)
    ax4.set_ylim(bottom=0)

    # Add device info as subtitle if available
    device_info = data.get("device_info", {})
    device_str = device_info.get("cpu", device_info.get("processor", ""))

    fig.suptitle("YOLO26 Benchmark Summary", fontsize=16, fontweight="bold", y=1.02)
    if device_str:
        fig.text(0.5, 0.98, f"Device: {device_str}", ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return True


# =============================================================================
# Main
# =============================================================================


def main():
    """Generate benchmark visualization charts from combined results JSON."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="YOLO26 Benchmark Chart Generator")
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
    logger.info("  YOLO26 Benchmark Chart Generator")
    logger.info("=" * 70)

    # Check for matplotlib
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
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
    logger.info("\n📊 Generating charts...")
    ext = args.format
    dpi = args.dpi

    charts = [
        ("inference_latency", create_inference_latency_chart, "Inference latency comparison"),
        ("inference_fps", create_inference_fps_chart, "Inference throughput (FPS)"),
        ("training_time", create_training_time_chart, "Training time comparison"),
        ("speedup", create_speedup_chart, "Speedup comparison"),
        ("accuracy", create_accuracy_chart, "Accuracy (mAP) comparison"),
        ("memory", create_memory_chart, "Memory usage comparison"),
        ("summary", create_summary_chart, "Summary dashboard"),
    ]

    created = 0
    for name, func, description in charts:
        output_path = charts_dir / f"yolo26_{name}.{ext}"
        logger.info(f"  • {description}...")

        if func(data, output_path, dpi=dpi):
            logger.info(f"✅ {output_path.name}")
            created += 1
        else:
            logger.info("⏭️  skipped (no data)")

    logger.info(f"\n✅ Generated {created}/{len(charts)} charts")
    logger.info(f"📁 Charts saved to: {charts_dir}")
    logger.info("\n✨ Chart generation complete!")


if __name__ == "__main__":
    main()
