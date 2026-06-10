#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 Pose Inference Accuracy vs Official Chart
================================================
Generates a chart comparing the MLX YOLO26-pose keypoint mAP on COCO
Keypoints val2017 against the official Ultralytics reference numbers.

This script reads the per-model COCO val results written by
``evaluate_coco_pose_val.py``:
    ../results/yolo26{n,s,m,l,x}-pose_coco_pose_val2017_results.json

And writes:
    ../results/charts/yolo26_pose_accuracy_vs_official.png

Usage:
    python benchmark_yolo26_pose_accuracy_vs_official.py
    python benchmark_yolo26_pose_accuracy_vs_official.py --format pdf
    python benchmark_yolo26_pose_accuracy_vs_official.py --output custom_charts/
"""

import argparse
import json
import logging
from pathlib import Path

from _runtime_dirs import ensure_runtime_dirs

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR / ".."
RESULTS_DIR = PROJECT_DIR / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

MODEL_SIZES = ["n", "s", "m", "l", "x"]
MODEL_LABELS = ["YOLO26n-pose", "YOLO26s-pose", "YOLO26m-pose", "YOLO26l-pose", "YOLO26x-pose"]

# Official YOLO26-pose keypoint mAP on COCO Keypoints val2017 (640px, end-to-end).
# Source: docs/plans/PLAN_POSE.md "Official YOLO26-Pose Benchmark Reference"
# (Ultralytics YOLO26 docs); kept in sync with evaluate_coco_pose_val.py.
OFFICIAL_MAP50_95 = {"n": 57.2, "s": 63.0, "m": 68.8, "l": 70.4, "x": 71.6}
OFFICIAL_MAP50 = {"n": 83.3, "s": 86.6, "m": 89.6, "l": 90.5, "x": 91.6}

COLOR_MLX = "#648FFF"
COLOR_OFFICIAL = "#FE6100"


def _model_key(size: str) -> str:
    return f"yolo26{size}-pose"


def load_mlx_scores(results_dir: Path) -> dict:
    """Load MLX COCO val keypoint mAP (percent) from per-model result JSONs.

    Args:
        results_dir: Directory holding ``*_coco_pose_val2017_results.json``.

    Returns:
        Mapping ``size -> {"map": pct, "map50": pct}`` for models found.
    """
    scores = {}
    for size in MODEL_SIZES:
        path = results_dir / f"{_model_key(size)}_coco_pose_val2017_results.json"
        if not path.exists():
            logger.warning("  Missing results for %s: %s", _model_key(size), path.name)
            continue
        with open(path) as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        scores[size] = {
            "map": float(metrics.get("mAP50-95_pose", 0.0)) * 100.0,
            "map50": float(metrics.get("mAP50_pose", 0.0)) * 100.0,
        }
    return scores


def create_accuracy_vs_official_chart(
    scores: dict,
    output_path: Path,
    figsize: tuple = (13, 6),
    dpi: int = 150,
) -> bool:
    """Plot MLX keypoint mAP next to the official reference per model.

    Args:
        scores: Mapping from :func:`load_mlx_scores`.
        output_path: Destination image path.
        figsize: Figure size (width, height).
        dpi: Output resolution.

    Returns:
        True when a chart was written.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.error("matplotlib is required for chart generation.")
        return False

    sizes = [s for s in MODEL_SIZES if s in scores]
    if not sizes:
        logger.error("No MLX COCO val results found; run evaluate_coco_pose_val.py first.")
        return False

    labels = [MODEL_LABELS[MODEL_SIZES.index(s)] for s in sizes]
    x = np.arange(len(sizes))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    def _panel(ax, mlx_vals, off_vals, title, ylabel):
        bars_mlx = ax.bar(
            x - width / 2,
            mlx_vals,
            width,
            label="MLX (this port)",
            color=COLOR_MLX,
            edgecolor="white",
            linewidth=0.5,
        )
        bars_off = ax.bar(
            x + width / 2,
            off_vals,
            width,
            label="Official (Ultralytics)",
            color=COLOR_OFFICIAL,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar in list(bars_mlx) + list(bars_off):
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        for xi, m, o in zip(x, mlx_vals, off_vals, strict=True):
            ax.annotate(
                f"{m - o:+.1f}",
                xy=(xi, max(m, o)),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#444444",
                fontweight="bold",
            )
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 100)

    _panel(
        ax1,
        [scores[s]["map"] for s in sizes],
        [OFFICIAL_MAP50_95[s] for s in sizes],
        "Keypoint mAP@0.50:0.95",
        "mAP@0.50:0.95 (pose) %",
    )
    _panel(
        ax2,
        [scores[s]["map50"] for s in sizes],
        [OFFICIAL_MAP50[s] for s in sizes],
        "Keypoint mAP@0.50",
        "mAP@0.50 (pose) %",
    )

    fig.suptitle(
        "YOLO26 Pose Inference Accuracy: MLX vs Official (COCO val2017)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def main():
    """Render the MLX-vs-official keypoint mAP comparison chart."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="YOLO26 Pose inference accuracy vs official chart."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_DIR,
        help="Directory with *_coco_pose_val2017_results.json files.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI (default: 150)")
    parser.add_argument(
        "--output",
        type=Path,
        default=CHARTS_DIR,
        help="Output directory for the chart.",
    )
    args = parser.parse_args()
    ensure_runtime_dirs(PROJECT_DIR)

    logger.info("=" * 70)
    logger.info("  YOLO26 Pose Inference Accuracy vs Official")
    logger.info("=" * 70)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        logger.error("matplotlib is required. Install with: pip install matplotlib")
        return

    scores = load_mlx_scores(args.results)
    if not scores:
        logger.error("No results found in %s", args.results)
        return

    logger.info("\nLoaded MLX scores (keypoint mAP %%):")
    logger.info(f"{'Model':<14}{'MLX 50-95':>11}{'Official':>10}{'Δ':>8}")
    for size in MODEL_SIZES:
        if size in scores:
            mlx = scores[size]["map"]
            off = OFFICIAL_MAP50_95[size]
            logger.info(f"{_model_key(size):<14}{mlx:>10.1f}%{off:>9.1f}%{mlx - off:>+8.1f}")

    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"yolo26_pose_accuracy_vs_official.{args.format}"
    if create_accuracy_vs_official_chart(scores, out, dpi=args.dpi):
        logger.info(f"\nChart saved to: {out}")
    else:
        logger.error("\nChart generation failed.")


if __name__ == "__main__":
    main()
