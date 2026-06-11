#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 MLX - COCO val2017 Pose Evaluation Script

Evaluates YOLO26-pose MLX models on the COCO Keypoints val2017 set. Reports
keypoint (OKS) mAP using the official COCO keypoint evaluation when
``pycocotools`` and the ``person_keypoints_val2017.json`` annotations are
available, and falls back to the in-repo ``PoseMetrics`` evaluator otherwise.

Usage:
    python scripts/evaluate_coco_pose_val.py --model yolo26n-pose
    python scripts/evaluate_coco_pose_val.py --model all
    python scripts/evaluate_coco_pose_val.py --model yolo26n-pose --subset 100
"""

import argparse
import io
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
from _runtime_dirs import ensure_runtime_dirs

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
MODELS_DIR = _PROJECT_DIR / "models"
RESULTS_DIR = _PROJECT_DIR / "results"

sys.path.insert(0, str(_PROJECT_DIR / "src"))

from yolo26mlx import YOLO  # noqa: E402
from yolo26mlx.data.coco_dataset import COCODataset  # noqa: E402
from yolo26mlx.utils.metrics import OKS_SIGMA, PoseMetrics  # noqa: E402

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)

ALL_MODELS = [
    "yolo26n-pose",
    "yolo26s-pose",
    "yolo26m-pose",
    "yolo26l-pose",
    "yolo26x-pose",
]

# COCO keypoint person category id (single-class pose).
PERSON_CATEGORY_ID = 1

# Official YOLO26-pose keypoint mAP on COCO Keypoints val2017 (640px,
# end-to-end). Source: docs/plans/PLAN_POSE.md "Official YOLO26-Pose Benchmark
# Reference" (Ultralytics YOLO26 docs). Values are percentages so the MLX
# numbers (fractions) are multiplied by 100 when tabulated alongside them.
OFFICIAL_POSE_MAP50_95 = {
    "yolo26n-pose": 57.2,
    "yolo26s-pose": 63.0,
    "yolo26m-pose": 68.8,
    "yolo26l-pose": 70.4,
    "yolo26x-pose": 71.6,
}
OFFICIAL_POSE_MAP50 = {
    "yolo26n-pose": 83.3,
    "yolo26s-pose": 86.6,
    "yolo26m-pose": 89.6,
    "yolo26l-pose": 90.5,
    "yolo26x-pose": 91.6,
}


def find_coco_pose_dataset(
    data_arg: str | None,
) -> tuple[Path, str, Path | None] | None:
    """Locate a COCO pose dataset root and its validation split.

    Searches the explicit ``--data`` path first, then the project and home
    ``datasets`` directories for the full COCO pose set as well as the small
    ``coco-pose`` / ``coco8-pose`` datasets used for quick fallback runs.

    Args:
        data_arg: Explicit dataset path from the CLI, or None to search
            default locations.

    Returns:
        ``(root, split, kpt_ann)`` where ``root`` is the dataset root,
        ``split`` is the validation split name (e.g. ``"val2017"`` or
        ``"val"``), and ``kpt_ann`` is the path to
        ``person_keypoints_val2017.json`` when present (enabling the official
        COCOeval backend) or None. Returns None if no dataset is found.
    """
    candidates: list[Path] = []
    if data_arg:
        candidates.append(Path(data_arg).expanduser().resolve())
    candidates.extend(
        [
            (_PROJECT_DIR / "datasets" / "coco").resolve(),
            (Path.home() / "datasets" / "coco").resolve(),
            (_PROJECT_DIR / "datasets" / "coco-pose").resolve(),
            (Path.home() / "datasets" / "coco-pose").resolve(),
            (_PROJECT_DIR / "datasets" / "coco8-pose").resolve(),
            (Path.home() / "datasets" / "coco8-pose").resolve(),
        ]
    )

    seen: set[Path] = set()
    ordered: list[Path] = []
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        ordered.append(root)

    # Prefer roots with official keypoint annotations + val2017 images
    # (enables the primary COCOeval backend).
    for root in ordered:
        ann = root / "annotations" / "person_keypoints_val2017.json"
        if ann.is_file() and (root / "images" / "val2017").is_dir():
            return root, "val2017", ann

    # Fallback: any pose dataset with YOLO-format labels under images/<split>.
    for root in ordered:
        for split in ("val2017", "val"):
            if (root / "images" / split).is_dir():
                return root, split, None

    return None


def _letterbox_to_orig_xy(
    x: np.ndarray,
    y: np.ndarray,
    ratio: float,
    pad: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Map letterboxed-pixel coordinates back to original image pixels.

    Undoes the letterbox transform (scale by ``ratio`` then pad) applied by
    the dataloader, mapping keypoint predictions back to original-image space.

    Args:
        x: Letterboxed-pixel x coordinates.
        y: Letterboxed-pixel y coordinates.
        ratio: Letterbox scale ratio used during preprocessing.
        pad: Letterbox padding ``(pad_x, pad_y)`` in pixels.

    Returns:
        Tuple ``(x_orig, y_orig)`` of original-image-pixel coordinates.
    """
    x_orig = (x - pad[0]) / ratio
    y_orig = (y - pad[1]) / ratio
    return x_orig, y_orig


def _append_coco_kpt_entries(
    results: list[dict],
    image_id: int,
    xyxy_lb: np.ndarray,
    scores: np.ndarray,
    kpts_lb: np.ndarray,
    ratio: float,
    pad: tuple[float, float],
    orig_size: tuple[int, int],
) -> None:
    """Append COCO keypoint detection entries in original-image pixel coords.

    Builds entries in the official COCO keypoint result format
    (``{"image_id", "category_id", "keypoints", "score"}``) with the 17
    keypoints flattened as ``[x1, y1, v1, ...]`` in original-image pixels.

    Args:
        results: List of result dicts to append to (modified in place).
        image_id: COCO image id for this image.
        xyxy_lb: Predicted boxes (N, 4) xyxy in letterboxed pixels (used to
            drop degenerate detections).
        scores: Predicted confidence scores (N,).
        kpts_lb: Predicted keypoints (N, K, 3) as (x, y, v) in letterboxed
            pixels.
        ratio: Letterbox scale ratio.
        pad: Letterbox padding ``(pad_x, pad_y)``.
        orig_size: Original image size as ``(height, width)``.
    """
    orig_h, orig_w = orig_size
    for i in range(len(scores)):
        x1, y1, x2, y2 = xyxy_lb[i]
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        kx = kpts_lb[i, :, 0]
        ky = kpts_lb[i, :, 1]
        kv = kpts_lb[i, :, 2]
        kx_orig, ky_orig = _letterbox_to_orig_xy(kx, ky, ratio, pad)
        kx_orig = np.clip(kx_orig, 0, orig_w)
        ky_orig = np.clip(ky_orig, 0, orig_h)

        flat = np.stack([kx_orig, ky_orig, kv], axis=1).reshape(-1)
        results.append(
            {
                "image_id": int(image_id),
                "category_id": PERSON_CATEGORY_ID,
                "keypoints": [float(v) for v in flat],
                "score": float(scores[i]),
            }
        )


def evaluate_with_pycocotools(ann_file: str, pred_file: str, person_only: bool = False) -> dict:
    """Run official COCO keypoint (OKS) evaluation.

    Args:
        ann_file: Path to ``person_keypoints_val2017.json`` ground truth.
        pred_file: Path to the JSON predictions file in COCO keypoint result
            format.
        person_only: Restrict scoring to images that contain at least one person
            keypoint annotation. This matches the Ultralytics ``coco-pose`` val
            split (``val2017.txt``), which excludes person-free images. Scoring
            the full 5000-image set instead counts false-positive person
            detections on person-free images and lowers AP.

    Returns:
        Dict with ``mAP50-95_pose`` (AP) and ``mAP50_pose`` (AP50) as floats
        in ``[0, 1]``.
    """
    if not PYCOCOTOOLS_AVAILABLE:
        raise ImportError("pycocotools not installed. Run: pip install pycocotools")

    with open(pred_file) as f:
        preds = json.load(f)
    pred_img_ids = {int(p["image_id"]) for p in preds}

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    coco_gt = COCO(ann_file)
    sys.stdout = old_stdout

    if person_only:
        # Images carrying at least one person annotation (the Ultralytics pose
        # val set); intersect with predicted images for a valid COCOeval run.
        person_ids = {int(a["image_id"]) for a in coco_gt.anns.values()}
        eval_img_ids = sorted(pred_img_ids & person_ids)
    else:
        eval_img_ids = sorted(pred_img_ids)

    coco_dt = coco_gt.loadRes(pred_file)

    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    coco_eval.params.imgIds = eval_img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP50-95_pose": float(coco_eval.stats[0]),
        "mAP50_pose": float(coco_eval.stats[1]),
        "mAP75_pose": float(coco_eval.stats[2]),
    }


def evaluate_model(
    model_name: str,
    dataset_root: Path,
    split: str,
    kpt_ann: Path | None,
    img_size: int,
    subset: int | None,
    verbose: bool,
    output_dir: Path,
    conf: float = 0.001,
    batch_size: int = 16,
    rect: bool = False,
    person_only: bool = False,
    weights_override: str | None = None,
) -> dict | None:
    """Run val2017 pose eval for one model.

    Args:
        model_name: Model variant name (e.g. ``"yolo26n-pose"``).
        dataset_root: COCO pose dataset root.
        split: Validation split name.
        kpt_ann: Path to ``person_keypoints_val2017.json`` (COCOeval backend)
            or None (PoseMetrics fallback).
        img_size: Letterbox input side length.
        subset: Use only the first N images, or None for all.
        verbose: Verbose logging.
        output_dir: Directory for JSON and prediction files.
        conf: Confidence threshold for inference.
        batch_size: Inference batch size.
        rect: Use rectangular (stride-aligned) letterboxing to match the
            Ultralytics ``rect=True`` validation protocol. Forces batch size 1
            (per-image canvas shape) and disables the box-mAP cross-check
            (PoseMetrics normalizes by a square ``img_size``, invalid for rect);
            the keypoint mAP comes from pycocotools in original-image pixels and
            is unaffected.
        person_only: Score keypoint mAP only on person-containing images (the
            Ultralytics coco-pose val split), making the number directly
            comparable to the published official metric.
        weights_override: Explicit weights path to load instead of the default
            ``models/<model_name>`` lookup (used to evaluate fine-tuned
            checkpoints).

    Returns:
        Results dict with keypoint and box mAP, or None if weights are missing.
    """
    logger.info("\n" + "=" * 70)
    logger.info("Evaluating: %s", model_name)
    logger.info("=" * 70)

    if weights_override is not None:
        weights_path = Path(weights_override)
        if not weights_path.exists():
            logger.warning("Weights override not found: %s", weights_path)
            return None
    else:
        weights_candidates = [
            MODELS_DIR / f"{model_name}.safetensors",
            MODELS_DIR / f"{model_name}.npz",
            MODELS_DIR / f"{model_name}.pt",
        ]
        weights_path = next((p for p in weights_candidates if p.exists()), None)
        if weights_path is None:
            logger.warning("No weights found for %s (tried safetensors, npz, pt).", model_name)
            logger.warning("Searched under %s", MODELS_DIR)
            return None

    logger.info("Loading model from %s ...", weights_path)
    try:
        model = YOLO(str(weights_path), task="pose", verbose=verbose)
    except Exception as e:
        logger.error("Failed to load model %s: %s", model_name, e)
        return None

    inner = model.model
    if inner is None:
        logger.error("Model has no inner module loaded.")
        return None
    if hasattr(inner, "eval"):
        inner.eval()
    # Rectangular eval feeds a different canvas shape per image; mx.compile
    # retraces per shape, so compiling hurts more than it helps. Square eval
    # uses one fixed shape and benefits from compilation.
    if not rect and hasattr(inner, "compile_for_inference"):
        inner.compile_for_inference()

    # Read keypoint shape from the head so the decode matches the model output.
    head = inner.model[-1] if hasattr(inner, "model") else None
    kpt_shape = tuple(getattr(head, "kpt_shape", (17, 3)))
    nkpt = int(kpt_shape[0])
    sigmas = OKS_SIGMA if tuple(kpt_shape) == (17, 3) else np.ones(nkpt) / nkpt

    stride = int(max(np.array(inner.stride).tolist())) if hasattr(inner, "stride") else 32
    dataset = COCODataset(
        str(dataset_root),
        split=split,
        img_size=img_size,
        task="pose",
        kpt_shape=kpt_shape,
        rect=rect,
        stride=stride,
    )
    if rect:
        batch_size = 1
    num_images = len(dataset)
    if subset is not None:
        num_images = min(int(subset), num_images)

    # PoseMetrics (fallback / cross-check) operates in letterbox-normalized
    # space, exactly like Trainer._validate_pose. Single person class.
    pose_metrics = PoseMetrics(num_classes=1, sigmas=sigmas)
    coco_predictions: list[dict] = []

    preprocess_time = 0.0
    inference_time = 0.0
    postprocess_time = 0.0
    processed = 0

    logger.info("%s images to process", num_images)

    for _batch_idx, (images, annotations) in enumerate(dataset.get_dataloader(batch_size)):
        if processed >= num_images:
            break
        actual_batch = min(images.shape[0], num_images - processed)

        t0 = time.perf_counter()
        t_pre = time.perf_counter()

        batch_mx = images[:actual_batch]
        outputs = inner(batch_mx)
        if isinstance(outputs, tuple):
            mx.eval(*outputs)
            det_np = np.array(outputs[0])
        else:
            mx.eval(outputs)
            det_np = np.array(outputs)

        t_inf = time.perf_counter()

        for i in range(actual_batch):
            ann = annotations[i]
            orig_h, orig_w = ann["orig_size"]
            ratio = float(ann["ratio"])
            pad = (float(ann["pad"][0]), float(ann["pad"][1]))

            pred_i = det_np[i]  # (max_det, 6 + K*3)
            scores_all = pred_i[:, 4]
            keep = scores_all > conf
            pred_i = pred_i[keep]

            if pred_i.shape[0] > 0 and pred_i.shape[1] >= 6 + nkpt * 3:
                bx = pred_i[:, :4]
                p_scores = pred_i[:, 4]
                p_labels = pred_i[:, 5].astype(np.int64)
                cx, cy, w, h = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
                xyxy_lb = np.stack(
                    [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1
                ).astype(np.float32)
                kpts_lb = pred_i[:, 6 : 6 + nkpt * 3].reshape(-1, nkpt, 3).astype(np.float32)
            else:
                xyxy_lb = np.zeros((0, 4), dtype=np.float32)
                p_scores = np.zeros(0, dtype=np.float32)
                p_labels = np.zeros(0, dtype=np.int64)
                kpts_lb = np.zeros((0, nkpt, 3), dtype=np.float32)

            # COCO keypoint detections in original-image pixel coords (primary).
            _append_coco_kpt_entries(
                coco_predictions,
                ann["image_id"],
                xyxy_lb,
                p_scores,
                kpts_lb,
                ratio,
                pad,
                (orig_h, orig_w),
            )

            # PoseMetrics update in letterbox-normalized space (fallback).
            p_boxes_norm = xyxy_lb / img_size
            p_kpts_norm = kpts_lb.copy()
            if p_kpts_norm.shape[0] > 0:
                p_kpts_norm[..., 0] /= img_size
                p_kpts_norm[..., 1] /= img_size

            gt_boxes = np.asarray(ann.get("boxes", np.zeros((0, 4))), dtype=np.float32)
            gt_labels = np.asarray(ann.get("labels", np.zeros(0, dtype=np.int64)), dtype=np.int64)
            gt_kpts = np.asarray(ann.get("keypoints", np.zeros((0, nkpt, 3))), dtype=np.float32)

            pose_metrics.update(
                p_boxes_norm,
                p_scores,
                p_labels,
                p_kpts_norm if p_kpts_norm.shape[0] > 0 else None,
                gt_boxes,
                gt_labels,
                gt_kpts if gt_kpts.shape[0] > 0 else None,
            )

        t_post = time.perf_counter()
        preprocess_time += t_pre - t0
        inference_time += t_inf - t_pre
        postprocess_time += t_post - t_inf
        processed += actual_batch

        if verbose or processed % 100 == 0:
            logger.info("  Processed %s/%s images", processed, num_images)

    logger.info("Processed %s/%s images", processed, num_images)

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{model_name}_coco_pose_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(coco_predictions, f, indent=2)
        f.write("\n")
    logger.info("Saved %s COCO-format predictions to %s", len(coco_predictions), pred_path)

    custom = pose_metrics.compute()

    use_pycoco = PYCOCOTOOLS_AVAILABLE and len(coco_predictions) > 0 and kpt_ann is not None
    pycoco: dict | None = None
    if use_pycoco:
        try:
            logger.info("Running official pycocotools keypoint evaluation...")
            pycoco = evaluate_with_pycocotools(
                str(kpt_ann), str(pred_path), person_only=person_only
            )
            logger.info("%s", "=" * 80)
            logger.info("Official COCO keypoint metrics (pycocotools)")
            logger.info("%s", "=" * 80)
            logger.info("  mAP@0.5:0.95 (pose) = %.1f%%", pycoco["mAP50-95_pose"] * 100)
            logger.info("  mAP@0.5 (pose)      = %.1f%%", pycoco["mAP50_pose"] * 100)
        except Exception as e:
            logger.warning("pycocotools evaluation failed, using PoseMetrics only: %s", e)
            pycoco = None

    # Prefer pycocotools (official OKS at original-image resolution) for the
    # primary keypoint metrics so MLX vs PyTorch is apples-to-apples. Box mAP
    # is reported from the in-repo PoseMetrics cross-check. The full custom
    # metric dict is preserved in metrics_custom for sanity comparison.
    if pycoco is not None:
        m50_pose = pycoco["mAP50_pose"]
        m5095_pose = pycoco["mAP50-95_pose"]
    else:
        m50_pose = custom["mAP50_pose"]
        m5095_pose = custom["mAP50-95_pose"]
    # PoseMetrics normalizes boxes by a square img_size, which is invalid under
    # rectangular letterboxing; report keypoint mAP (pycocotools) only there.
    if rect:
        m50_box = float("nan")
        m5095_box = float("nan")
    else:
        m50_box = custom["mAP50_box"]
        m5095_box = custom["mAP50-95_box"]

    avg_pre = preprocess_time / max(processed, 1) * 1000
    avg_inf = inference_time / max(processed, 1) * 1000
    avg_post = postprocess_time / max(processed, 1) * 1000
    avg_total = avg_pre + avg_inf + avg_post

    logger.info(
        "Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image",
        avg_pre,
        avg_inf,
        avg_post,
    )
    logger.info("Total: %.1fms per image (%.1f FPS)", avg_total, 1000.0 / max(avg_total, 1e-9))

    results = {
        "model": model_name,
        "framework": "mlx",
        "dataset": "coco_val2017",
        "num_images": processed,
        "imgsz": img_size,
        "letterbox": "rect" if rect else "square",
        "eval_set": "person_only" if person_only else "all_val2017",
        "conf_thresh": conf,
        "pycocotools": pycoco is not None,
        "metrics": {
            "mAP50_pose": float(m50_pose),
            "mAP50-95_pose": float(m5095_pose),
            "mAP50_box": float(m50_box),
            "mAP50-95_box": float(m5095_box),
        },
        "metrics_custom": {k: float(v) for k, v in custom.items()},
        "speed": {
            "preprocess_ms": avg_pre,
            "inference_ms": avg_inf,
            "postprocess_ms": avg_post,
            "total_ms": avg_total,
            "fps": 1000.0 / max(avg_total, 1e-9),
        },
        "timestamp": datetime.now().isoformat(),
    }

    per_model_json = output_dir / f"{model_name}_coco_pose_val2017_results.json"
    with open(per_model_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", per_model_json)

    return results


def parse_args():
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO26-pose MLX on COCO val2017 (keypoint mAP).",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["yolo26n-pose"],
        help="Model variant(s). Use 'all' for all five sizes.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="COCO pose root (default: search datasets/coco, coco-pose, coco8-pose).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for inference",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use only the first N images (quick test)",
    )
    parser.add_argument(
        "--rect",
        action="store_true",
        help="Rectangular (stride-aligned) letterbox matching Ultralytics rect=True val "
        "(less padding; forces batch size 1).",
    )
    parser.add_argument(
        "--person-only",
        action="store_true",
        help="Score keypoint mAP only on person-containing images (the Ultralytics "
        "coco-pose val split), making the number comparable to the official metric.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Explicit weights path to evaluate instead of models/<model_name> "
        "(e.g. a fine-tuned checkpoint). Use with a single --model.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR),
        help="Output directory for JSON and prediction files",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point: evaluate the requested YOLO26-pose models on COCO val2017.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    ensure_runtime_dirs(_PROJECT_DIR)

    models = args.model
    if "all" in [m.lower() for m in models]:
        models = list(ALL_MODELS)
    else:
        for m in models:
            if m not in ALL_MODELS:
                logger.error("Unknown model: %s", m)
                logger.error("Available: %s", ", ".join(ALL_MODELS))
                return 1

    found = find_coco_pose_dataset(args.data)
    if found is None:
        logger.error("COCO pose val set not found.")
        logger.error("Expected annotations/person_keypoints_val2017.json + images/val2017,")
        logger.error("or a YOLO-format pose dataset (coco-pose / coco8-pose).")
        logger.error(
            "Tried --data (if set), %s, ~/datasets/coco, datasets/coco-pose, datasets/coco8-pose",
            _PROJECT_DIR / "datasets" / "coco",
        )
        return 1

    data_root, split, kpt_ann = found
    backend = "pycocotools (OKS)" if (PYCOCOTOOLS_AVAILABLE and kpt_ann) else "PoseMetrics"

    logger.info("=" * 70)
    logger.info("YOLO26 MLX - COCO val2017 Pose Evaluation")
    logger.info("=" * 70)
    logger.info("Models:       %s", ", ".join(models))
    logger.info("Dataset:      %s", data_root)
    logger.info("Split:        %s", split)
    logger.info("Eval backend: %s", backend)
    logger.info("Image size:   %s", args.imgsz)
    logger.info("Letterbox:    %s", "rectangular (rect=True)" if args.rect else "square")
    logger.info(
        "Eval set:     %s",
        "person-only (official-comparable)" if args.person_only else "all val2017 (5000)",
    )
    logger.info("Batch size:   %s", 1 if args.rect else args.batch)
    logger.info("Conf thresh:  %s", args.conf)
    if args.subset:
        logger.info("Subset:       %s images", args.subset)
    logger.info("=" * 70)

    output_dir = Path(args.output)
    all_results: dict[str, dict] = {}

    for name in models:
        try:
            r = evaluate_model(
                name,
                data_root,
                split,
                kpt_ann,
                args.imgsz,
                args.subset,
                args.verbose,
                output_dir,
                conf=args.conf,
                batch_size=args.batch,
                rect=args.rect,
                person_only=args.person_only,
                weights_override=args.weights,
            )
            if r is not None:
                all_results[name] = r
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", name, e)
            import traceback

            traceback.print_exc()

    if all_results:
        logger.info("\n" + "=" * 92)
        logger.info("SUMMARY - Keypoint mAP@0.5:0.95 vs official (%%)")
        logger.info("=" * 92)
        logger.info(
            "%-14s %14s %12s %14s %8s %10s",
            "Model",
            "mAP50-95 pose",
            "(Official)",
            "mAP50 pose",
            "FPS",
            "ms/img",
        )
        logger.info("-" * 92)
        for m in ALL_MODELS:
            if m not in all_results:
                continue
            r = all_results[m]
            p5095 = r["metrics"]["mAP50-95_pose"] * 100
            p50 = r["metrics"]["mAP50_pose"] * 100
            off_p = OFFICIAL_POSE_MAP50_95.get(m, 0.0)
            fps = r["speed"]["fps"]
            ms = r["speed"]["total_ms"]
            logger.info(
                "%-14s %13.1f%% %10.1f%% %13.1f%% %7.1f %9.1fms",
                m,
                p5095,
                off_p,
                p50,
                fps,
                ms,
            )
        logger.info("=" * 92)

        combined = output_dir / "yolo26_pose_coco_val_results.json"
        with open(combined, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("\nCombined results saved to %s", combined)

    if args.subset:
        logger.info(
            "\nNote: subset mode (%s images). Full val2017 has 5000 images.",
            args.subset,
        )

    return 0 if all_results else 1


if __name__ == "__main__":
    sys.exit(main())
