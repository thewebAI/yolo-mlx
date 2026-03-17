#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 MLX - COCO val2017 Evaluation Script

Evaluates YOLO26 MLX models on COCO val2017 using the official COCO evaluation protocol
with pycocotools. Reports mAP@0.5 and mAP@0.5:0.95 matching Ultralytics format.

Usage:
    # Full validation on all 5000 images (recommended for accurate benchmarks)
    python scripts/evaluate_coco_val.py --model yolo26n
    python scripts/evaluate_coco_val.py --model all      # All models: n, s, m, l, x

    # Quick testing with subset
    python scripts/evaluate_coco_val.py --model yolo26n --subset 100

    # Custom dataset path
    python scripts/evaluate_coco_val.py --model yolo26n --data /path/to/coco

    # Verbose output
    python scripts/evaluate_coco_val.py --model yolo26n --verbose

    # Custom output directory
    python scripts/evaluate_coco_val.py --model yolo26n --output /path/to/output

Results are saved to results/ directory by default (overridable via --output).
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
import yaml
from _runtime_dirs import ensure_runtime_dirs

# Resolve project root relative to this script (scripts/../)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yolo26mlx import YOLO  # noqa: E402
from yolo26mlx.data.coco_dataset import COCODataset, COCOResultsWriter  # noqa: E402
from yolo26mlx.nn.tasks import build_model  # noqa: E402
from yolo26mlx.utils.coco_metrics import COCOMetrics  # noqa: E402

# Try to import pycocotools
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)


def evaluate_with_pycocotools(ann_file: str, pred_file: str) -> dict:
    """Run official COCO evaluation using pycocotools.

    Args:
        ann_file: Path to COCO annotations JSON file
        pred_file: Path to predictions JSON file in COCO format

    Returns:
        dict with mAP metrics
    """
    if not PYCOCOTOOLS_AVAILABLE:
        raise ImportError("pycocotools not installed. Run: pip install pycocotools")

    # Get actual image IDs from predictions (for subset evaluation)
    with open(pred_file) as f:
        preds = json.load(f)
    pred_img_ids = list(set(p["image_id"] for p in preds))

    # Load ground truth (suppress verbose output)
    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    coco_gt = COCO(ann_file)
    sys.stdout = old_stdout

    # Load predictions
    coco_dt = coco_gt.loadRes(pred_file)

    # Run evaluation LIMITED to images with predictions
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = pred_img_ids  # Only evaluate on images with predictions
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    # stats[0] = AP @ IoU=0.50:0.95 (primary COCO metric)
    # stats[1] = AP @ IoU=0.50
    # stats[2] = AP @ IoU=0.75
    # stats[3] = AP for small objects
    # stats[4] = AP for medium objects
    # stats[5] = AP for large objects
    return {
        "mAP50-95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "mAP75": float(coco_eval.stats[2]),
        "mAP_small": float(coco_eval.stats[3]),
        "mAP_medium": float(coco_eval.stats[4]),
        "mAP_large": float(coco_eval.stats[5]),
        "precision": float(coco_eval.stats[0]),  # Use mAP as proxy
        "recall": float(coco_eval.stats[8]) if len(coco_eval.stats) > 8 else 0.0,
    }


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace with model, data, imgsz, batch, conf, iou, and other options.
    """
    parser = argparse.ArgumentParser(description="Evaluate YOLO26 MLX on COCO val2017")
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["yolo26n"],
        help="Model variant(s) to evaluate. Use 'all' for all models or specify one or more: yolo26n yolo26s yolo26m yolo26l yolo26x",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(_PROJECT_DIR / "datasets" / "coco"),
        help="Path to COCO dataset root",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument(
        "--subset", type=int, default=None, help="Use only first N images (for quick testing)"
    )
    parser.add_argument("--save-json", action="store_true", help="Save results in COCO JSON format")
    parser.add_argument(
        "--pycocotools",
        action="store_true",
        default=True,
        help="Use official pycocotools for metrics (default: True)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_DIR / "results"),
        help="Output directory for results",
    )
    return parser.parse_args()


def compute_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of bounding boxes.

    Args:
        boxes1: First set of bounding boxes (N, 4) in xyxy format.
        boxes2: Second set of bounding boxes (M, 4) in xyxy format.

    Returns:
        (N, M) pairwise IoU matrix.
    """
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-7)


def _apply_nms(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thresh: float, max_det: int
) -> np.ndarray:
    """Apply NMS per class and format output.

    Args:
        boxes: (N, 4) xyxy format
        scores: (N,) confidence scores
        classes: (N,) class indices
        iou_thresh: NMS IoU threshold
        max_det: Maximum detections

    Returns:
        (M, 6) detections [x1, y1, x2, y2, conf, cls]
    """
    keep_indices = []
    for cls in np.unique(classes):
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        # Simple NMS
        order = np.argsort(-cls_scores)
        cls_keep = []

        while len(order) > 0:
            i = order[0]
            cls_keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            remaining = order[1:]
            ious = compute_iou_np(cls_boxes[i : i + 1], cls_boxes[remaining])[0]

            # Keep boxes with IoU < threshold
            order = remaining[ious < iou_thresh]

        # Map back to original indices
        cls_indices = np.where(cls_mask)[0]
        keep_indices.extend(cls_indices[cls_keep].tolist())

    # Sort by score and limit to max_det
    if len(keep_indices) > 0:
        keep_scores = scores[keep_indices]
        sorted_indices = np.argsort(-keep_scores)
        keep_indices = [keep_indices[i] for i in sorted_indices[:max_det]]

        det = np.column_stack([boxes[keep_indices], scores[keep_indices], classes[keep_indices]])
    else:
        det = np.zeros((0, 6))

    return det


def decode_predictions(
    outputs, conf_thresh: float = 0.001, iou_thresh: float = 0.7, max_det: int = 300
) -> list:
    """Decode raw model outputs to detections.

    Args:
        outputs: Model outputs - can be:
            - mx.array of shape (batch, anchors, 4 + num_classes) [YOLO26 MLX format]
            - dict with 'one2one' key containing boxes/scores [end2end format]
        conf_thresh: Confidence threshold
        iou_thresh: NMS IoU threshold
        max_det: Maximum detections per image

    Returns:
        List of detections per image, each (N, 6) [x1, y1, x2, y2, conf, cls]
    """
    # Handle dict output format from end2end models
    if isinstance(outputs, dict):
        if "one2one" in outputs:
            data = outputs["one2one"]
        else:
            data = outputs

        # Extract boxes and scores
        boxes_raw = np.array(data["boxes"])  # (B, anchors, 4)
        scores_raw = np.array(data["scores"])  # (B, anchors, nc)

        # Apply sigmoid to scores if not already applied
        if scores_raw.max() > 1.0:
            scores_raw = 1 / (1 + np.exp(-scores_raw))

        batch_size = boxes_raw.shape[0]

        results = []
        for b in range(batch_size):
            boxes_xywh = boxes_raw[b]  # (anchors, 4)
            class_scores = scores_raw[b]  # (anchors, nc)

            # Convert xywh to xyxy (boxes are in xywh format)
            x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

            # Get max class score and class index
            class_conf = class_scores.max(axis=1)
            class_idx = class_scores.argmax(axis=1)

            # Filter by confidence
            mask = class_conf > conf_thresh
            boxes = boxes_xyxy[mask]
            scores = class_conf[mask]
            classes = class_idx[mask]

            if len(boxes) == 0:
                results.append(np.zeros((0, 6)))
                continue

            # Apply NMS and limit detections (reuse existing NMS code)
            detections = _apply_nms(boxes, scores, classes, iou_thresh, max_det)
            results.append(detections)

        return results

    # Handle tensor output format
    outputs_np = np.array(outputs)
    batch_size = outputs_np.shape[0]

    # Check if this is end2end format (B, max_det, 6) with [x,y,w,h,conf,class]
    if outputs_np.shape[2] == 6:
        # End-to-end output: already processed, just convert xywh to xyxy and filter
        results = []
        for b in range(batch_size):
            pred = outputs_np[b]  # (max_det, 6) = [x, y, w, h, conf, class]

            # Convert xywh to xyxy
            x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            conf = pred[:, 4]
            cls = pred[:, 5]

            # Filter by confidence
            mask = conf > conf_thresh
            if mask.sum() == 0:
                results.append(np.zeros((0, 6)))
                continue

            # Already sorted by confidence, no NMS needed for end2end
            detections = np.stack(
                [x1[mask], y1[mask], x2[mask], y2[mask], conf[mask], cls[mask]], axis=1
            )

            # Limit to max_det (should already be <= max_det)
            detections = detections[:max_det]
            results.append(detections)

        return results

    results = []

    for b in range(batch_size):
        # Extract boxes and class scores - format is (anchors, 4 + nc)
        pred = outputs_np[b]  # (anchors, 4 + nc)

        # First 4 values are box coordinates (x, y, w, h)
        boxes_xywh = pred[:, :4]  # (anchors, 4)

        # Remaining values are class scores (already sigmoided)
        class_scores = pred[:, 4:]  # (anchors, num_classes)

        # Convert xywh to xyxy
        x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)  # (num_anchors, 4)

        # Get max class score and class index
        class_conf = class_scores.max(axis=1)  # (num_anchors,)
        class_idx = class_scores.argmax(axis=1)  # (num_anchors,)

        # Filter by confidence
        mask = class_conf > conf_thresh
        boxes = boxes_xyxy[mask]
        scores = class_conf[mask]
        classes = class_idx[mask]

        if len(boxes) == 0:
            results.append(np.zeros((0, 6)))
            continue

        # Apply NMS and limit detections
        detections = _apply_nms(boxes, scores, classes, iou_thresh, max_det)
        results.append(detections)

    return results


ALL_MODELS = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]


def evaluate_model(model_name: str, args, dataset) -> dict:
    """Evaluate a single model and return results dict.

    Args:
        model_name: YOLO26 model variant name (e.g. 'yolo26n', 'yolo26s').
        args: Parsed command-line arguments with evaluation config.
        dataset: COCODataset instance for val2017 images and annotations.

    Returns:
        Dict with model name, metrics (mAP50, mAP50-95), speed stats, and metadata.
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"Evaluating: {model_name}")
    logger.info("=" * 70)

    # Load model
    logger.info(f"\nLoading model {model_name}...")

    # Try to load from converted weights first
    model_path = Path(__file__).parent.parent / "models" / f"{model_name}.safetensors"
    npz_path = Path(__file__).parent.parent / "models" / f"{model_name}.npz"

    # Get model scale from name (yolo26n -> n)
    scale = model_name.replace("yolo26", "")

    # Find YAML config
    import yolo26mlx

    pkg_dir = Path(yolo26mlx.__file__).parent
    yaml_path = pkg_dir / "cfg" / "models" / "26" / "yolo26.yaml"

    # Build model first
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg["scale"] = scale

    model = YOLO(str(yaml_path))
    model.model = build_model(cfg, verbose=False)
    model._setup_metadata()

    # Load weights if available
    def map_pytorch_to_mlx_name(pt_name: str) -> str:
        """Map PyTorch parameter name to MLX naming convention.

        Args:
            pt_name: PyTorch-style dot-separated parameter name (e.g. 'model.0.conv.weight').

        Returns:
            Equivalent MLX parameter name with layers/nested structure remapped.
        """
        import re

        name = pt_name

        # 1. Replace 'model.X.' with 'model.layers.X.'
        name = re.sub(r"^model\.(\d+)\.", r"model.layers.\1.", name)

        # 2. Handle layer 10 (C2PSA) - m.0. -> m.psa0.
        name = re.sub(r"layers\.10\.m\.(\d+)\.", r"layers.10.m.psa\1.", name)

        # 3. Handle layer 22 (C3k2 with attn) - m.0.X. -> m.0.layers.X.
        name = re.sub(r"layers\.22\.m\.0\.(\d+)\.", r"layers.22.m.0.layers.\1.", name)

        # 4. Handle layer 23 (Detect head) - cv2.X.Y. -> cv2.layerX.layers.Y.
        name = re.sub(r"layers\.23\.cv2\.(\d+)\.(\d+)\.", r"layers.23.cv2.layer\1.layers.\2.", name)

        # 5. Handle cv3 nested structure: cv3.scale.block.layer -> cv3.layerN.layers.M
        # PyTorch: cv3.0.0.0 (scale 0, block 0, layer 0) -> MLX: cv3.layer0.layers.0
        # PyTorch: cv3.0.0.1 (scale 0, block 0, layer 1) -> MLX: cv3.layer0.layers.1
        # PyTorch: cv3.0.1.0 (scale 0, block 1, layer 0) -> MLX: cv3.layer0.layers.2
        # PyTorch: cv3.0.1.1 (scale 0, block 1, layer 1) -> MLX: cv3.layer0.layers.3
        # PyTorch: cv3.0.2 (scale 0, final conv) -> MLX: cv3.layer0.layers.4
        def map_cv3_nested(match):
            """Convert cv3 block.layer indices to flattened MLX layer index."""
            scale = match.group(1)
            block = int(match.group(2))
            layer = int(match.group(3))
            # Each block has 2 layers, so flat_idx = block * 2 + layer
            flat_idx = block * 2 + layer
            return f"layers.23.cv3.layer{scale}.layers.{flat_idx}."

        def map_cv3_final(match):
            """Convert cv3 final conv index to flattened MLX layer index."""
            scale = match.group(1)
            block = int(match.group(2))
            # Final conv is at index 4 (after 2 blocks with 2 layers each)
            return f"layers.23.cv3.layer{scale}.layers.{block * 2}."

        # Apply cv3 mappings (order matters - more specific first)
        name = re.sub(r"layers\.23\.cv3\.(\d+)\.(\d+)\.(\d+)\.", map_cv3_nested, name)
        name = re.sub(r"layers\.23\.cv3\.(\d+)\.(\d+)\.", map_cv3_final, name)

        # 6. Handle one2one detection heads (same pattern as cv2/cv3)
        name = re.sub(
            r"layers\.23\.one2one_cv2\.(\d+)\.(\d+)\.",
            r"layers.23.one2one_cv2.layer\1.layers.\2.",
            name,
        )

        def map_one2one_cv3_nested(match):
            """Convert one2one_cv3 block.layer indices to flattened MLX layer index."""
            scale = match.group(1)
            block = int(match.group(2))
            layer = int(match.group(3))
            flat_idx = block * 2 + layer
            return f"layers.23.one2one_cv3.layer{scale}.layers.{flat_idx}."

        def map_one2one_cv3_final(match):
            """Convert one2one_cv3 final conv index to flattened MLX layer index."""
            scale = match.group(1)
            block = int(match.group(2))
            return f"layers.23.one2one_cv3.layer{scale}.layers.{block * 2}."

        name = re.sub(
            r"layers\.23\.one2one_cv3\.(\d+)\.(\d+)\.(\d+)\.", map_one2one_cv3_nested, name
        )
        name = re.sub(r"layers\.23\.one2one_cv3\.(\d+)\.(\d+)\.", map_one2one_cv3_final, name)

        return name

    if model_path.exists():
        logger.info(f"  Loading weights from safetensors: {model_path}")
        model.model.load_weights(str(model_path))
        logger.info("  Weights loaded successfully")
    elif npz_path.exists():
        logger.info(f"  Loading weights from npz: {npz_path}")
        # Load npz weights and map names
        weights = dict(mx.load(str(npz_path)))
        mapped_weights = [(map_pytorch_to_mlx_name(k), v) for k, v in weights.items()]

        # Get MLX model parameter names
        def get_param_names(params, prefix=""):
            """Recursively collect fully-qualified parameter names from nested dicts/lists.

            Args:
                params: Nested dict/list of model parameters (leaves are arrays).
                prefix: Dot-separated path prefix accumulated during recursion.

            Returns:
                Set of fully-qualified parameter name strings.
            """
            names = set()
            if isinstance(params, dict):
                for k, v in params.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    names.update(get_param_names(v, new_prefix))
            elif isinstance(params, list):
                for i, v in enumerate(params):
                    new_prefix = f"{prefix}.{i}" if prefix else str(i)
                    names.update(get_param_names(v, new_prefix))
            elif hasattr(params, "shape"):
                names.add(prefix)
            return names

        mlx_param_names = get_param_names(model.model.parameters())

        # Filter to only matching weights
        matching_weights = [(k, v) for k, v in mapped_weights if k in mlx_param_names]
        missing = [k for k, v in mapped_weights if k not in mlx_param_names]

        logger.info(f"  Matching weights: {len(matching_weights)}/{len(mapped_weights)}")
        if missing:
            logger.warning(f"  Warning: {len(missing)} weights not matched (architecture mismatch)")

        model.model.load_weights(matching_weights, strict=False)
        logger.info("  Weights loaded successfully")
    else:
        # Try loading from PyTorch .pt file and convert on-the-fly
        pt_path = Path(__file__).parent.parent / "models" / f"{model_name}.pt"
        if pt_path.exists():
            logger.info(f"  Loading weights from PyTorch file: {pt_path}")
            logger.info("  Converting PyTorch weights to MLX format...")
            try:
                import torch

                pt_weights = torch.load(str(pt_path), map_location="cpu", weights_only=False)

                # Extract state dict
                if hasattr(pt_weights, "state_dict"):
                    state_dict = pt_weights.state_dict()
                elif isinstance(pt_weights, dict) and "model" in pt_weights:
                    state_dict = (
                        pt_weights["model"].state_dict()
                        if hasattr(pt_weights["model"], "state_dict")
                        else pt_weights["model"]
                    )
                elif isinstance(pt_weights, dict):
                    state_dict = pt_weights
                else:
                    state_dict = (
                        pt_weights.model.state_dict()
                        if hasattr(pt_weights.model, "state_dict")
                        else {}
                    )

                # Convert PyTorch tensors to MLX arrays
                def convert_tensor(t):
                    """Convert a PyTorch tensor to an MLX array, transposing conv weights from OIHW to OHWI."""
                    if hasattr(t, "numpy"):
                        arr = t.float().numpy()
                        # Transpose conv weights from PyTorch OIHW to MLX OHWI
                        if len(arr.shape) == 4:
                            arr = arr.transpose(0, 2, 3, 1)  # OIHW -> OHWI
                        return mx.array(arr)
                    return t

                converted_weights = {k: convert_tensor(v) for k, v in state_dict.items()}

                # Map names and load
                mapped_weights = [
                    (map_pytorch_to_mlx_name(k), v) for k, v in converted_weights.items()
                ]

                # Get MLX model parameter names
                def get_param_names_pt(params, prefix=""):
                    """Recursively collect fully-qualified parameter names from nested dicts/lists.

                    Args:
                        params: Nested dict/list of model parameters (leaves are arrays).
                        prefix: Dot-separated path prefix accumulated during recursion.

                    Returns:
                        Set of fully-qualified parameter name strings.
                    """
                    names = set()
                    if isinstance(params, dict):
                        for k, v in params.items():
                            new_prefix = f"{prefix}.{k}" if prefix else k
                            names.update(get_param_names_pt(v, new_prefix))
                    elif isinstance(params, list):
                        for i, v in enumerate(params):
                            new_prefix = f"{prefix}.{i}" if prefix else str(i)
                            names.update(get_param_names_pt(v, new_prefix))
                    elif hasattr(params, "shape"):
                        names.add(prefix)
                    return names

                mlx_param_names = get_param_names_pt(model.model.parameters())
                matching_weights = [(k, v) for k, v in mapped_weights if k in mlx_param_names]

                logger.info(f"  Matching weights: {len(matching_weights)}/{len(mapped_weights)}")
                model.model.load_weights(matching_weights, strict=False)
                logger.info("  Weights loaded successfully from PyTorch file")

                # Save as .npz for faster loading next time
                save_path = Path(__file__).parent.parent / "models" / f"{model_name}.npz"
                mx.savez(str(save_path), **{k: v for k, v in matching_weights})
                logger.info(f"  Saved converted weights to {save_path}")

            except Exception as e:
                logger.error(f"  ERROR: Failed to load PyTorch weights: {e}")
                logger.warning("  WARNING: Using random weights! Results will not be meaningful.")
        else:
            logger.warning("  WARNING: No weights found! Results will not be meaningful.")
            logger.warning(f"  Searched: {model_path}, {npz_path}, {pt_path}")

    # Set model to eval mode - need to set on the inner model for inference output format
    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()
    elif hasattr(model, "eval"):
        model.eval()

    # Count parameters
    def count_params(params):
        """Recursively count total scalar parameters in nested dict of arrays.

        Args:
            params: Nested dict of model parameter arrays.

        Returns:
            Total number of scalar parameters.
        """
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif hasattr(params, "size"):
            total += params.size
        return total

    if hasattr(model, "model") and hasattr(model.model, "model"):
        param_count = sum(count_params(layer.parameters()) for layer in model.model.model.layers)
    elif hasattr(model, "model"):
        param_count = sum(count_params(layer.parameters()) for layer in model.model.layers)
    else:
        param_count = 0
    logger.info(f"  Loaded {param_count:,} parameters")

    # Get number of images to process
    num_images = len(dataset)
    if args.subset:
        num_images = min(args.subset, num_images)

    logger.info(f"  {num_images} images to process")

    # Initialize metrics
    metrics_calculator = COCOMetrics(num_classes=80, class_names=dataset.class_names)

    # Initialize results writer (needed for pycocotools or --save-json)
    results_writer = COCOResultsWriter() if (args.save_json or args.pycocotools) else None

    # Timing
    preprocess_time = 0
    inference_time = 0
    postprocess_time = 0

    # Process images
    logger.info("\nRunning evaluation...")
    processed = 0

    for _batch_idx, (images, annotations) in enumerate(dataset.get_dataloader(args.batch)):
        if processed >= num_images:
            break

        batch_size = images.shape[0]
        actual_batch = min(batch_size, num_images - processed)

        # Inference
        t_start = time.perf_counter()

        # MLX models use NHWC format - images from dataset are already NHWC
        # No format conversion needed

        t_preprocess = time.perf_counter()

        # Get the actual model for inference
        actual_model = model.model if hasattr(model, "model") else model
        outputs = actual_model(images[:actual_batch])
        mx.eval(outputs)

        t_inference = time.perf_counter()

        # Decode predictions
        detections = decode_predictions(outputs, conf_thresh=args.conf, iou_thresh=args.iou)

        t_postprocess = time.perf_counter()

        # Update timing
        preprocess_time += t_preprocess - t_start
        inference_time += t_inference - t_preprocess
        postprocess_time += t_postprocess - t_inference

        # Update metrics
        for i in range(actual_batch):
            det = detections[i]
            ann = annotations[i]

            # Format predictions
            pred = {
                "boxes": det[:, :4] / args.imgsz if len(det) > 0 else np.zeros((0, 4)),
                "scores": det[:, 4] if len(det) > 0 else np.zeros(0),
                "labels": (
                    det[:, 5].astype(np.int64) if len(det) > 0 else np.zeros(0, dtype=np.int64)
                ),
            }

            # Format ground truth
            gt = {"boxes": ann["boxes"], "labels": ann["labels"], "iscrowd": ann["iscrowd"]}

            metrics_calculator.update(pred, gt, ann["image_id"])

            # Save to results if needed
            if results_writer and len(det) > 0:
                results_writer.add_predictions(
                    image_id=ann["image_id"],
                    boxes=det[:, :4] / args.imgsz,
                    scores=det[:, 4],
                    labels=det[:, 5].astype(np.int64),
                    orig_size=ann["orig_size"],
                    ratio=ann["ratio"],
                    pad=ann["pad"],
                    img_size=args.imgsz,
                )

        processed += actual_batch

        # Progress
        if args.verbose or processed % 100 == 0:
            logger.info(f"  Processed {processed}/{num_images} images")

    logger.info(f"  Processed {processed}/{num_images} images")

    # Save predictions first (needed for pycocotools)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always save COCO format predictions for pycocotools
    coco_pred_file = output_dir / f"{model_name}_coco_predictions.json"
    if results_writer:
        results_writer.save(str(coco_pred_file))

    # Compute metrics
    logger.info("\nComputing metrics...")

    # Use pycocotools if available and requested
    use_pycocotools = args.pycocotools and PYCOCOTOOLS_AVAILABLE and (results_writer is not None)

    if use_pycocotools and len(results_writer.results) > 0:
        logger.info("  Using official pycocotools evaluation...")
        ann_file = Path(args.data) / "annotations" / "instances_val2017.json"
        metrics = evaluate_with_pycocotools(str(ann_file), str(coco_pred_file))

        # Print pycocotools-style results
        logger.info(f"\n{'=' * 80}")
        logger.info("Official COCO Metrics (pycocotools)")
        logger.info(f"{'=' * 80}")
        logger.info(f"  mAP@0.5:0.95 = {metrics['mAP50-95'] * 100:.1f}%")
        logger.info(f"  mAP@0.5      = {metrics['mAP50'] * 100:.1f}%")
        logger.info(f"  mAP@0.75     = {metrics['mAP75'] * 100:.1f}%")
        logger.info(f"  mAP (small)  = {metrics['mAP_small'] * 100:.1f}%")
        logger.info(f"  mAP (medium) = {metrics['mAP_medium'] * 100:.1f}%")
        logger.info(f"  mAP (large)  = {metrics['mAP_large'] * 100:.1f}%")
        logger.info(f"{'=' * 80}")
    else:
        if args.pycocotools and not PYCOCOTOOLS_AVAILABLE:
            logger.warning("  WARNING: pycocotools not installed, using custom metrics")
        metrics = metrics_calculator.compute()
        metrics_calculator.print_results(metrics)

    # Print timing
    avg_preprocess = preprocess_time / processed * 1000
    avg_inference = inference_time / processed * 1000
    avg_postprocess = postprocess_time / processed * 1000
    avg_total = avg_preprocess + avg_inference + avg_postprocess

    logger.info(
        f"\nSpeed: {avg_preprocess:.1f}ms preprocess, "
        f"{avg_inference:.1f}ms inference, "
        f"{avg_postprocess:.1f}ms postprocess per image"
    )
    logger.info(f"Total: {avg_total:.1f}ms per image ({1000 / avg_total:.1f} FPS)")

    # Build results dict
    results = {
        "model": model_name,
        "framework": "mlx",
        "dataset": "coco_val2017",
        "num_images": processed,
        "imgsz": args.imgsz,
        "conf_thresh": args.conf,
        "iou_thresh": args.iou,
        "pycocotools": use_pycocotools,
        "metrics": {
            "mAP50": (
                metrics["mAP50"]
                if isinstance(metrics["mAP50"], float) and metrics["mAP50"] <= 1
                else metrics["mAP50"] / 100
            ),
            "mAP50-95": (
                metrics["mAP50-95"]
                if isinstance(metrics["mAP50-95"], float) and metrics["mAP50-95"] <= 1
                else metrics["mAP50-95"] / 100
            ),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
        },
        "speed": {
            "preprocess_ms": avg_preprocess,
            "inference_ms": avg_inference,
            "postprocess_ms": avg_postprocess,
            "total_ms": avg_total,
            "fps": 1000 / avg_total,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    output_file = output_dir / f"{model_name}_coco_val2017_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")

    return results


def main():
    """Main evaluation function.

    Returns:
        Exit code: 0 on success, 1 on error (unknown model or missing dataset).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    ensure_runtime_dirs(_PROJECT_DIR)

    # Expand model list
    models = args.model
    if "all" in [m.lower() for m in models]:
        models = ALL_MODELS
    else:
        # Validate model names
        for m in models:
            if m not in ALL_MODELS:
                logger.error(f"[ERROR] Unknown model: {m}")
                logger.error(f"Available models: {', '.join(ALL_MODELS)}")
                return 1

    logger.info("=" * 70)
    logger.info("YOLO26 MLX - COCO val2017 Evaluation")
    logger.info("=" * 70)
    logger.info(f"Models:       {', '.join(models)}")
    logger.info(f"Dataset:      {args.data}")
    logger.info(f"Image size:   {args.imgsz}")
    logger.info(f"Batch size:   {args.batch}")
    logger.info(f"Conf thresh:  {args.conf}")
    logger.info(f"IoU thresh:   {args.iou}")
    if args.subset:
        logger.info(f"Subset:       {args.subset} images")
    logger.info("=" * 70)

    # Check dataset exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"\n[ERROR] Dataset not found at {data_path}")
        logger.error("Please download COCO val2017:")
        logger.error("  ./scripts/download_coco_val2017.sh")
        return 1

    # Load dataset once (shared across all models)
    logger.info(f"\nLoading COCO val2017 from {args.data}...")
    dataset = COCODataset(args.data, split="val2017", img_size=args.imgsz)
    logger.info(f"  {len(dataset)} images available")

    # Evaluate each model
    all_results = {}
    for model_name in models:
        try:
            results = evaluate_model(model_name, args, dataset)
            all_results[model_name] = results
        except Exception as e:
            logger.error(f"\n[ERROR] Failed to evaluate {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Print summary table
    if len(all_results) > 0:
        logger.info("\n" + "=" * 90)
        logger.info("SUMMARY - All Models")
        logger.info("=" * 90)

        # Official benchmarks
        official_map5095 = {
            "yolo26n": 40.1,
            "yolo26s": 47.8,
            "yolo26m": 52.5,
            "yolo26l": 54.4,
            "yolo26x": 56.9,
        }
        official_map50 = {
            "yolo26n": 55.3,
            "yolo26s": 64.0,
            "yolo26m": 68.0,
            "yolo26l": 70.0,
            "yolo26x": 72.0,
        }

        logger.info(
            f"{'Model':<12} {'mAP50-95':>10} {'(Official)':>12} {'mAP50':>10} {'(Official)':>12} {'FPS':>8} {'ms/img':>10}"
        )
        logger.info("-" * 90)

        for model_name in ALL_MODELS:
            if model_name in all_results:
                r = all_results[model_name]
                mlx_5095 = r["metrics"]["mAP50-95"] * 100
                mlx_50 = r["metrics"]["mAP50"] * 100
                off_5095 = official_map5095.get(model_name, 0)
                off_50 = official_map50.get(model_name, 0)
                fps = r["speed"]["fps"]
                ms = r["speed"]["total_ms"]
                logger.info(
                    f"{model_name:<12} {mlx_5095:>9.1f}% {off_5095:>10.1f}%  {mlx_50:>9.1f}% {off_50:>10.1f}%  {fps:>7.1f} {ms:>9.1f}ms"
                )

        logger.info("=" * 90)

        # Save combined results
        output_dir = Path(args.output)
        combined_file = output_dir / "yolo26_all_models_results.json"
        with open(combined_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nCombined results saved to {combined_file}")

    if args.subset:
        logger.info(
            f"\nNote: Evaluated on {args.subset} image subset. Full val2017 has 5000 images."
        )
        logger.info("Subset results may differ from full benchmark due to image selection.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
