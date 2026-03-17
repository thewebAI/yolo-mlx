# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
COCO Evaluation Metrics for YOLO26 MLX

Implements COCO-style mAP calculation following the official COCO evaluation protocol.
"""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class COCOMetrics:
    """COCO-style evaluation metrics for object detection.

    Computes mAP@0.5 and mAP@0.5:0.95 following the official COCO protocol.

    Attributes:
        num_classes: Number of object classes (80 for COCO)
        iou_thresholds: IoU thresholds for mAP calculation
        class_names: List of class names
    """

    def __init__(self, num_classes: int = 80, class_names: list[str] | None = None):
        """Initialize COCO metrics calculator.

        Args:
            num_classes: Number of object classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]

        # IoU thresholds: 0.5:0.05:0.95 (10 thresholds)
        self.iou_thresholds = np.linspace(0.5, 0.95, 10)

        # Statistics storage
        self.reset()

    def reset(self):
        """Reset all accumulated statistics."""
        # Per-class, per-image statistics for proper matching
        # {class_id: [(image_id, pred_scores, iou_matrix, num_gt)]}
        self.class_data = defaultdict(list)
        self.num_gt_per_class = defaultdict(int)  # {class_id: count}

        # Image-level tracking
        self.processed_images = set()

    def update(self, predictions: dict, ground_truth: dict, image_id: int):
        """Update metrics with predictions for one image.

        Args:
            predictions: Dict with 'boxes', 'scores', 'labels'
                - boxes: (N, 4) in xyxy format, normalized [0, 1]
                - scores: (N,) confidence scores
                - labels: (N,) class indices
            ground_truth: Dict with 'boxes', 'labels', 'iscrowd'
                - boxes: (M, 4) in xyxy format, normalized [0, 1]
                - labels: (M,) class indices
                - iscrowd: (M,) crowd flags
            image_id: Unique image identifier
        """
        if image_id in self.processed_images:
            return
        self.processed_images.add(image_id)

        pred_boxes = predictions.get("boxes", np.zeros((0, 4)))
        pred_scores = predictions.get("scores", np.zeros(0))
        pred_labels = predictions.get("labels", np.zeros(0, dtype=np.int64))

        gt_boxes = ground_truth.get("boxes", np.zeros((0, 4)))
        gt_labels = ground_truth.get("labels", np.zeros(0, dtype=np.int64))
        gt_iscrowd = ground_truth.get("iscrowd", np.zeros(len(gt_labels), dtype=np.int64))

        # Ensure gt_iscrowd has correct shape
        if len(gt_iscrowd) != len(gt_labels):
            gt_iscrowd = np.zeros(len(gt_labels), dtype=np.int64)

        # Count ground truth per class (excluding crowd)
        for cls_id in gt_labels[gt_iscrowd == 0]:
            self.num_gt_per_class[cls_id] += 1

        # Process predictions per class
        for cls_id in range(self.num_classes):
            # Get predictions for this class
            pred_mask = pred_labels == cls_id
            cls_pred_boxes = pred_boxes[pred_mask]
            cls_pred_scores = pred_scores[pred_mask]

            # Get ground truth for this class (excluding crowd)
            gt_mask = (gt_labels == cls_id) & (gt_iscrowd == 0)
            cls_gt_boxes = gt_boxes[gt_mask]

            if len(cls_pred_boxes) == 0 and len(cls_gt_boxes) == 0:
                continue

            # Sort predictions by score
            if len(cls_pred_scores) > 0:
                sort_idx = np.argsort(-cls_pred_scores)
                cls_pred_boxes = cls_pred_boxes[sort_idx]
                cls_pred_scores = cls_pred_scores[sort_idx]

            # Compute IoU matrix and store for later matching
            if len(cls_pred_boxes) > 0 and len(cls_gt_boxes) > 0:
                ious = self._compute_iou(cls_pred_boxes, cls_gt_boxes)
            elif len(cls_pred_boxes) > 0:
                # No GT - all predictions are FP
                ious = np.zeros((len(cls_pred_boxes), 0))
            else:
                # No predictions
                ious = np.zeros((0, len(cls_gt_boxes)))

            # Store data for this image/class for later matching
            self.class_data[cls_id].append(
                {"scores": cls_pred_scores, "ious": ious, "num_gt": len(cls_gt_boxes)}
            )

    def _compute_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of boxes.

        Args:
            boxes1: (N, 4) boxes in xyxy format
            boxes2: (M, 4) boxes in xyxy format

        Returns:
            IoU matrix (N, M)
        """
        # Intersection
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

        inter_w = np.maximum(0, x2 - x1)
        inter_h = np.maximum(0, y2 - y1)
        inter_area = inter_w * inter_h

        # Areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Union
        union_area = area1[:, None] + area2[None, :] - inter_area

        return inter_area / (union_area + 1e-7)

    def compute(self) -> dict[str, float]:
        """Compute final mAP metrics.

        Returns:
            Dictionary with mAP metrics
        """
        ap_per_class = {}
        ap50_per_class = {}

        total_tp_50 = 0
        total_fp_50 = 0
        total_gt = 0

        # Compute AP for each class
        for cls_id in range(self.num_classes):
            class_images = self.class_data[cls_id]
            num_gt = sum(img["num_gt"] for img in class_images)
            total_gt += num_gt

            if num_gt == 0:
                ap_per_class[cls_id] = 0.0
                ap50_per_class[cls_id] = 0.0
                continue

            # Collect all predictions for this class
            has_predictions = any(len(img["scores"]) > 0 for img in class_images)
            if not has_predictions:
                ap_per_class[cls_id] = 0.0
                ap50_per_class[cls_id] = 0.0
                continue

            # Compute AP at each IoU threshold
            aps = []
            for iou_thresh in self.iou_thresholds:
                ap, tp, fp = self._compute_ap_at_threshold_proper(class_images, num_gt, iou_thresh)
                aps.append(ap)

                # Track TP/FP at IoU=0.5 for precision/recall
                if abs(iou_thresh - 0.5) < 0.01:
                    total_tp_50 += tp
                    total_fp_50 += fp

            # mAP@0.5:0.95
            ap_per_class[cls_id] = np.mean(aps)

            # mAP@0.5
            ap50_per_class[cls_id] = aps[0]

        # Compute mean across classes (only for classes with GT)
        valid_classes = [
            c
            for c in range(self.num_classes)
            if sum(img["num_gt"] for img in self.class_data[c]) > 0
        ]

        if len(valid_classes) == 0:
            return {
                "mAP50": 0.0,
                "mAP50-95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "per_class_ap": {},
                "per_class_ap50": {},
            }

        mAP50 = np.mean([ap50_per_class[c] for c in valid_classes])
        mAP50_95 = np.mean([ap_per_class[c] for c in valid_classes])

        precision = total_tp_50 / (total_tp_50 + total_fp_50 + 1e-7)
        recall = total_tp_50 / (total_gt + 1e-7)

        return {
            "mAP50": float(mAP50),
            "mAP50-95": float(mAP50_95),
            "precision": float(precision),
            "recall": float(recall),
            "per_class_ap": {self.class_names[c]: float(ap_per_class[c]) for c in valid_classes},
            "per_class_ap50": {
                self.class_names[c]: float(ap50_per_class[c]) for c in valid_classes
            },
        }

    def _compute_ap_at_threshold_proper(
        self, class_images: list, num_gt: int, iou_thresh: float
    ) -> tuple[float, int, int]:
        """Compute AP at a specific IoU threshold with proper greedy matching.

        Uses per-image greedy matching where each GT can only match one prediction.

        Args:
            class_images: List of {scores, ious, num_gt} per image
            num_gt: Total number of ground truth objects
            iou_thresh: IoU threshold for TP/FP assignment

        Returns:
            Tuple of (AP value, total_tp, total_fp)
        """
        # Collect all predictions across images with their matching info
        all_preds = []  # [(score, is_tp)]

        for img_data in class_images:
            scores = img_data["scores"]
            ious = img_data["ious"]
            img_num_gt = img_data["num_gt"]

            if len(scores) == 0:
                continue

            # Greedy matching for this image
            # Predictions are already sorted by score
            gt_matched = np.zeros(img_num_gt, dtype=bool) if img_num_gt > 0 else np.array([])

            for i, score in enumerate(scores):
                if img_num_gt == 0 or ious.shape[1] == 0:
                    # No GT - this is a false positive
                    all_preds.append((score, False))
                    continue

                # Find best unmatched GT
                pred_ious = ious[i].copy()
                pred_ious[gt_matched] = 0  # Mask already matched GTs

                best_iou = pred_ious.max()
                if best_iou >= iou_thresh:
                    best_gt_idx = pred_ious.argmax()
                    gt_matched[best_gt_idx] = True
                    all_preds.append((score, True))  # TP
                else:
                    all_preds.append((score, False))  # FP

        if len(all_preds) == 0:
            return 0.0, 0, 0

        # Sort all predictions by score (globally across images)
        all_preds.sort(key=lambda x: -x[0])

        # Compute precision-recall curve
        tp_array = np.array([1.0 if p[1] else 0.0 for p in all_preds])
        fp_array = 1.0 - tp_array

        tp_cumsum = np.cumsum(tp_array)
        fp_cumsum = np.cumsum(fp_array)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
        recall = tp_cumsum / (num_gt + 1e-7)

        # 101-point interpolation (COCO style)
        recall_thresholds = np.linspace(0, 1, 101)
        interpolated_precision = np.zeros(101)

        for i, r_thresh in enumerate(recall_thresholds):
            mask = recall >= r_thresh
            if mask.any():
                interpolated_precision[i] = precision[mask].max()
            else:
                interpolated_precision[i] = 0

        ap = float(np.mean(interpolated_precision))
        total_tp = int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
        total_fp = int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0

        return ap, total_tp, total_fp

    def print_results(self, metrics: dict):
        """Print results in Ultralytics format.

        Args:
            metrics: Dictionary of computed metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info(
            f"{'Class':>20} {'Images':>10} {'Instances':>10} {'P':>10} {'R':>10} {'mAP50':>10} {'mAP50-95':>10}"
        )
        logger.info("=" * 80)

        # Print overall results
        logger.info(
            f"{'all':>20} {len(self.processed_images):>10} {sum(self.num_gt_per_class.values()):>10} "
            f"{metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
            f"{metrics['mAP50']:>10.3f} {metrics['mAP50-95']:>10.3f}"
        )

        # Print per-class results
        if "per_class_ap50" in metrics:
            logger.info("-" * 80)
            for cls_name in sorted(metrics["per_class_ap50"].keys()):
                cls_idx = self.class_names.index(cls_name) if cls_name in self.class_names else -1
                num_instances = self.num_gt_per_class.get(cls_idx, 0)
                ap50 = metrics["per_class_ap50"].get(cls_name, 0)
                ap = metrics["per_class_ap"].get(cls_name, 0)
                logger.info(
                    f"{cls_name:>20} {'-':>10} {num_instances:>10} {'-':>10} {'-':>10} {ap50:>10.3f} {ap:>10.3f}"
                )

        logger.info("=" * 80)


def compute_coco_metrics(
    predictions: list[dict],
    ground_truths: list[dict],
    num_classes: int = 80,
    class_names: list[str] | None = None,
) -> dict:
    """Convenience function to compute COCO metrics.

    Args:
        predictions: List of prediction dicts per image
        ground_truths: List of ground truth dicts per image
        num_classes: Number of classes
        class_names: Optional class names

    Returns:
        Dictionary of metrics
    """
    metrics = COCOMetrics(num_classes=num_classes, class_names=class_names)

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths, strict=True)):
        image_id = gt.get("image_id", i)
        metrics.update(pred, gt, image_id)

    return metrics.compute()
