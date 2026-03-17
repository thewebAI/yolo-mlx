# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 MLX Utilities Package

Utility functions for loss computation, operations, and task-aligned assignment.
Reference: ultralytics/ultralytics/utils/

MLX specifics:
- All utilities use MLX arrays and operations
- NHWC format throughout
"""

from yolo26mlx.utils.coco_metrics import COCOMetrics, compute_coco_metrics
from yolo26mlx.utils.loss import (
    # Component losses
    BboxLoss,
    BCEDiceLoss,
    DFLoss,
    E2EDetectLoss,
    # E2E losses (YOLO26 primary)
    E2ELoss,
    FocalLoss,
    KeypointLoss,
    MultiChannelDiceLoss,
    VarifocalLoss,
    # Utilities
    bbox_iou,
    v8ClassificationLoss,
    # Detection losses
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from yolo26mlx.utils.ops import (
    bbox2dist,
    box_iou,
    dist2bbox,
    make_anchors,
    non_max_suppression,
    xywh2xyxy,
    xyxy2xywh,
)
from yolo26mlx.utils.tal import TaskAlignedAssigner

__all__ = [
    # E2E Loss (YOLO26 primary)
    "E2ELoss",
    "E2EDetectLoss",
    # Detection losses
    "v8DetectionLoss",
    "v8SegmentationLoss",
    "v8PoseLoss",
    "v8OBBLoss",
    "v8ClassificationLoss",
    # Component losses
    "BboxLoss",
    "DFLoss",
    "FocalLoss",
    "VarifocalLoss",
    "KeypointLoss",
    "MultiChannelDiceLoss",
    "BCEDiceLoss",
    "bbox_iou",
    # Operations
    "xywh2xyxy",
    "xyxy2xywh",
    "make_anchors",
    "dist2bbox",
    "bbox2dist",
    "box_iou",
    "non_max_suppression",
    # Assignment
    "TaskAlignedAssigner",
    # COCO Metrics
    "COCOMetrics",
    "compute_coco_metrics",
]
