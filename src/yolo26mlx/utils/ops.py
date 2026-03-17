# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 Operations - Pure MLX Implementation

Common operations for bounding box manipulation and anchor generation.
Reference: ultralytics/ultralytics/utils/ops.py and tal.py

MLX specifics:
- All operations work on MLX arrays
- NHWC format for feature maps
"""

import mlx.core as mx
import numpy as np


def xywh2xyxy(x: mx.array) -> mx.array:
    """Convert bounding boxes from xywh to xyxy format.

    Reference: ultralytics xywh2xyxy function

    Args:
        x: Boxes in (x_center, y_center, width, height) format, shape (..., 4)

    Returns:
        Boxes in (x1, y1, x2, y2) format, shape (..., 4)
    """
    assert x.shape[-1] == 4, f"Input shape last dim expected 4, got {x.shape}"

    # Centers and half-sizes
    xy = x[..., :2]  # center x, y
    wh = x[..., 2:] / 2  # half width, height

    # Convert to corners
    return mx.concatenate(
        [
            xy - wh,  # top-left (x1, y1)
            xy + wh,  # bottom-right (x2, y2)
        ],
        axis=-1,
    )


def xyxy2xywh(x: mx.array) -> mx.array:
    """Convert bounding boxes from xyxy to xywh format.

    Reference: ultralytics xyxy2xywh function

    Args:
        x: Boxes in (x1, y1, x2, y2) format, shape (..., 4)

    Returns:
        Boxes in (x_center, y_center, width, height) format, shape (..., 4)
    """
    assert x.shape[-1] == 4, f"Input shape last dim expected 4, got {x.shape}"

    x1, y1 = x[..., 0:1], x[..., 1:2]
    x2, y2 = x[..., 2:3], x[..., 3:4]

    return mx.concatenate(
        [
            (x1 + x2) / 2,  # center x
            (y1 + y2) / 2,  # center y
            x2 - x1,  # width
            y2 - y1,  # height
        ],
        axis=-1,
    )


def make_anchors(
    feats: list[mx.array], strides: mx.array, grid_cell_offset: float = 0.5
) -> tuple[mx.array, mx.array]:
    """Generate anchor points from feature maps.

    Reference: ultralytics make_anchors function

    Args:
        feats: List of feature maps, each (B, H, W, C) NHWC format
        strides: Stride values for each feature level
        grid_cell_offset: Offset for grid cell centers (0.5 = center)

    Returns:
        Tuple of (anchor_points, stride_tensor)
        - anchor_points: (total_anchors, 2) absolute coordinates
        - stride_tensor: (total_anchors, 1) stride for each anchor
    """
    anchor_points = []
    stride_tensor = []

    for i, feat in enumerate(feats):
        _, h, w, _ = feat.shape  # NHWC format
        stride = float(strides[i]) if hasattr(strides[i], "item") else strides[i]

        # Create grid coordinates
        sx = mx.arange(w, dtype=mx.float32) + grid_cell_offset
        sy = mx.arange(h, dtype=mx.float32) + grid_cell_offset

        # Meshgrid: create 2D grid
        # sy_grid[i,j] = sy[i], sx_grid[i,j] = sx[j]
        sy_grid = mx.broadcast_to(mx.reshape(sy, (h, 1)), (h, w))
        sx_grid = mx.broadcast_to(mx.reshape(sx, (1, w)), (h, w))

        # Stack (x, y) and flatten to (H*W, 2)
        anchor_point = mx.stack([mx.reshape(sx_grid, (-1,)), mx.reshape(sy_grid, (-1,))], axis=-1)

        anchor_points.append(anchor_point)
        stride_tensor.append(mx.full((h * w, 1), stride, dtype=mx.float32))

    return mx.concatenate(anchor_points, axis=0), mx.concatenate(stride_tensor, axis=0)


def dist2bbox(
    distance: mx.array, anchor_points: mx.array, xywh: bool = True, dim: int = -1
) -> mx.array:
    """Transform distance (ltrb) to bbox (xyxy or xywh).

    Reference: ultralytics dist2bbox function

    Args:
        distance: Distance predictions (left, top, right, bottom)
        anchor_points: Anchor point coordinates (x, y)
        xywh: Return xywh format if True, xyxy otherwise
        dim: Dimension to split along

    Returns:
        Bounding boxes in xywh or xyxy format
    """
    # Split into left-top and right-bottom
    lt, rb = mx.split(distance, 2, axis=dim)

    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb

    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return mx.concatenate([c_xy, wh], axis=dim)

    return mx.concatenate([x1y1, x2y2], axis=dim)


def bbox2dist(anchor_points: mx.array, bbox: mx.array, reg_max: int | None = None) -> mx.array:
    """Transform bbox (xyxy) to distance (ltrb).

    Reference: ultralytics bbox2dist function

    Args:
        anchor_points: Anchor point coordinates (x, y)
        bbox: Bounding boxes in xyxy format
        reg_max: Maximum regression value (optional clipping)

    Returns:
        Distance values (left, top, right, bottom)
    """
    x1y1, x2y2 = mx.split(bbox, 2, axis=-1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = mx.concatenate([lt, rb], axis=-1)

    if reg_max is not None:
        dist = mx.clip(dist, 0, reg_max - 0.01)

    return dist


def box_iou(box1: mx.array, box2: mx.array, eps: float = 1e-7) -> mx.array:
    """Compute IoU between two sets of boxes.

    Args:
        box1: Boxes (N, 4) in xyxy format
        box2: Boxes (M, 4) in xyxy format
        eps: Small value for numerical stability

    Returns:
        IoU matrix (N, M)
    """
    # Get areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Intersection
    inter_x1 = mx.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = mx.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = mx.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = mx.minimum(box1[:, None, 3], box2[None, :, 3])

    inter_w = mx.maximum(inter_x2 - inter_x1, 0)
    inter_h = mx.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    # Union
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + eps)


def non_max_suppression(
    boxes: mx.array,
    scores: mx.array,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
    max_detections: int = 300,
) -> mx.array:
    """Non-Maximum Suppression.

    Note: YOLO26 with end2end=True doesn't require NMS.
    This is provided for compatibility with non-E2E models.

    Implementation uses numpy internally because:
    - MLX 0.30.3 does not support single-arg mx.where() (nonzero)
    - MLX 0.30.3 does not support boolean array indexing
    - NMS is inherently sequential (greedy loop), so no GPU benefit

    Args:
        boxes: Bounding boxes (N, 4) in xyxy format
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for suppression
        score_threshold: Score threshold for filtering
        max_detections: Maximum detections to keep

    Returns:
        Indices of kept boxes
    """
    # Convert to numpy for the sequential NMS loop
    mx.eval(boxes, scores)
    np_boxes = np.array(boxes)
    np_scores = np.array(scores)

    # Filter by score
    valid = np.where(np_scores > score_threshold)[0]
    if len(valid) == 0:
        return mx.array([], dtype=mx.int32)

    np_boxes = np_boxes[valid]
    np_scores = np_scores[valid]

    # Sort by score (descending)
    order = np.argsort(-np_scores)
    np_boxes = np_boxes[order]
    valid = valid[order]

    # Greedy NMS
    keep = []
    x1, y1, x2, y2 = np_boxes[:, 0], np_boxes[:, 1], np_boxes[:, 2], np_boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    remaining = list(range(len(np_boxes)))

    while remaining and len(keep) < max_detections:
        i = remaining[0]
        keep.append(int(valid[i]))
        remaining = remaining[1:]

        if not remaining:
            break

        # Compute IoU between kept box and remaining boxes
        rem = np.array(remaining)
        ix1 = np.maximum(x1[i], x1[rem])
        iy1 = np.maximum(y1[i], y1[rem])
        ix2 = np.minimum(x2[i], x2[rem])
        iy2 = np.minimum(y2[i], y2[rem])

        iw = np.maximum(ix2 - ix1, 0)
        ih = np.maximum(iy2 - iy1, 0)
        inter = iw * ih
        union = areas[i] + areas[rem] - inter
        ious = inter / (union + 1e-7)

        # Keep boxes with IoU < threshold
        remaining = [remaining[j] for j in range(len(remaining)) if ious[j] < iou_threshold]

    return mx.array(keep, dtype=mx.int32)
