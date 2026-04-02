# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Distance computation and assignment utilities for object tracking.

All tensor math uses MLX; the Hungarian algorithm uses the `lap` package
(Jonker-Volgenant) for speed, with a pure-Python fallback.
"""

import mlx.core as mx
import numpy as np

from yolo26mlx.utils.ops import box_iou


def linear_assignment(cost_matrix, thresh):
    """Solve the linear assignment problem with a cost threshold.

    Uses lap.lapjv (Jonker-Volgenant) if available, otherwise falls back to
    scipy.optimize.linear_sum_assignment.

    Args:
        cost_matrix: mx.array cost matrix of shape (N, M).
        thresh: Maximum cost for a valid assignment.

    Returns:
        matches: list of [row, col] pairs for valid assignments.
        unmatched_a: tuple of unmatched row indices.
        unmatched_b: tuple of unmatched column indices.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    # Evaluate MLX array to numpy for the assignment solver
    if isinstance(cost_matrix, mx.array):
        mx.eval(cost_matrix)
        cost_np = np.array(cost_matrix, copy=False)
    else:
        cost_np = np.asarray(cost_matrix)

    try:
        import lap

        _, x, y = lap.lapjv(cost_np, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx_] for ix, mx_ in enumerate(x) if mx_ >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    except ImportError:
        try:
            import scipy.optimize
        except ImportError:
            raise ImportError(
                "Tracking requires 'lap' or 'scipy'. "
                "Install with: pip install yolo-mlx[tracking]"
            ) from None

        x, y = scipy.optimize.linear_sum_assignment(cost_np)
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_np[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(range(cost_np.shape[0]))
            unmatched_b = list(range(cost_np.shape[1]))
        else:
            unmatched_a = list(set(range(cost_np.shape[0])) - set(matches[:, 0]))
            unmatched_b = list(set(range(cost_np.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks, btracks):
    """Compute cost matrix based on IoU between two sets of tracks.

    Args:
        atracks: list of STrack objects or mx.array bounding boxes (xyxy).
        btracks: list of STrack objects or mx.array bounding boxes (xyxy).

    Returns:
        mx.array cost matrix of shape (len(atracks), len(btracks)), where
        cost = 1 - IoU.
    """
    if (atracks and isinstance(atracks[0], mx.array)) or (
        btracks and isinstance(btracks[0], mx.array)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xyxy for track in atracks]
        btlbrs = [track.xyxy for track in btracks]

    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return mx.zeros((len(atlbrs), len(btlbrs)))

    boxes_a = mx.stack(atlbrs) if not isinstance(atlbrs, mx.array) else atlbrs
    boxes_b = mx.stack(btlbrs) if not isinstance(btlbrs, mx.array) else btlbrs

    # Ensure 2D
    if boxes_a.ndim == 1:
        boxes_a = mx.expand_dims(boxes_a, 0)
    if boxes_b.ndim == 1:
        boxes_b = mx.expand_dims(boxes_b, 0)

    ious = box_iou(boxes_a, boxes_b)
    return 1.0 - ious


def embedding_distance(tracks, detections, metric="cosine"):
    """Compute distance between tracks and detections based on embeddings.

    Args:
        tracks: list of STrack objects with smooth_feat attribute.
        detections: list of STrack/BaseTrack objects with curr_feat attribute.
        metric: Distance metric — "cosine" supported.

    Returns:
        mx.array cost matrix of shape (len(tracks), len(detections)).
    """
    if len(tracks) == 0 or len(detections) == 0:
        return mx.zeros((len(tracks), len(detections)))

    det_features = mx.stack([det.curr_feat for det in detections])  # (M, D)
    track_features = mx.stack([t.smooth_feat for t in tracks])  # (N, D)

    if metric == "cosine":
        # Normalize
        t_norm = mx.linalg.norm(track_features, axis=1, keepdims=True)
        d_norm = mx.linalg.norm(det_features, axis=1, keepdims=True)
        t_normed = track_features / mx.maximum(t_norm, mx.array([1e-6]))
        d_normed = det_features / mx.maximum(d_norm, mx.array([1e-6]))
        similarity = mx.matmul(t_normed, mx.transpose(d_normed))
        cost = mx.maximum(1.0 - similarity, mx.array([0.0]))
    else:
        raise ValueError(f"Unsupported embedding distance metric: {metric}")

    return cost


def fuse_score(cost_matrix, detections):
    """Fuse cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix: mx.array cost matrix of shape (N, M).
        detections: list of objects with a score attribute.

    Returns:
        mx.array fused cost matrix of shape (N, M).
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = mx.array([det.score for det in detections])
    # Broadcast: (N, M) * (1, M)
    fuse_sim = iou_sim * mx.expand_dims(det_scores, 0)
    return 1.0 - fuse_sim
