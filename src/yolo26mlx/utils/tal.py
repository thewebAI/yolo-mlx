# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Task-Aligned Assigner for YOLO26 - Pure MLX Implementation

Label assignment for training.
Reference: ultralytics/ultralytics/utils/tal.py

MLX specifics:
- Fully vectorized operations using mx.array.at[].add() for scatter operations
- Uses mx.take_along_axis for advanced indexing
- No Python loops over batch/anchors - all operations run on GPU
"""

import math

import mlx.core as mx


class TaskAlignedAssigner:
    """Task-Aligned Assigner for YOLO26.

    Reference: ultralytics TaskAlignedAssigner

    Assigns ground-truth objects to anchors based on task-aligned metric,
    combining classification and localization information.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
        topk2: int | None = None,
        stride: list[int] | None = None,
    ):
        """Initialize TaskAlignedAssigner.

        Args:
            topk: Number of top-k candidates to consider
            num_classes: Number of object classes
            alpha: Weight for classification score in alignment metric
            beta: Weight for IoU score in alignment metric
            eps: Small epsilon for numerical stability
            topk2: Secondary topk for filtering (e.g. 1 for one2one head).
                   If None, defaults to topk (no secondary filtering).
            stride: List of stride values for feature levels [8, 16, 32].
                    Used for small-box expansion in anchor selection.
        """
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.stride = stride or [8, 16, 32]
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def __call__(
        self,
        pd_scores: mx.array,
        pd_bboxes: mx.array,
        anc_points: mx.array,
        gt_labels: mx.array,
        gt_bboxes: mx.array,
        mask_gt: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Compute task-aligned assignment.

        Reference: ultralytics TaskAlignedAssigner.forward

        Args:
            pd_scores: Predicted scores (B, N, C), N=num_anchors, C=num_classes
            pd_bboxes: Predicted boxes (B, N, 4) in xyxy format
            anc_points: Anchor points (N, 2)
            gt_labels: Ground truth labels (B, M, 1), M=max_num_boxes
            gt_bboxes: Ground truth boxes (B, M, 4) in xyxy format
            mask_gt: Valid ground truth mask (B, M, 1)

        Returns:
            Tuple of (target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            return self._get_empty_assignments(pd_scores)

        # Get positive mask and alignment metric
        mask_pos, align_metric, overlaps = self._get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        # Select highest overlaps for conflicting assignments
        target_gt_idx, fg_mask, mask_pos = self._select_highest_overlaps(
            mask_pos, overlaps, align_metric
        )

        # Get final assignments
        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # Normalize target scores by alignment metric
        # Reference: ultralytics uses amax for normalization
        align_metric = align_metric * mask_pos
        pos_align_metrics = mx.max(align_metric, axis=-1, keepdims=True)  # (B, M)
        pos_overlaps = mx.max(overlaps * mask_pos, axis=-1, keepdims=True)  # (B, M)

        # Normalize: align_metric * max_overlap / max_align_metric
        norm_align_metric = mx.max(
            align_metric * pos_overlaps / (pos_align_metrics + self.eps), axis=-2
        )  # (B, N)
        target_scores = target_scores * mx.expand_dims(norm_align_metric, -1)

        return target_labels, target_bboxes, target_scores, fg_mask.astype(mx.bool_), target_gt_idx

    def _get_pos_mask(
        self,
        pd_scores: mx.array,
        pd_bboxes: mx.array,
        gt_labels: mx.array,
        gt_bboxes: mx.array,
        anc_points: mx.array,
        mask_gt: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Get positive mask and alignment metrics.

        Args:
            pd_scores: Predicted class scores (B, N, C), N=num_anchors, C=num_classes.
            pd_bboxes: Predicted bounding boxes (B, N, 4) in xyxy format.
            gt_labels: Ground truth class labels (B, M, 1), M=max_num_boxes.
            gt_bboxes: Ground truth bounding boxes (B, M, 4) in xyxy format.
            anc_points: Anchor center points (N, 2).
            mask_gt: Valid ground truth mask (B, M, 1).

        Returns:
            Tuple of (positive assignment mask, alignment metric, IoU overlaps).
        """
        # Check if anchors are inside GT boxes (with small-box expansion)
        mask_in_gts = self._select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)  # (B, M, N)

        # Get alignment metric and overlaps
        align_metric, overlaps = self._get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )

        # Select top-k candidates
        mask_topk = self._select_topk_candidates(
            align_metric,
            topk_mask=mx.broadcast_to(mask_gt, (self.bs, self.n_max_boxes, self.topk)).astype(
                mx.bool_
            ),
        )

        # Final positive mask
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def _get_box_metrics(
        self,
        pd_scores: mx.array,
        pd_bboxes: mx.array,
        gt_labels: mx.array,
        gt_bboxes: mx.array,
        mask_gt: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute alignment metric as score^alpha * IoU^beta.

        Args:
            pd_scores: Predicted class scores (B, N, C).
            pd_bboxes: Predicted bounding boxes (B, N, 4) in xyxy format.
            gt_labels: Ground truth class labels (B, M, 1).
            gt_bboxes: Ground truth bounding boxes (B, M, 4) in xyxy format.
            mask_gt: Valid ground truth mask (B, M, N), broadcast from (B, M, 1).

        Returns:
            Tuple of (alignment metric (B, M, N), IoU overlaps (B, M, N)).
        """
        na = pd_bboxes.shape[1]  # num anchors

        # Compute IoU between predicted and GT boxes
        # pd_bboxes: (B, N, 4) -> (B, 1, N, 4)
        # gt_bboxes: (B, M, 4) -> (B, M, 1, 4)
        overlaps = self._compute_iou(
            mx.expand_dims(gt_bboxes, 2),  # (B, M, 1, 4)
            mx.expand_dims(pd_bboxes, 1),  # (B, 1, N, 4)
        )  # (B, M, N)

        # Get scores for GT classes (VECTORIZED using take_along_axis)
        # gt_labels: (B, M, 1) -> indices into pd_scores along class dim
        gt_labels_int = gt_labels.squeeze(-1).astype(mx.int32)  # (B, M)

        # Expand gt_labels to match anchor dimension for gathering
        # gt_labels_expanded: (B, M, N) - same class index repeated for all anchors
        gt_labels_expanded = mx.broadcast_to(
            mx.expand_dims(gt_labels_int, -1), (self.bs, self.n_max_boxes, na)  # (B, M, 1)
        )

        # Expand pd_scores to match GT dimension for gathering
        # pd_scores: (B, N, C) -> (B, 1, N, C) -> (B, M, N, C)
        pd_scores_expanded = mx.broadcast_to(
            mx.expand_dims(pd_scores, 1),  # (B, 1, N, C)
            (self.bs, self.n_max_boxes, na, self.num_classes),
        )

        # Gather scores using take_along_axis
        # We need to select the score at gt_labels_expanded[b,m,n] from pd_scores_expanded[b,m,n,:]
        bbox_scores = mx.take_along_axis(
            pd_scores_expanded, mx.expand_dims(gt_labels_expanded, -1), axis=-1  # (B, M, N, 1)
        ).squeeze(
            -1
        )  # (B, M, N)

        # Apply mask - zero out invalid GT positions
        mask_gt_expanded = mx.broadcast_to(mask_gt, (self.bs, self.n_max_boxes, na))
        bbox_scores = bbox_scores * mask_gt_expanded

        # Alignment metric = score^alpha * IoU^beta
        align_metric = mx.power(bbox_scores, self.alpha) * mx.power(overlaps, self.beta)

        return align_metric, overlaps

    def _select_topk_candidates(
        self,
        metrics: mx.array,
        topk_mask: mx.array,
    ) -> mx.array:
        """Select top-k candidates based on metrics.

        Reference: ultralytics select_topk_candidates

        FULLY VECTORIZED using mx.array.at[].add() for scatter operations.

        Args:
            metrics: Alignment metrics (B, M, N)
            topk_mask: Valid mask (B, M, topk)

        Returns:
            Mask of top-k candidates (B, M, N)
        """
        n_anchors = metrics.shape[-1]

        # Get top-k indices using argsort (MLX topk returns VALUES, not indices!)
        # argsort returns indices in ascending order, so take the last k for largest
        sorted_indices = mx.argsort(metrics, axis=-1)  # (B, M, N) - ascending order
        topk_idxs = sorted_indices[..., -self.topk :]  # (B, M, topk) - last k are largest

        # Mask invalid indices (set to 0 so scatter doesn't affect invalid positions)
        topk_idxs = mx.where(topk_mask, topk_idxs, mx.zeros_like(topk_idxs))

        # VECTORIZED scatter_add using mx.array.at[].add()
        # Create count tensor using flat indexing for multi-dim scatter
        count_tensor = mx.zeros((self.bs, self.n_max_boxes, n_anchors), dtype=mx.float32)

        # Create batch and GT indices for all top-k positions
        # batch_idx: (B, M, topk) - same batch index for each position
        batch_idx = mx.broadcast_to(
            mx.arange(self.bs)[:, None, None], (self.bs, self.n_max_boxes, self.topk)
        )
        # gt_idx: (B, M, topk) - same GT index for each position
        gt_idx = mx.broadcast_to(
            mx.arange(self.n_max_boxes)[None, :, None], (self.bs, self.n_max_boxes, self.topk)
        )

        # Compute flat indices for scatter: batch * (M * N) + gt * N + anchor
        flat_idx = (
            batch_idx * (self.n_max_boxes * n_anchors) + gt_idx * n_anchors + topk_idxs
        ).astype(mx.int32)

        # Flatten everything for scatter
        flat_idx_1d = flat_idx.reshape(-1)  # (B * M * topk,)
        valid_mask_1d = topk_mask.reshape(-1).astype(mx.float32)  # (B * M * topk,)

        # Scatter add: add 1.0 for each valid top-k selection
        count_flat = count_tensor.flatten()
        count_flat = count_flat.at[flat_idx_1d].add(valid_mask_1d)
        count_tensor = count_flat.reshape(self.bs, self.n_max_boxes, n_anchors)

        # Return binary mask (count >= 1 means selected)
        # Note: We keep it as count for potential duplicate handling elsewhere
        return (count_tensor > 0).astype(mx.float32)

    def _select_highest_overlaps(
        self,
        mask_pos: mx.array,
        overlaps: mx.array,
        align_metric: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Select highest overlap when anchor is assigned to multiple GTs.

        Args:
            mask_pos: Positive assignment mask (B, M, N).
            overlaps: IoU overlap matrix (B, M, N).
            align_metric: Alignment scores (B, M, N).

        Returns:
            Tuple of (target GT index per anchor (B, N), foreground mask (B, N), updated mask_pos (B, M, N)).
        """
        n_anchors = mask_pos.shape[2]

        # Foreground mask: sum over GT dimension
        fg_mask = mx.sum(mask_pos, axis=1)  # (B, N)

        # Always run multi-GT resolution path (no data-dependent branch).
        # When no anchor has multiple GTs, mask_multi_gts is all-False and
        # mx.where returns the original mask_pos — same result, but
        # mx.compile-compatible.
        mask_multi_gts = mx.broadcast_to(
            mx.expand_dims(fg_mask, 1) > 1, mask_pos.shape
        )  # (B, M, N)

        # Get index of max overlap GT for each anchor
        max_overlaps_idx = mx.argmax(overlaps, axis=1)  # (B, N)

        # VECTORIZED one-hot creation using scatter
        is_max_overlaps = mx.zeros_like(mask_pos)  # (B, M, N)

        # Create indices for scatter
        batch_idx = mx.broadcast_to(mx.arange(self.bs)[:, None], (self.bs, n_anchors))  # (B, N)
        anchor_idx = mx.broadcast_to(mx.arange(n_anchors)[None, :], (self.bs, n_anchors))  # (B, N)

        # Flat index: batch * (M * N) + gt * N + anchor
        flat_idx = (
            (batch_idx * (self.n_max_boxes * n_anchors) + max_overlaps_idx * n_anchors + anchor_idx)
            .astype(mx.int32)
            .reshape(-1)
        )  # (B * N,)

        # Scatter: set 1.0 at max overlap positions
        is_max_flat = is_max_overlaps.flatten()
        is_max_flat = is_max_flat.at[flat_idx].add(mx.ones((self.bs * n_anchors,)))
        is_max_overlaps = is_max_flat.reshape(self.bs, self.n_max_boxes, n_anchors)

        # Update mask_pos: where multi-GT, use max overlap mask; otherwise keep original
        mask_pos = mx.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mx.sum(mask_pos, axis=1)

        # Secondary topk2 filtering (for one2one head: topk=7, topk2=1)
        # PyTorch: if self.topk2 != self.topk, filter mask_pos to keep only
        # topk2 best anchors per GT based on align_metric.
        if self.topk2 != self.topk:
            align_metric_masked = align_metric * mask_pos
            # Get topk2 best anchor indices per GT
            sorted_idx = mx.argsort(align_metric_masked, axis=-1)  # ascending
            topk2_idx = sorted_idx[..., -self.topk2 :]  # (B, M, topk2) — largest

            # Build a mask from topk2 indices using scatter
            topk2_mask = mx.zeros_like(mask_pos)
            batch_idx_t2 = mx.broadcast_to(
                mx.arange(self.bs)[:, None, None], (self.bs, self.n_max_boxes, self.topk2)
            )
            gt_idx_t2 = mx.broadcast_to(
                mx.arange(self.n_max_boxes)[None, :, None], (self.bs, self.n_max_boxes, self.topk2)
            )
            flat_idx_t2 = (
                (batch_idx_t2 * (self.n_max_boxes * n_anchors) + gt_idx_t2 * n_anchors + topk2_idx)
                .astype(mx.int32)
                .reshape(-1)
            )
            topk2_flat = topk2_mask.flatten()
            topk2_flat = topk2_flat.at[flat_idx_t2].add(mx.ones(flat_idx_t2.shape))
            topk2_mask = topk2_flat.reshape(self.bs, self.n_max_boxes, n_anchors)

            mask_pos = mask_pos * (topk2_mask > 0).astype(mask_pos.dtype)
            fg_mask = mx.sum(mask_pos, axis=1)

        # Get target GT index (which GT each anchor is assigned to)
        target_gt_idx = mx.argmax(mask_pos, axis=1)  # (B, N)

        return target_gt_idx, fg_mask, mask_pos

    def _get_targets(
        self,
        gt_labels: mx.array,
        gt_bboxes: mx.array,
        target_gt_idx: mx.array,
        fg_mask: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Get target assignments by gathering GT labels and boxes for each anchor.

        Args:
            gt_labels: Ground truth class labels (B, M, 1).
            gt_bboxes: Ground truth bounding boxes (B, M, 4).
            target_gt_idx: Assigned GT index per anchor (B, N).
            fg_mask: Foreground mask indicating positive anchors (B, N).

        Returns:
            Tuple of (target labels (B, N), target boxes (B, N, 4), one-hot target scores (B, N, C)).
        """
        bs, n_anchors = target_gt_idx.shape

        # Get target labels and boxes using gather
        # target_gt_idx: (B, N) - index into GT dimension

        # Flatten and gather
        batch_ind = mx.arange(bs)[:, None] * self.n_max_boxes  # (B, 1)
        flat_idx = (target_gt_idx + batch_ind).astype(mx.int32)  # (B, N)

        # Gather labels
        gt_labels_flat = gt_labels.reshape(-1)  # (B*M,)
        target_labels = gt_labels_flat[flat_idx.reshape(-1)].reshape(bs, n_anchors)  # (B, N)

        # Gather boxes
        gt_bboxes_flat = gt_bboxes.reshape(-1, 4)  # (B*M, 4)
        target_bboxes = gt_bboxes_flat[flat_idx.reshape(-1)].reshape(bs, n_anchors, 4)  # (B, N, 4)

        # Clamp labels to valid range
        target_labels = mx.clip(target_labels, 0, self.num_classes - 1).astype(mx.int32)

        # VECTORIZED one-hot creation using scatter
        target_scores = mx.zeros((bs, n_anchors, self.num_classes))

        # Create indices for all foreground anchors
        batch_idx = mx.broadcast_to(mx.arange(bs)[:, None], (bs, n_anchors))  # (B, N)
        anchor_idx = mx.broadcast_to(mx.arange(n_anchors)[None, :], (bs, n_anchors))  # (B, N)

        # Flat index: batch * (N * C) + anchor * C + class
        flat_idx_scores = (
            (
                batch_idx * (n_anchors * self.num_classes)
                + anchor_idx * self.num_classes
                + target_labels
            )
            .astype(mx.int32)
            .reshape(-1)
        )  # (B * N,)

        # Values: 1.0 where fg_mask is True, 0.0 otherwise
        fg_mask_flat = (fg_mask > 0).astype(mx.float32).reshape(-1)  # (B * N,)

        # Scatter: set target_scores at (b, n, class) = fg_mask value
        target_scores_flat = target_scores.flatten()
        target_scores_flat = target_scores_flat.at[flat_idx_scores].add(fg_mask_flat)
        target_scores = target_scores_flat.reshape(bs, n_anchors, self.num_classes)

        return target_labels, target_bboxes, target_scores

    def _get_empty_assignments(
        self,
        pd_scores: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Return empty assignments when no ground truth is present.

        Args:
            pd_scores: Predicted class scores (B, N, C), used to infer output shapes.

        Returns:
            Tuple of (background labels, zero boxes, zero scores, empty fg_mask, zero gt_idx).
        """
        bs, n_anchors, nc = pd_scores.shape

        return (
            mx.full(
                (bs, n_anchors), self.num_classes, dtype=mx.int32
            ),  # labels = num_classes (background)
            mx.zeros((bs, n_anchors, 4)),
            mx.zeros((bs, n_anchors, nc)),
            mx.zeros((bs, n_anchors), dtype=mx.bool_),
            mx.zeros((bs, n_anchors), dtype=mx.int32),
        )

    def _compute_iou(self, box1: mx.array, box2: mx.array) -> mx.array:
        """Compute CIoU between two sets of boxes (broadcast).

        Matches PyTorch: bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True)

        CIoU = IoU - center_dist/diag_dist - alpha*v
        where v = (4/pi^2) * (atan(w2/h2) - atan(w1/h1))^2
              alpha = v / (v - IoU + 1 + eps)

        Args:
            box1: (B, M, 1, 4) ground truth boxes in xyxy
            box2: (B, 1, N, 4) predicted boxes in xyxy

        Returns:
            CIoU: (B, M, N) clamped to [0, inf)
        """
        # Intersection
        inter_x1 = mx.maximum(box1[..., 0], box2[..., 0])
        inter_y1 = mx.maximum(box1[..., 1], box2[..., 1])
        inter_x2 = mx.minimum(box1[..., 2], box2[..., 2])
        inter_y2 = mx.minimum(box1[..., 3], box2[..., 3])

        inter_w = mx.maximum(inter_x2 - inter_x1, 0)
        inter_h = mx.maximum(inter_y2 - inter_y1, 0)
        inter_area = inter_w * inter_h

        # Areas
        w1 = box1[..., 2] - box1[..., 0]
        h1 = box1[..., 3] - box1[..., 1]
        w2 = box2[..., 2] - box2[..., 0]
        h2 = box2[..., 3] - box2[..., 1]
        area1 = w1 * h1
        area2 = w2 * h2

        # Union & IoU
        union_area = area1 + area2 - inter_area
        iou = inter_area / (union_area + self.eps)

        # Center distance
        cx1 = (box1[..., 0] + box1[..., 2]) / 2
        cy1 = (box1[..., 1] + box1[..., 3]) / 2
        cx2 = (box2[..., 0] + box2[..., 2]) / 2
        cy2 = (box2[..., 1] + box2[..., 3]) / 2
        center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

        # Enclosing box diagonal
        enc_x1 = mx.minimum(box1[..., 0], box2[..., 0])
        enc_y1 = mx.minimum(box1[..., 1], box2[..., 1])
        enc_x2 = mx.maximum(box1[..., 2], box2[..., 2])
        enc_y2 = mx.maximum(box1[..., 3], box2[..., 3])
        enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + self.eps

        rho = center_dist_sq / enc_diag_sq

        # Aspect ratio penalty
        v = (4 / (math.pi**2)) * (
            mx.arctan(w2 / (h2 + self.eps)) - mx.arctan(w1 / (h1 + self.eps))
        ) ** 2
        alpha = v / (v - iou + 1.0 + self.eps)
        # stop_gradient on alpha (matches PyTorch: alpha is detached)
        alpha = mx.stop_gradient(alpha)

        ciou = iou - rho - alpha * v
        return mx.maximum(ciou, 0.0)

    def _select_candidates_in_gts(
        self,
        anc_points: mx.array,
        gt_bboxes: mx.array,
        mask_gt: mx.array | None = None,
    ) -> mx.array:
        """Select anchor points inside ground truth boxes.

        Reference: ultralytics select_candidates_in_gts

        Includes small-box expansion (matching PyTorch): GT boxes smaller than
        stride[0] are expanded to stride[1] before point-in-box check. This
        ensures small objects get enough positive anchors.

        Args:
            anc_points: Anchor points (N, 2)
            gt_bboxes: GT boxes (B, M, 4) in xyxy format
            mask_gt: Valid GT mask (B, M, 1), used for small-box expansion

        Returns:
            Mask (B, M, N) where 1 = anchor inside GT box
        """
        n_anchors = anc_points.shape[0]

        # Small-box expansion: expand GT boxes smaller than stride[0]
        # to minimum stride[1] size (matching PyTorch tal.py L303-307)
        if mask_gt is not None:
            # Convert xyxy → xywh for size check
            cx = (gt_bboxes[..., 0:1] + gt_bboxes[..., 2:3]) * 0.5
            cy = (gt_bboxes[..., 1:2] + gt_bboxes[..., 3:4]) * 0.5
            w = gt_bboxes[..., 2:3] - gt_bboxes[..., 0:1]
            h = gt_bboxes[..., 3:4] - gt_bboxes[..., 1:2]

            # Expand small boxes: where w or h < stride[0] AND gt is valid
            stride_min = float(self.stride[0])  # smallest stride (e.g. 8)
            stride_val = float(self.stride[1])  # expansion target (e.g. 16)
            wh_small = mx.concatenate([w, h], axis=-1) < stride_min  # (B, M, 2)
            expand_mask = (wh_small * mask_gt).astype(mx.bool_)  # only valid GTs

            wh = mx.concatenate([w, h], axis=-1)  # (B, M, 2)
            wh = mx.where(expand_mask, mx.array(stride_val), wh)

            # Convert back to xyxy
            x1 = cx - wh[..., 0:1] * 0.5
            y1 = cy - wh[..., 1:2] * 0.5
            x2 = cx + wh[..., 0:1] * 0.5
            y2 = cy + wh[..., 1:2] * 0.5
            gt_bboxes = mx.concatenate([x1, y1, x2, y2], axis=-1)

        # Expand dimensions for broadcasting
        # anc_points: (N, 2) -> (1, 1, N, 2)
        # gt_bboxes: (B, M, 4) -> (B, M, 1, 4)
        points = mx.reshape(anc_points, (1, 1, n_anchors, 2))
        boxes = mx.expand_dims(gt_bboxes, 2)  # (B, M, 1, 4)

        # Check if points are inside boxes
        # lt = point - box_min, rb = box_max - point
        # All should be positive for point to be inside
        lt = points - boxes[..., :2]  # (B, M, N, 2)
        rb = boxes[..., 2:] - points  # (B, M, N, 2)

        bbox_deltas = mx.concatenate([lt, rb], axis=-1)  # (B, M, N, 4)

        # Point is inside if all deltas > 0
        return (mx.min(bbox_deltas, axis=-1) > self.eps).astype(mx.float32)
