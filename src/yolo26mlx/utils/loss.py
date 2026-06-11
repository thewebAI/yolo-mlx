# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 Loss Functions - Pure MLX Implementation

Loss functions for detection, segmentation, pose, and OBB tasks.
Reference: ultralytics/ultralytics/utils/loss.py

MLX v0.30.3 specifics:
- Uses mlx.nn.losses for BCE, cross_entropy
- Uses mx.einsum for mask computation
- Uses mx.array.at[].add() for scatter operations
- Fully vectorized operations
"""

import math
from typing import Any

import mlx.core as mx
import mlx.nn.losses as losses

from yolo26mlx.utils.tal import TaskAlignedAssigner

from .ops import bbox2dist, crop_mask, make_anchors


def bbox_iou(
    box1: mx.array,
    box2: mx.array,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> mx.array:
    """Compute IoU between box1 and box2.

    Reference: ultralytics/utils/metrics.py bbox_iou

    Supports various shapes as long as last dimension is 4.

    Args:
        box1: First boxes (..., 4)
        box2: Second boxes (..., 4)
        xywh: If True, boxes are in xywh format, else xyxy
        GIoU: If True, compute Generalized IoU.
        DIoU: If True, compute Distance IoU.
        CIoU: If True, compute Complete IoU.
        eps: Small epsilon for numerical stability

    Returns:
        IoU values
    """
    # Get coordinates
    if xywh:
        # Transform from xywh to xyxy
        x1, y1, w1, h1 = box1[..., 0:1], box1[..., 1:2], box1[..., 2:3], box1[..., 3:4]
        x2, y2, w2, h2 = box2[..., 0:1], box2[..., 1:2], box2[..., 2:3], box2[..., 3:4]
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_
    else:
        # Already xyxy
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0:1], box1[..., 1:2], box1[..., 2:3], box1[..., 3:4]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0:1], box2[..., 1:2], box2[..., 2:3], box2[..., 3:4]
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = mx.maximum(mx.minimum(b1_x2, b2_x2) - mx.maximum(b1_x1, b2_x1), 0) * mx.maximum(
        mx.minimum(b1_y2, b2_y2) - mx.maximum(b1_y1, b2_y1), 0
    )

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if CIoU or DIoU or GIoU:
        # Convex (smallest enclosing box)
        cw = mx.maximum(b1_x2, b2_x2) - mx.minimum(b1_x1, b2_x1)  # convex width
        ch = mx.maximum(b1_y2, b2_y2) - mx.minimum(b1_y1, b2_y1)  # convex height

        if CIoU or DIoU:
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared

            if CIoU:
                v = (4 / (math.pi**2)) * mx.power(
                    mx.arctan(w2 / (h2 + eps)) - mx.arctan(w1 / (h1 + eps)), 2
                )
                # Alpha should not contribute gradients (matches PyTorch's torch.no_grad())
                alpha = mx.stop_gradient(v / (v - iou + (1 + eps)))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU

        # GIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area

    return iou


class VarifocalLoss:
    """Varifocal Loss by Zhang et al.

    Reference: ultralytics/utils/loss.py VarifocalLoss
    https://arxiv.org/abs/2008.13367

    Focuses on hard-to-classify examples and balances positive/negative samples.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize VarifocalLoss with focusing and balancing parameters.

        Args:
            gamma: Focusing parameter that controls down-weighting of easy examples.
            alpha: Balancing factor between positive and negative samples.
        """
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, pred_score: mx.array, gt_score: mx.array, label: mx.array) -> mx.array:
        """Compute varifocal loss between predictions and ground truth.

        Args:
            pred_score: Predicted scores (logits)
            gt_score: Ground truth scores
            label: Binary labels (1 for positive, 0 for negative)

        Returns:
            Varifocal loss
        """
        pred_sigmoid = mx.sigmoid(pred_score)
        weight = self.alpha * mx.power(pred_sigmoid, self.gamma) * (1 - label) + gt_score * label

        # BCE with logits
        bce = losses.binary_cross_entropy(pred_score, gt_score, with_logits=True, reduction="none")
        loss = (bce * weight).mean(axis=1).sum()

        return loss


class FocalLoss:
    """Focal Loss for addressing class imbalance.

    Reference: ultralytics/utils/loss.py FocalLoss

    Down-weights easy examples and focuses on hard negatives.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss with focusing and balancing parameters.

        Args:
            gamma: Focusing parameter that increases loss for hard-to-classify examples.
            alpha: Balancing factor between positive and negative classes.
        """
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, pred: mx.array, label: mx.array) -> mx.array:
        """Calculate focal loss with modulating factors.

        Args:
            pred: Predictions (logits)
            label: Target labels

        Returns:
            Focal loss
        """
        # BCE with logits
        loss = losses.binary_cross_entropy(pred, label, with_logits=True, reduction="none")

        # Modulating factor
        pred_prob = mx.sigmoid(pred)
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = mx.power(1.0 - p_t, self.gamma)
        loss = loss * modulating_factor

        # Alpha factor
        if self.alpha > 0:
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss = loss * alpha_factor

        return loss.mean(axis=1).sum()


class DFLoss:
    """Distribution Focal Loss.

    Reference: ultralytics/utils/loss.py DFLoss
    https://ieeexplore.ieee.org/document/9792391

    FULLY VECTORIZED using take_along_axis for gather operations.
    """

    def __init__(self, reg_max: int = 16):
        """Initialize DFL with reg_max.

        Args:
            reg_max: Number of distribution bins for box regression.
        """
        self.reg_max = reg_max

    def __call__(self, pred_dist: mx.array, target: mx.array) -> mx.array:
        """Compute DFL loss.

        Args:
            pred_dist: Predicted distribution (N, reg_max)
            target: Target values (N,)

        Returns:
            DFL loss (N, 1)
        """
        target = mx.clip(target, 0, self.reg_max - 1 - 0.01)
        tl = mx.floor(target).astype(mx.int32)  # target left
        tr = tl + 1  # target right
        wl = tr.astype(mx.float32) - target  # weight left
        wr = 1 - wl  # weight right

        # Log softmax for numerical stability
        log_softmax = pred_dist - mx.logsumexp(pred_dist, axis=-1, keepdims=True)

        # VECTORIZED gather using take_along_axis
        tl_expanded = mx.expand_dims(tl, -1)  # (N, 1)
        tr_expanded = mx.expand_dims(tr, -1)  # (N, 1)

        # Gather log probabilities at target indices (cross entropy)
        loss_l = -mx.take_along_axis(log_softmax, tl_expanded, axis=-1).squeeze(-1)  # (N,)
        loss_r = -mx.take_along_axis(log_softmax, tr_expanded, axis=-1).squeeze(-1)  # (N,)

        return (loss_l * wl + loss_r * wr).mean(axis=-1, keepdims=True)


class BboxLoss:
    """Bounding Box Loss for YOLO26.

    Reference: ultralytics/utils/loss.py BboxLoss

    Handles both reg_max > 1 (DFL) and reg_max = 1 (L1) cases.

    Note: MLX does not support boolean indexing for reading (only for assignment).
    We use mx.where() and masked operations instead.
    """

    def __init__(self, reg_max: int = 16):
        """Initialize BboxLoss with DFL settings.

        Args:
            reg_max: Number of distribution bins; >1 uses DFL loss, =1 uses L1 loss.
        """
        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def __call__(
        self,
        pred_dist: mx.array,
        pred_bboxes: mx.array,
        anchor_points: mx.array,
        target_bboxes: mx.array,
        target_scores: mx.array,
        target_scores_sum: mx.array,
        fg_mask: mx.array,
        imgsz: mx.array,
        stride: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute IoU and DFL/L1 losses for bounding boxes.

        Reference: ultralytics BboxLoss.forward

        MLX Note: Uses mx.where() for masked operations since MLX doesn't
        support boolean indexing for reading arrays.

        Args:
            pred_dist: Predicted distributions (B, N, 4*reg_max) or (B, N, 4)
            pred_bboxes: Decoded predicted boxes (B, N, 4) in xyxy format
            anchor_points: Anchor points (N, 2)
            target_bboxes: Target boxes (B, N, 4) in xyxy format
            target_scores: Target scores (B, N, nc)
            target_scores_sum: Sum of target scores
            fg_mask: Foreground mask (B, N) - boolean
            imgsz: Image size (2,) as (H, W)
            stride: Stride tensor (N, 1)

        Returns:
            Tuple of (iou_loss, dfl_loss)
        """
        # Weight from target scores - sum over classes
        weight = mx.sum(target_scores, axis=-1, keepdims=True)  # (B, N, 1)

        # Expand fg_mask for broadcasting: (B, N) -> (B, N, 1)
        fg_mask_expanded = mx.expand_dims(fg_mask.astype(mx.float32), axis=-1)

        # NOTE: No early return for num_fg==0 — the masking below
        # naturally produces zero loss. Removing the data-dependent
        # branch makes this function mx.compile-compatible.

        # CIoU loss - compute for all, then mask
        # bbox_iou returns (..., 1) shaped tensor
        iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, CIoU=True)  # (B, N, 1)

        # Apply mask: zero out non-foreground contributions
        loss_iou_per_box = (1.0 - iou) * weight * fg_mask_expanded
        loss_iou = mx.sum(loss_iou_per_box) / target_scores_sum

        # DFL or L1 loss
        if self.dfl_loss is not None:
            # DFL case (reg_max > 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)

            # FULLY VECTORIZED DFL loss computation
            # pred_dist: (B, N, 4*reg_max) -> (B*N*4, reg_max) to compute all coords at once
            B, N, _ = pred_dist.shape
            # Reshape to separate coordinates: (B, N, 4, reg_max)
            pred_dist_4d = pred_dist.reshape(B, N, 4, self.reg_max)
            # Flatten B, N, coord dims: (B*N*4, reg_max)
            pred_dist_flat = pred_dist_4d.reshape(B * N * 4, self.reg_max)
            # Target: (B, N, 4) -> (B*N*4,)
            target_flat = target_ltrb.reshape(-1)

            # Single DFL call for all coordinates (no Python loop!)
            loss_dfl_flat = self.dfl_loss(pred_dist_flat, target_flat)  # (B*N*4, 1)

            # Reshape back: (B, N, 4) -> mean over coords -> (B, N, 1)
            loss_dfl_per_coord = loss_dfl_flat.reshape(B, N, 4)
            loss_dfl_per_box = loss_dfl_per_coord.mean(axis=-1, keepdims=True)  # (B, N, 1)

            # Apply mask and weight
            loss_dfl = mx.sum(loss_dfl_per_box * weight * fg_mask_expanded) / target_scores_sum
        else:
            # L1 case (reg_max = 1) - YOLO26 default
            target_ltrb = bbox2dist(anchor_points, target_bboxes)

            # Normalize by image size
            target_ltrb_scaled = target_ltrb * stride
            target_ltrb_norm = mx.concatenate(
                [
                    target_ltrb_scaled[..., 0:1] / imgsz[1],
                    target_ltrb_scaled[..., 1:2] / imgsz[0],
                    target_ltrb_scaled[..., 2:3] / imgsz[1],
                    target_ltrb_scaled[..., 3:4] / imgsz[0],
                ],
                axis=-1,
            )

            pred_dist_scaled = pred_dist * stride
            pred_dist_norm = mx.concatenate(
                [
                    pred_dist_scaled[..., 0:1] / imgsz[1],
                    pred_dist_scaled[..., 1:2] / imgsz[0],
                    pred_dist_scaled[..., 2:3] / imgsz[1],
                    pred_dist_scaled[..., 3:4] / imgsz[0],
                ],
                axis=-1,
            )

            # L1 loss per box, averaged over 4 coords
            l1_per_coord = mx.abs(pred_dist_norm - target_ltrb_norm)  # (B, N, 4)
            loss_dfl_per_box = l1_per_coord.mean(axis=-1, keepdims=True)  # (B, N, 1)

            # Apply mask and weight
            loss_dfl = mx.sum(loss_dfl_per_box * weight * fg_mask_expanded) / target_scores_sum

        return loss_iou, loss_dfl


class KeypointLoss:
    """Keypoint Loss for pose estimation.

    Reference: ultralytics/utils/loss.py KeypointLoss
    """

    def __init__(self, sigmas: mx.array):
        """Initialize KeypointLoss with OKS sigmas.

        Args:
            sigmas: Per-keypoint OKS sigma values controlling scale normalization (K,).
        """
        self.sigmas = sigmas

    def __call__(
        self, pred_kpts: mx.array, gt_kpts: mx.array, kpt_mask: mx.array, area: mx.array
    ) -> mx.array:
        """Calculate keypoint loss using OKS-based metric.

        Args:
            pred_kpts: Predicted keypoints (N, K, 2 or 3)
            gt_kpts: Ground truth keypoints (N, K, 2 or 3)
            kpt_mask: Mask for valid keypoints (N, K)
            area: Bounding box areas (N,)

        Returns:
            Keypoint loss
        """
        # d has shape (N, K)
        d = mx.power(pred_kpts[..., 0] - gt_kpts[..., 0], 2) + mx.power(
            pred_kpts[..., 1] - gt_kpts[..., 1], 2
        )

        kpt_loss_factor = kpt_mask.shape[1] / (mx.sum(kpt_mask != 0, axis=1) + 1e-9)

        # From cocoeval: e = d / ((2 * sigmas)^2 * (area + eps) * 2)
        # sigmas: (K,), area: (N,), d: (N, K)
        # Expand area to (N, 1) for proper broadcasting
        area_expanded = mx.expand_dims(area, axis=-1)  # (N, 1)
        sigma_squared = mx.power(2 * self.sigmas, 2)  # (K,)
        e = d / (sigma_squared * (area_expanded + 1e-9) * 2)

        return mx.mean(kpt_loss_factor.reshape(-1, 1) * ((1 - mx.exp(-e)) * kpt_mask))


class MultiChannelDiceLoss:
    """Multi-channel Dice Loss for segmentation.

    Reference: ultralytics/utils/loss.py MultiChannelDiceLoss
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize with smoothing and reduction options.

        Args:
            smooth: Laplacian smoothing factor to prevent division by zero.
            reduction: Reduction mode over the batch: 'mean', 'sum', or 'none'.
        """
        self.smooth = smooth
        self.reduction = reduction

    def __call__(self, pred: mx.array, target: mx.array) -> mx.array:
        """Calculate multi-channel Dice loss.

        Args:
            pred: Predicted masks (logits)
            target: Target masks

        Returns:
            Dice loss
        """
        pred = mx.sigmoid(pred)
        intersection = mx.sum(pred * target, axis=(2, 3))
        union = mx.sum(pred, axis=(2, 3)) + mx.sum(target, axis=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = mx.mean(dice_loss, axis=1)

        if self.reduction == "mean":
            return mx.mean(dice_loss)
        elif self.reduction == "sum":
            return mx.sum(dice_loss)
        return dice_loss


class BCEDiceLoss:
    """Combined BCE and Dice Loss for segmentation.

    Reference: ultralytics/utils/loss.py BCEDiceLoss
    """

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize with BCE and Dice weights.

        Args:
            weight_bce: Weight for the BCE loss component in the combined loss.
            weight_dice: Weight for the Dice loss component in the combined loss.
        """
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.dice = MultiChannelDiceLoss(smooth=1)

    def __call__(self, pred: mx.array, target: mx.array) -> mx.array:
        """Calculate combined BCE and Dice loss.

        Args:
            pred: Predicted masks (logits)
            target: Target masks

        Returns:
            Combined loss
        """
        bce = losses.binary_cross_entropy(pred, target, with_logits=True, reduction="mean")
        dice = self.dice(pred, target)
        return self.weight_bce * bce + self.weight_dice * dice


class v8DetectionLoss:
    """Detection Loss for YOLO26.

    Reference: ultralytics/utils/loss.py v8DetectionLoss

    Complete loss function for object detection training.

    MLX 0.30.3 Optimization Notes:
    - Uses fixed max_boxes for mx.compile compatibility
    - Avoids int(mx.array) which breaks compilation
    - Uses mx.where for conditional operations
    """

    # Maximum boxes per image - fixed for compile compatibility
    MAX_BOXES_PER_IMAGE = 100

    def __init__(
        self,
        model: Any,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        max_boxes: int = 100,
    ):
        """Initialize v8DetectionLoss with model parameters.

        Args:
            model: YOLO model (for accessing head parameters)
            tal_topk: Top-k for task-aligned assigner
            tal_topk2: Secondary top-k (optional)
            max_boxes: Maximum boxes per image (for compile compatibility)
        """
        # Get model hyperparameters
        self.hyp = model.args if hasattr(model, "args") else {}

        # Get Detect head parameters
        m = model.model[-1] if hasattr(model, "model") else model
        self.stride = m.stride if hasattr(m, "stride") else mx.array([8, 16, 32])
        self.nc = m.nc if hasattr(m, "nc") else 80
        self.reg_max = m.reg_max if hasattr(m, "reg_max") else 1
        self.no = self.nc + self.reg_max * 4

        self.use_dfl = self.reg_max > 1

        # Fixed max_boxes for compile compatibility
        self.max_boxes = max_boxes

        # Internal task-aligned assigner (matches PyTorch ultralytics pattern)
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            topk2=tal_topk2,
            stride=self.stride.tolist() if hasattr(self.stride, "tolist") else list(self.stride),
        )

        # Initialize losses
        self.bbox_loss = BboxLoss(self.reg_max)

        # DFL projection (only if reg_max > 1)
        self.proj = mx.arange(self.reg_max, dtype=mx.float32) if self.use_dfl else None

        # Default hyperparameters
        self.box_gain = self.hyp.get("box", 7.5) if isinstance(self.hyp, dict) else 7.5
        self.cls_gain = self.hyp.get("cls", 0.5) if isinstance(self.hyp, dict) else 0.5
        self.dfl_gain = self.hyp.get("dfl", 1.5) if isinstance(self.hyp, dict) else 1.5

    def preprocess(self, targets: mx.array, batch_size: int, scale_tensor: mx.array) -> mx.array:
        """Preprocess targets by converting format and scaling.

        FULLY VECTORIZED using sorting and cumsum for segmented counting.
        Uses fixed max_boxes for mx.compile compatibility.

        Args:
            targets: Raw targets (N, 6) with [batch_idx, cls, x, y, w, h]
            batch_size: Batch size
            scale_tensor: Scale factor for coordinates

        Returns:
            Processed targets (B, max_boxes, 5)
        """
        nl = targets.shape[0]

        # Use fixed max_boxes for compile compatibility
        # This avoids int(mx.max(...)) which breaks mx.compile
        max_boxes = self.max_boxes

        # Handle empty targets with fixed size output
        if nl == 0:
            return mx.zeros((batch_size, max_boxes, 5))

        # Get batch indices
        batch_idx = targets[:, 0].astype(mx.int32)

        # Count targets per batch (VECTORIZED)
        one_hot = (mx.expand_dims(batch_idx, 1) == mx.arange(batch_size)).astype(mx.int32)
        counts = mx.sum(one_hot, axis=0)  # (batch_size,)

        # Sort targets by batch index (stable sort preserves order within batch)
        sort_idx = mx.argsort(batch_idx)
        sorted_batch_idx = batch_idx[sort_idx]
        sorted_targets = targets[sort_idx]

        # Compute within-batch position using cumulative counts (VECTORIZED)
        cumsum_counts = mx.concatenate([mx.array([0]), mx.cumsum(counts[:-1], axis=0)])

        global_idx = mx.arange(nl)
        within_batch = global_idx - cumsum_counts[sorted_batch_idx]

        # Clamp within_batch to max_boxes-1 to avoid overflow
        within_batch = mx.minimum(within_batch, max_boxes - 1)

        # Build flat index for scatter
        flat_idx = (sorted_batch_idx * max_boxes + within_batch).astype(mx.int32)

        # Vectorized scatter using one-hot encoding + matmul
        # Create one-hot for positions: (nl,) -> (nl, batch_size * max_boxes)
        total_slots = batch_size * max_boxes
        position_one_hot = (mx.expand_dims(flat_idx, 1) == mx.arange(total_slots)).astype(
            mx.float32
        )
        # (total_slots, nl) @ (nl, 5) -> (total_slots, 5)
        target_data = sorted_targets[:, 1:]  # (nl, 5) - cls, x, y, w, h
        out = mx.transpose(position_one_hot, (1, 0)) @ target_data
        out = out.reshape(batch_size, max_boxes, 5)

        # Scale to xyxy format: out[..., 1:5] is xywh, convert and scale
        boxes = out[..., 1:5]
        xy = boxes[..., :2]
        wh = boxes[..., 2:] / 2
        xyxy = mx.concatenate([xy - wh, xy + wh], axis=-1)
        xyxy = xyxy * scale_tensor

        # Create output with scaled boxes
        out = mx.concatenate([out[..., :1], xyxy], axis=-1)

        return out

    def bbox_decode(self, anchor_points: mx.array, pred_dist: mx.array) -> mx.array:
        """Decode predicted distributions to bounding boxes.

        Args:
            anchor_points: Anchor points (N, 2)
            pred_dist: Predicted distributions (B, N, 4*reg_max) or (B, N, 4)

        Returns:
            Decoded boxes (B, N, 4) in xyxy format
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape
            # Reshape and apply softmax
            pred_dist = pred_dist.reshape(b, a, 4, c // 4)
            pred_dist = mx.softmax(pred_dist, axis=-1)
            # Project to distances
            pred_dist = mx.sum(pred_dist * self.proj, axis=-1)

        # Convert distance (ltrb) to bbox (xyxy)
        lt = pred_dist[..., :2]
        rb = pred_dist[..., 2:]

        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb

        return mx.concatenate([x1y1, x2y2], axis=-1)

    def parse_output(self, preds):
        """Extract the feature tensor from raw model predictions.

        Args:
            preds: Raw model output, either a tuple (images, features) or a single tensor.

        Returns:
            Feature predictions dict, unwrapped from tuple if necessary.
        """
        return preds[1] if isinstance(preds, tuple) else preds

    def loss(
        self,
        preds: dict[str, mx.array],
        batch: dict[str, mx.array],
    ) -> tuple[mx.array, mx.array]:
        """Compute detection loss using the internal task-aligned assigner.

        Args:
            preds: Predictions dict with 'boxes', 'scores', 'feats' keys.
            batch: Batch dict with 'batch_idx', 'cls', 'bboxes' targets.

        Returns:
            Tuple of (total_loss scaled by batch size, per-component loss tensor [box, cls, dfl]).
        """
        return self._compute_loss(preds, batch, self.assigner)

    def __call__(
        self,
        preds: dict[str, mx.array],
        batch: dict[str, mx.array],
        assigner: Any | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Calculate detection loss.

        Args:
            preds: Predictions dict with 'boxes', 'scores', 'feats'
                   OR end2end format with 'one2many', 'one2one' keys
            batch: Batch dict with 'batch_idx', 'cls', 'bboxes'
            assigner: TaskAlignedAssigner instance (optional, uses internal if None)

        Returns:
            Tuple of (total_loss, loss_items)
        """
        if assigner is None:
            assigner = self.assigner
        return self._compute_loss(preds, batch, assigner)

    def _compute_loss(
        self,
        preds: dict[str, mx.array],
        batch: dict[str, mx.array],
        assigner: Any,
    ) -> tuple[mx.array, mx.array]:
        """Core loss computation.

        Args:
            preds: Predictions dict with 'boxes', 'scores', 'feats'
                   OR end2end format with 'one2many', 'one2one' keys
            batch: Batch dict with 'batch_idx', 'cls', 'bboxes'
            assigner: TaskAlignedAssigner instance

        Returns:
            Tuple of (total_loss, loss_items)
        """
        # Handle end2end format - extract one2many predictions for training
        if "one2many" in preds:
            preds = preds["one2many"]

        loss = mx.zeros(3)  # box, cls, dfl

        # Parse predictions - check if transposition is needed
        boxes = preds["boxes"]
        scores = preds["scores"]

        # Check shape - if (B, anchors, C) then already transposed, else transpose
        if len(boxes.shape) == 3 and boxes.shape[-1] == 4 * self.reg_max:
            # Already (B, N, 4*reg_max)
            pred_distri = boxes
        else:
            # (B, C, N) -> (B, N, C)
            pred_distri = mx.transpose(boxes, (0, 2, 1))

        if len(scores.shape) == 3 and scores.shape[-1] == self.nc:
            # Already (B, N, nc)
            pred_scores = scores
        else:
            # (B, C, N) -> (B, N, C)
            pred_scores = mx.transpose(scores, (0, 2, 1))

        # Generate anchors
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        batch_size = pred_scores.shape[0]
        dtype = pred_scores.dtype
        imgsz = mx.array(preds["feats"][0].shape[1:3], dtype=dtype) * float(self.stride[0])

        # Preprocess targets
        targets = mx.concatenate(
            [batch["batch_idx"].reshape(-1, 1), batch["cls"].reshape(-1, 1), batch["bboxes"]],
            axis=1,
        )
        targets = self.preprocess(targets, batch_size, imgsz[mx.array([1, 0, 1, 0])])
        gt_labels, gt_bboxes = targets[..., :1], targets[..., 1:5]
        mask_gt = mx.sum(gt_bboxes, axis=2, keepdims=True) > 0

        # Decode predictions
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Task-aligned assignment
        # IMPORTANT: Stop gradient on inputs to TAL - target assignment should not
        # propagate gradients back through the assignment logic (same as PyTorch detach)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
            mx.stop_gradient(mx.sigmoid(pred_scores)),
            mx.stop_gradient((pred_bboxes * stride_tensor).astype(gt_bboxes.dtype)),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # Also stop gradient on targets - they are constants for loss computation
        target_bboxes = mx.stop_gradient(target_bboxes)
        target_scores = mx.stop_gradient(target_scores)
        fg_mask = mx.stop_gradient(fg_mask)
        target_gt_idx = mx.stop_gradient(target_gt_idx)

        # Store assignment results for subclass access (v8SegmentationLoss)
        self._last_assign = (fg_mask, target_gt_idx, target_bboxes, target_scores)
        # Store anchors/strides for subclass access (v8PoseLoss keypoint decode).
        # anchor_points are in grid units; stride_tensor is (N, 1).
        self._last_anchors = (anchor_points, stride_tensor)

        target_scores_sum = mx.maximum(mx.sum(target_scores), 1.0)

        # Classification loss (BCE)
        loss_cls = losses.binary_cross_entropy(
            pred_scores, target_scores, with_logits=True, reduction="sum"
        )
        loss_cls_norm = loss_cls / target_scores_sum

        # Box and DFL loss — always compute (masking inside BboxLoss
        # ensures zero loss when no foreground). Removing the data-dependent
        # branch makes this function mx.compile-compatible.
        loss_iou, loss_dfl = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )

        # Apply gains and construct loss array
        loss_box = loss_iou * self.box_gain
        loss_cls_final = loss_cls_norm * self.cls_gain
        loss_dfl_final = loss_dfl * self.dfl_gain

        loss = mx.stack([loss_box, loss_cls_final, loss_dfl_final])

        return mx.sum(loss) * batch_size, loss


class v8SegmentationLoss(v8DetectionLoss):
    """Segmentation Loss for YOLO26.

    Reference: ultralytics/utils/loss.py v8SegmentationLoss

    Combines detection loss (box, cls, dfl) with per-instance mask loss
    and optional semantic segmentation auxiliary loss from Proto26.
    """

    MAX_FG_PER_IMAGE = 200

    def __init__(
        self,
        model: Any,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
    ):
        """Initialize segmentation loss.

        Args:
            model: YOLO model instance (for accessing head parameters like nc, reg_max, stride).
            tal_topk: Number of top-k candidates for the task-aligned assigner.
            tal_topk2: Secondary top-k for filtering (optional, defaults to tal_topk).
        """
        super().__init__(model, tal_topk, tal_topk2)
        self.overlap = getattr(model.args, "overlap_mask", True) if hasattr(model, "args") else True
        m = model.model[-1] if hasattr(model, "model") else model
        self.nm = m.nm if hasattr(m, "nm") else 32
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def parse_output(self, preds):
        """Extract the predictions dict from segmentation model output.

        Segmentation models return (preds_dict, proto_raw) during training.
        The preds_dict already contains "proto" and "semseg" keys.
        """
        if isinstance(preds, tuple):
            return preds[0]
        return preds

    @staticmethod
    def single_mask_loss(
        gt_mask: mx.array,
        pred: mx.array,
        proto: mx.array,
        xyxy: mx.array,
        area: mx.array,
    ) -> mx.array:
        """Compute mask loss for foreground anchors in a single image.

        Reference: ultralytics v8SegmentationLoss.single_mask_loss

        Args:
            gt_mask: Ground truth masks (K, H, W).
            pred: Predicted mask coefficients for foreground anchors (K, nm).
            proto: Prototype masks (H, W, nm) in NHWC.
            xyxy: GT boxes in xyxy at mask resolution (K, 4).
            area: Bounding box areas at mask resolution (K,).

        Returns:
            Summed mask loss over all instances.
        """
        pred_mask = mx.einsum("kn,hwn->khw", pred, proto)  # (K, H, W)

        loss = losses.binary_cross_entropy(pred_mask, gt_mask, with_logits=True, reduction="none")
        loss = crop_mask(loss, xyxy)  # zero loss outside boxes
        loss_per_instance = mx.sum(loss, axis=(1, 2)) / (area + 1e-7)
        return mx.sum(loss_per_instance)

    def loss(
        self,
        preds: dict[str, mx.array],
        batch: dict[str, mx.array],
    ) -> tuple[mx.array, mx.array]:
        """Compute combined detection and segmentation loss.

        Overrides v8DetectionLoss.loss to add mask loss. Called from E2ELoss
        for the one2many branch (with mask data) and one2one branch (without).

        Args:
            preds: Predictions dict. For one2many: includes "mask_coeff", "proto",
                   and optionally "semseg". For one2one: only detection keys.
            batch: Batch dict with "batch_idx", "cls", "bboxes", and optionally "masks".

        Returns:
            Tuple of (total_loss, loss_items) where loss_items has 5 components:
            [box, seg, cls, dfl, semseg].
        """
        mask_coeff = preds.get("mask_coeff")
        proto = preds.get("proto")
        semseg = preds.get("semseg")

        det_preds = {k: v for k, v in preds.items() if k not in ("mask_coeff", "proto", "semseg")}

        det_total, det_items = self._compute_loss(det_preds, batch, self.assigner)

        has_mask_data = mask_coeff is not None and proto is not None and "masks" in batch

        loss = mx.zeros(5)  # box, seg, cls, dfl, semseg
        loss = loss.at[0].add(det_items[0])
        loss = loss.at[2].add(det_items[1])
        loss = loss.at[3].add(det_items[2])

        if has_mask_data:
            fg_mask, target_gt_idx, target_bboxes, _ = self._last_assign
            # imgsz in pixel space = P3_feature_hw * stride[0]. Matches PT:
            # ``imgsz = torch.tensor(preds["feats"][0].shape[2:]) * self.stride[0]``.
            # Must NOT be derived from proto.shape because Proto26 upsamples the
            # P3 feature map (proto is 2× finer than P3), so using proto would
            # double imgsz and halve the per-instance box in mask coordinates —
            # that bug causes mask mAP to degrade during training while leaving
            # box mAP intact.
            p3_hw = det_preds["feats"][0].shape[1:3]
            imgsz_px = float(self.stride[0]) * float(p3_hw[0])
            seg_loss = self._calculate_segmentation_loss(
                fg_mask, target_gt_idx, target_bboxes, mask_coeff, proto, batch, imgsz_px
            )
            loss = loss.at[1].add(seg_loss * self.box_gain)

            if semseg is not None and "masks" in batch:
                sem_loss = self._calculate_semseg_loss(semseg, batch)
                loss = loss.at[4].add(sem_loss * self.box_gain)

        batch_size = det_preds["boxes"].shape[0]
        return mx.sum(loss) * batch_size, loss

    def _calculate_segmentation_loss(
        self,
        fg_mask: mx.array,
        target_gt_idx: mx.array,
        target_bboxes: mx.array,
        mask_coeff: mx.array,
        proto: mx.array,
        batch: dict[str, mx.array],
        imgsz_px: float,
    ) -> mx.array:
        """Calculate per-instance segmentation loss.

        Uses fixed-size top-K foreground selection per image for MLX
        compatibility (no data-dependent shapes).

        Args:
            fg_mask: (B, N) foreground boolean mask from TAL.
            target_gt_idx: (B, N) assigned GT index per anchor.
            target_bboxes: (B, N, 4) assigned GT boxes in xyxy pixel coords.
            mask_coeff: (B, total_anchors, nm) mask coefficients.
            proto: (B, H_proto, W_proto, nm) prototypes in NHWC.
            batch: Batch dict with "masks" key.
            imgsz_px: Image size in pixel space used to normalise GT boxes
                before converting into mask coordinates (matches PT's
                ``imgsz = feats[0].shape[2:] * stride[0]``).

        Returns:
            Normalized segmentation loss scalar.
        """
        masks_gt = batch["masks"]  # (B, H_mask, W_mask) overlap map
        batch_size = fg_mask.shape[0]
        _, mh, mw, nm = proto.shape

        total_loss = mx.array(0.0)
        total_fg = mx.array(0.0)

        for b in range(batch_size):
            fg_b = fg_mask[b].astype(mx.float32)  # (N,)
            n_fg = mx.sum(fg_b)

            sort_idx = mx.argsort(-fg_b)
            k = min(self.MAX_FG_PER_IMAGE, fg_b.shape[0])
            top_idx = sort_idx[:k]

            mc_fg = mask_coeff[b][top_idx]  # (K, nm)
            gt_idx_fg = target_gt_idx[b][top_idx]  # (K,)
            valid = fg_b[top_idx]  # (K,) 0 or 1
            tb_fg = target_bboxes[b][top_idx]  # (K, 4) xyxy pixel

            # Scale boxes from pixel space to mask resolution. ``imgsz_px`` is
            # the P3-derived pixel size (not proto-derived — see loss()).
            proto_b = proto[b]  # (mh, mw, nm)
            scale = mx.array([mw, mh, mw, mh], dtype=mx.float32)
            box_scale = scale / imgsz_px
            tb_mask = tb_fg * box_scale

            # Build GT masks from overlap map for each selected anchor
            gt_overlap = masks_gt[b]  # (H_mask, W_mask)
            gt_idx_expanded = gt_idx_fg[:, None, None] + 1  # (K, 1, 1)
            gt_masks = (gt_overlap[None, :, :] == gt_idx_expanded).astype(mx.float32)  # (K, mh, mw)

            # Compute mask loss
            pred_masks = mx.einsum("kn,hwn->khw", mc_fg, proto_b)  # (K, mh, mw)
            loss = losses.binary_cross_entropy(
                pred_masks, gt_masks, with_logits=True, reduction="none"
            )
            loss = crop_mask(loss, tb_mask)

            box_w = tb_mask[:, 2] - tb_mask[:, 0]
            box_h = tb_mask[:, 3] - tb_mask[:, 1]
            area = mx.maximum(box_w * box_h, mx.array(1.0))
            loss_per = mx.sum(loss, axis=(1, 2)) / area
            total_loss = total_loss + mx.sum(loss_per * valid)
            total_fg = total_fg + n_fg

        return total_loss / mx.maximum(total_fg, mx.array(1.0))

    def _calculate_semseg_loss(
        self,
        pred_semseg: mx.array,
        batch: dict[str, mx.array],
    ) -> mx.array:
        """Calculate semantic segmentation auxiliary loss.

        Reference: ultralytics v8SegmentationLoss.__call__ semseg branch

        Uses pre-computed ``sem_masks`` (per-pixel class index) from the
        data loader, converts to one-hot, zeros out background, and
        computes BCEDice loss against ``pred_semseg``.

        Args:
            pred_semseg: (B, H_feat, W_feat, nc) predicted class logits (NHWC).
            batch: Batch dict with "sem_masks" (B, H_mask, W_mask) per-pixel
                   class indices and "masks" (B, H_mask, W_mask) overlap map.

        Returns:
            Semantic segmentation loss scalar.
        """
        if "sem_masks" not in batch:
            return mx.array(0.0)

        sem_gt = batch["sem_masks"]  # (B, H_mask, W_mask) int class indices
        masks_gt = batch["masks"]  # (B, H_mask, W_mask) overlap map
        bs, sh, sw, nc = pred_semseg.shape

        # Downsample sem_gt and masks_gt to pred resolution if sizes differ
        _, mh, mw = sem_gt.shape
        if mh != sh or mw != sw:
            r_h = max(1, mh // sh)
            r_w = max(1, mw // sw)
            sem_ds = sem_gt[:, ::r_h, ::r_w][:, :sh, :sw]
            masks_ds = masks_gt[:, ::r_h, ::r_w][:, :sh, :sw]
        else:
            sem_ds = sem_gt
            masks_ds = masks_gt

        # One-hot encode: (B, H, W) → (B, H, W, nc) → (B, nc, H, W) for BCEDice
        sem_clamped = mx.clip(sem_ds, 0, nc - 1).astype(mx.int32)
        sem_onehot = mx.zeros((bs, sh, sw, nc))
        for b in range(bs):
            oh = mx.zeros((sh * sw, nc))
            flat_cls = sem_clamped[b].reshape(-1)
            idx = mx.arange(sh * sw)
            oh = oh.at[idx, flat_cls].add(mx.ones(sh * sw))
            sem_onehot = sem_onehot.at[b].add(oh.reshape(sh, sw, nc))

        sem_onehot = mx.clip(sem_onehot, 0, 1)

        # Zero out background (where overlap map == 0)
        bg_mask = (masks_ds == 0).astype(mx.float32)  # (B, H, W)
        sem_onehot = sem_onehot * (1.0 - bg_mask[:, :, :, None])

        # Transpose to NCHW for BCEDiceLoss (expects sum over axes 2, 3)
        pred_nchw = mx.transpose(pred_semseg, (0, 3, 1, 2))  # (B, nc, H, W)
        target_nchw = mx.transpose(sem_onehot, (0, 3, 1, 2))  # (B, nc, H, W)

        return self.bcedice_loss(pred_nchw, target_nchw)


class v8PoseLoss(v8DetectionLoss):
    """Pose Estimation Loss for YOLO26 (flow-based Pose26 / RLE).

    Reference: ultralytics/utils/loss.py PoseLoss26 (and v8PoseLoss).

    Computes the full 6-component pose loss
    ``[box, pose (OKS), kobj (visibility BCE), cls, dfl, rle]`` matching the
    released ``Pose26`` head. The ``rle`` term evaluates the residual
    log-likelihood of the keypoint error under the ported MLX ``RealNVP``
    flow model, training the ``cv4_sigma`` heads alongside the keypoint and
    box branches. When the head has no ``flow_model`` (plain ``Pose``), the
    loss collapses to the 5-component OKS form.

    Combined with ``v8DetectionLoss`` via ``E2ELoss`` exactly like
    ``v8SegmentationLoss``: keypoint outputs are forwarded to the one2many
    branch, and the assignment captured in ``self._last_assign`` /
    ``self._last_anchors`` drives keypoint target selection.
    """

    # Fixed foreground budget per image for compile-friendly selection
    # (mirrors v8SegmentationLoss.MAX_FG_PER_IMAGE).
    MAX_FG_PER_IMAGE = 200

    # COCO OKS sigmas (Ultralytics OKS_SIGMA = np.array([...]) / 10).
    _COCO_OKS_SIGMA = (
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    )
    # Ultralytics RLE_WEIGHT — per-keypoint target weights for the RLE term.
    _COCO_RLE_WEIGHT = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.2,
        1.2,
        1.5,
        1.5,
        1.0,
        1.0,
        1.2,
        1.2,
        1.5,
        1.5,
    )

    def __init__(
        self,
        model: Any,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
    ):
        """Initialize pose loss.

        Args:
            model: YOLO model instance (for accessing head parameters like nc, reg_max, stride).
            tal_topk: Number of top-k candidates for the task-aligned assigner.
            tal_topk2: Secondary top-k for filtering (optional, defaults to tal_topk).
        """
        super().__init__(model, tal_topk, tal_topk2)

        m = model.model[-1] if hasattr(model, "model") else model
        self.kpt_shape = m.kpt_shape if hasattr(m, "kpt_shape") else (17, 3)
        self.nkpt = self.kpt_shape[0]

        # Compare against a tuple so list/tuple YAML shapes both select the
        # COCO sigmas (regression guard for the [17, 3] vs (17, 3) bug).
        is_pose = tuple(self.kpt_shape) == (17, 3)

        if is_pose:
            sigmas = mx.array(self._COCO_OKS_SIGMA)
            rle_weight = mx.array(self._COCO_RLE_WEIGHT)
        else:
            sigmas = mx.ones(self.nkpt) / self.nkpt
            rle_weight = mx.ones(self.nkpt)

        self.sigmas = sigmas
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)
        self.rle_weight = rle_weight

        # Flow model (RealNVP) for the residual log-likelihood term. Present
        # only on the Pose26 head; when absent the loss is 5-component.
        self.flow_model = m.flow_model if hasattr(m, "flow_model") else None

        # Pose-specific gains.
        self.pose_gain = self.hyp.get("pose", 12.0) if isinstance(self.hyp, dict) else 12.0
        self.kobj_gain = self.hyp.get("kobj", 1.0) if isinstance(self.hyp, dict) else 1.0
        self.rle_gain = self.hyp.get("rle", 1.0) if isinstance(self.hyp, dict) else 1.0

    @staticmethod
    def kpts_decode(anchor_points: mx.array, pred_kpts: mx.array) -> mx.array:
        """Decode predicted keypoints to grid space (Pose26 formula).

        Reference: ultralytics PoseLoss26.kpts_decode — adds the (grid-unit)
        anchor centre to the predicted offset with no ``×2``/``−0.5`` term.

        Args:
            anchor_points: Anchor centres in grid units (N, 2).
            pred_kpts: Predicted keypoints (B, N, K, D) with D >= 2.

        Returns:
            Keypoints decoded into per-anchor grid coordinates.
        """
        y = mx.array(pred_kpts)  # copy
        y = y.at[..., 0].add(anchor_points[:, 0].reshape(1, -1, 1))
        y = y.at[..., 1].add(anchor_points[:, 1].reshape(1, -1, 1))
        return y

    def _preprocess_keypoints(
        self, keypoints: mx.array, batch_idx: mx.array, batch_size: int
    ) -> mx.array:
        """Scatter per-object keypoints into a fixed (B, max_boxes, K, 3) tensor.

        Mirrors ``v8DetectionLoss.preprocess`` exactly (same stable sort and
        cumulative-offset scatter on ``batch_idx``) so the object ordering
        matches the box targets — this is what makes the assigner's
        ``target_gt_idx`` valid for gathering keypoints.

        Args:
            keypoints: Per-object keypoints (N, K, 3) in normalized coords.
            batch_idx: Image index per object (N,); padding uses ``batch_size``.
            batch_size: Number of images in the batch.

        Returns:
            Keypoints tensor (B, max_boxes, K, 3) aligned with the box targets.
        """
        nl = keypoints.shape[0]
        kdim = keypoints.shape[1] * keypoints.shape[2]
        max_boxes = self.max_boxes
        if nl == 0:
            return mx.zeros((batch_size, max_boxes, self.nkpt, 3))

        batch_idx = batch_idx.astype(mx.int32)
        one_hot = (mx.expand_dims(batch_idx, 1) == mx.arange(batch_size)).astype(mx.int32)
        counts = mx.sum(one_hot, axis=0)

        sort_idx = mx.argsort(batch_idx)
        sorted_batch_idx = batch_idx[sort_idx]
        sorted_kpts = keypoints[sort_idx].reshape(nl, kdim)

        cumsum_counts = mx.concatenate([mx.array([0]), mx.cumsum(counts[:-1], axis=0)])
        global_idx = mx.arange(nl)
        within_batch = global_idx - cumsum_counts[sorted_batch_idx]
        within_batch = mx.minimum(within_batch, max_boxes - 1)

        flat_idx = (sorted_batch_idx * max_boxes + within_batch).astype(mx.int32)
        total_slots = batch_size * max_boxes
        position_one_hot = (mx.expand_dims(flat_idx, 1) == mx.arange(total_slots)).astype(
            mx.float32
        )
        out = mx.transpose(position_one_hot, (1, 0)) @ sorted_kpts
        return out.reshape(batch_size, max_boxes, self.nkpt, 3)

    def loss(
        self,
        preds: dict[str, mx.array],
        batch: dict[str, mx.array],
    ) -> tuple[mx.array, mx.array]:
        """Compute combined detection and keypoint loss.

        Called from ``E2ELoss`` for the one2many branch (with keypoint data)
        and the one2one branch (detection only).

        Args:
            preds: Predictions dict. For one2many: includes "keypoints" and
                "kpts_sigma". For one2one: only detection keys.
            batch: Batch dict with "batch_idx", "cls", "bboxes", and (for
                pose) "keypoints".

        Returns:
            Tuple of (total_loss, loss_items) with components
            ``[box, pose, kobj, cls, dfl, rle]`` (rle omitted if no flow model).
        """
        kpts = preds.get("keypoints")
        kpts_sigma = preds.get("kpts_sigma")
        det_preds = {k: v for k, v in preds.items() if k not in ("keypoints", "kpts_sigma")}

        _, det_items = self._compute_loss(det_preds, batch, self.assigner)

        n_comp = 6 if self.flow_model is not None else 5
        loss = mx.zeros(n_comp)
        loss = loss.at[0].add(det_items[0])  # box
        loss = loss.at[3].add(det_items[1])  # cls
        loss = loss.at[4].add(det_items[2])  # dfl

        has_kpts = kpts is not None and "keypoints" in batch
        if has_kpts:
            fg_mask, target_gt_idx, target_bboxes, _ = self._last_assign
            anchor_points, stride_tensor = self._last_anchors
            feats = det_preds["feats"]
            imgsz_h = float(self.stride[0]) * feats[0].shape[1]
            imgsz_w = float(self.stride[0]) * feats[0].shape[2]

            kpt_loss, kobj_loss, rle_loss = self._calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                target_bboxes,
                anchor_points,
                stride_tensor,
                kpts,
                kpts_sigma,
                batch,
                imgsz_h,
                imgsz_w,
            )
            loss = loss.at[1].add(kpt_loss * self.pose_gain)
            loss = loss.at[2].add(kobj_loss * self.kobj_gain)
            if n_comp == 6:
                loss = loss.at[5].add(rle_loss * self.rle_gain)

        batch_size = det_preds["boxes"].shape[0]
        return mx.sum(loss) * batch_size, loss

    def __call__(
        self,
        preds: dict[str, mx.array],
        batch: dict[str, mx.array],
        assigner: Any | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Calculate pose loss (delegates to ``loss`` with the internal assigner)."""
        return self.loss(preds, batch)

    def _calculate_keypoints_loss(
        self,
        fg_mask: mx.array,
        target_gt_idx: mx.array,
        target_bboxes: mx.array,
        anchor_points: mx.array,
        stride_tensor: mx.array,
        pred_kpts: mx.array,
        pred_sigma: mx.array | None,
        batch: dict[str, mx.array],
        imgsz_h: float,
        imgsz_w: float,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Compute OKS, visibility BCE and RLE keypoint losses.

        Uses a fixed-size top-K foreground selection per image (same technique
        as ``_calculate_segmentation_loss``) so shapes stay static. All
        coordinates are converted into per-anchor grid units, matching
        ``PoseLoss26``.

        Args:
            fg_mask: (B, A) foreground mask from TAL.
            target_gt_idx: (B, A) assigned GT object index per anchor.
            target_bboxes: (B, A, 4) assigned GT boxes in xyxy pixel coords.
            anchor_points: (A, 2) anchor centres in grid units.
            stride_tensor: (A, 1) per-anchor stride.
            pred_kpts: (B, A, K*3) raw keypoint predictions.
            pred_sigma: (B, A, K*2) raw sigma predictions, or None.
            batch: Batch dict with "keypoints" (N, K, 3) and "batch_idx" (N,).
            imgsz_h: Image height in pixels (P3_h * stride[0]).
            imgsz_w: Image width in pixels (P3_w * stride[0]).

        Returns:
            Tuple of (oks_loss, kobj_loss, rle_loss) scalars.
        """
        bs = pred_kpts.shape[0]
        kp = self.nkpt

        # Reshape predictions to (B, A, K, D) and decode to grid space.
        pk = pred_kpts.reshape(bs, pred_kpts.shape[1], kp, 3)
        if self.flow_model is not None and pred_sigma is not None:
            ps = pred_sigma.reshape(bs, pred_sigma.shape[1], kp, 2)
            pk = mx.concatenate([pk, ps], axis=-1)  # (B, A, K, 5)
        pk = self.kpts_decode(anchor_points, pk)

        # Build per-object GT keypoints aligned with the box targets, scale to
        # pixel space (matches PT: keypoints[..., 0] *= imgsz[1]).
        batched_kpts = self._preprocess_keypoints(batch["keypoints"], batch["batch_idx"], bs)
        batched_kpts = batched_kpts.at[..., 0].multiply(imgsz_w)
        batched_kpts = batched_kpts.at[..., 1].multiply(imgsz_h)

        sigma_sq = mx.power(2.0 * self.sigmas, 2)  # (K,)
        k_fg = min(self.MAX_FG_PER_IMAGE, fg_mask.shape[1])

        oks_num = mx.array(0.0)
        kobj_num = mx.array(0.0)
        rle_num = mx.array(0.0)
        fg_total = mx.array(0.0)
        vis_total = mx.array(0.0)

        for b in range(bs):
            fg_b = fg_mask[b].astype(mx.float32)  # (A,)
            sort_idx = mx.argsort(-fg_b)
            top_idx = sort_idx[:k_fg]
            valid = fg_b[top_idx]  # (K_fg,) 1 for real fg anchors

            gt_idx = target_gt_idx[b][top_idx]  # (K_fg,)
            stride_sel = stride_tensor[top_idx]  # (K_fg, 1)

            pkpt = pk[b][top_idx]  # (K_fg, K, D) decoded grid
            gkpt = batched_kpts[b][gt_idx]  # (K_fg, K, 3) pixel
            # Pixel -> per-anchor grid units.
            gkpt = gkpt.at[..., :2].multiply(mx.expand_dims(1.0 / stride_sel, axis=-1))

            tb = target_bboxes[b][top_idx] / stride_sel  # (K_fg, 4) grid xyxy
            box_w = tb[:, 2] - tb[:, 0]
            box_h = tb[:, 3] - tb[:, 1]
            area = box_w * box_h  # (K_fg,)

            gt_vis = (gkpt[..., 2] != 0).astype(mx.float32)  # (K_fg, K)
            kpt_mask = gt_vis * mx.expand_dims(valid, axis=-1)  # visible AND fg

            # OKS location loss (matches KeypointLoss, summed for batch mean).
            d = mx.power(pkpt[..., 0] - gkpt[..., 0], 2) + mx.power(pkpt[..., 1] - gkpt[..., 1], 2)
            kpt_loss_factor = kp / (mx.sum(kpt_mask, axis=1) + 1e-9)  # (K_fg,)
            e = d / (sigma_sq * (mx.expand_dims(area, axis=-1) + 1e-9) * 2)
            oks_term = mx.expand_dims(kpt_loss_factor, axis=-1) * ((1 - mx.exp(-e)) * kpt_mask)
            oks_num = oks_num + mx.sum(oks_term)

            # Visibility (kobj) BCE on the predicted visibility channel.
            bce = losses.binary_cross_entropy(
                pkpt[..., 2], gt_vis, with_logits=True, reduction="none"
            )
            kobj_num = kobj_num + mx.sum(bce * mx.expand_dims(valid, axis=-1))

            fg_total = fg_total + mx.sum(valid)

            # RLE residual log-likelihood (only when the flow model exists).
            if self.flow_model is not None and pkpt.shape[-1] >= 5:
                psig = mx.sigmoid(pkpt[..., 3:5])  # (K_fg, K, 2)
                error = (pkpt[..., 0:2] - gkpt[..., 0:2]) / (psig + 1e-9)
                error = mx.clip(error, -100.0, 100.0)
                vmask = kpt_mask  # (K_fg, K)
                error = error * mx.expand_dims(vmask, axis=-1)  # zero invalid

                flat_err = error.reshape(-1, 2)
                flat_sigma = psig.reshape(-1, 2)
                flat_vmask = vmask.reshape(-1)  # (K_fg*K,)
                tw = mx.broadcast_to(
                    mx.expand_dims(self.rle_weight, axis=0), (top_idx.shape[0], kp)
                ).reshape(-1)  # (K_fg*K,)

                log_phi = self.flow_model.log_prob(flat_err)  # (M,)
                log_sigma = mx.log(flat_sigma)  # (M, 2)
                rle = log_sigma - mx.expand_dims(log_phi, axis=-1)
                rle = rle + mx.log(flat_sigma * 2.0) + mx.abs(flat_err)
                rle = rle * mx.expand_dims(tw, axis=-1)
                rle = rle * mx.expand_dims(flat_vmask, axis=-1)  # zero invalid
                rle_num = rle_num + mx.sum(rle)
                vis_total = vis_total + mx.sum(flat_vmask)

        denom = mx.maximum(fg_total, mx.array(1.0)) * kp
        oks_loss = oks_num / denom
        kobj_loss = kobj_num / denom
        rle_loss = mx.maximum(rle_num / mx.maximum(vis_total, mx.array(1.0)), mx.array(0.0))

        return oks_loss, kobj_loss, rle_loss


class v8OBBLoss(v8DetectionLoss):
    """Oriented Bounding Box Loss for YOLO26.

    Reference: ultralytics/utils/loss.py v8OBBLoss
    """

    def __init__(
        self,
        model: Any,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
    ):
        """Initialize OBB loss.

        Args:
            model: YOLO model instance (for accessing head parameters like nc, reg_max, stride).
            tal_topk: Number of top-k candidates for the task-aligned assigner.
            tal_topk2: Secondary top-k for filtering (optional, defaults to tal_topk).
        """
        super().__init__(model, tal_topk, tal_topk2)
        # OBB-specific initialization would go here

    def __call__(
        self,
        preds: dict[str, mx.array],
        batch: dict[str, mx.array],
        assigner: Any,
    ) -> tuple[mx.array, mx.array]:
        """Calculate OBB loss.

        Args:
            preds: Predictions with 'boxes', 'scores', 'feats', 'angle'
            batch: Batch with rotated bounding boxes
            assigner: RotatedTaskAlignedAssigner

        Returns:
            Tuple of (total_loss, loss_items)
        """
        # For now, use detection loss as base
        # Full OBB loss would include angle loss
        return super().__call__(preds, batch, assigner)


class v8ClassificationLoss:
    """Classification Loss for YOLO26.

    Reference: ultralytics/utils/loss.py v8ClassificationLoss
    """

    def __call__(self, preds: mx.array, batch: dict[str, mx.array]) -> tuple[mx.array, mx.array]:
        """Compute classification loss.

        Args:
            preds: Predictions (can be tuple)
            batch: Batch with 'cls'

        Returns:
            Tuple of (loss, loss_detached)
        """
        if isinstance(preds, list | tuple):
            preds = preds[1]

        loss = losses.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss


class E2EDetectLoss:
    """End-to-End Detection Loss for YOLO26.

    Reference: ultralytics/utils/loss.py E2EDetectLoss

    Uses one-to-many (topk=10) and one-to-one (topk=1) assignment
    for NMS-free end-to-end detection training.
    """

    def __init__(self, model: Any):
        """Initialize E2EDetectLoss with one-to-many and one-to-one losses.

        Args:
            model: Detection model with detect head
        """
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, mx.array]) -> tuple[mx.array, mx.array]:
        """Calculate combined one-to-many and one-to-one losses.

        Args:
            preds: Model predictions with 'one2many' and 'one2one' keys
            batch: Batch data with targets

        Returns:
            Tuple of (total_loss, loss_items)
        """
        preds = preds[1] if isinstance(preds, tuple) else preds

        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)

        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)

        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ELoss:
    """End-to-End Loss for YOLO26 with adaptive weighting.

    Reference: ultralytics/utils/loss.py E2ELoss

    This is the primary loss class for YOLO26 end-to-end training.
    It combines one-to-many and one-to-one losses with decaying weights.

    The one-to-many loss weight decays from 0.8 to 0.1 during training,
    while one-to-one loss weight increases correspondingly.
    """

    def __init__(
        self,
        model: Any,
        loss_fn: type | None = None,  # Default to v8DetectionLoss
    ):
        """Initialize E2ELoss with adaptive weighting.

        Args:
            model: Detection model
            loss_fn: Loss function class (default: v8DetectionLoss)
        """
        if loss_fn is None:
            loss_fn = v8DetectionLoss

        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)

        # Training progress tracking
        self.updates = 0
        self.total = 1.0

        # Initial weights: o2m=0.8, o2o=0.2
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m

        # Final weight for o2m (decays to 0.1)
        self.final_o2m = 0.1

        # Epoch count for decay schedule — set by trainer via set_epochs().
        # PyTorch reads self.one2one.hyp.epochs. We default to 100 but
        # the trainer MUST call set_epochs() with the actual count.
        self._epochs = 100

    def set_epochs(self, epochs: int):
        """Set the total number of training epochs for decay schedule.

        Must be called by the trainer after creating this loss, so the
        one2many weight decays over the correct schedule.

        Args:
            epochs: Total training epochs
        """
        self._epochs = epochs

    def __call__(self, preds: Any, batch: dict[str, mx.array]) -> tuple[mx.array, mx.array]:
        """Calculate weighted one-to-many and one-to-one losses.

        Args:
            preds: Model predictions (dict with 'one2many' and 'one2one').
                   For segmentation, also contains 'mask_coeff', 'proto', 'semseg'.
            batch: Batch data with targets

        Returns:
            Tuple of (total_loss, loss_items)
        """
        preds = self.one2many.parse_output(preds)

        one2many = preds["one2many"]
        one2one = preds["one2one"]

        # Segmentation forwards shared mask data (proto/coeff/semseg) to the
        # one2many branch. Pose keypoints are emitted per-branch by the head
        # (one2many + one2one), so each branch's loss reads its own keypoints
        # and the one2one keypoint head trains — matching PyTorch.
        extra_keys = {k: preds[k] for k in ("mask_coeff", "proto", "semseg") if k in preds}
        if extra_keys:
            one2many = {**one2many, **extra_keys}

        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)

        # Weighted combination
        total_loss = loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o

        return total_loss, loss_one2one[1]

    def update(self) -> None:
        """Update the weights for one-to-many and one-to-one losses.

        Called at the end of each epoch to decay o2m weight.
        """
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x: int) -> float:
        """Calculate decayed weight for one-to-many loss.

        Linear decay from o2m_copy to final_o2m over epochs.

        Args:
            x: Current update step

        Returns:
            Decayed weight value
        """
        return (
            max(1 - x / max(self._epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m)
            + self.final_o2m
        )
