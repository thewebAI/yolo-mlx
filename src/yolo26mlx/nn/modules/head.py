# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 Detection Heads - Pure MLX Implementation

Detection, Segmentation, Pose, and OBB heads for YOLO26.
Reference: ultralytics/ultralytics/nn/modules/head.py

MLX specifics:
- NHWC format for all tensors
- Feature map shapes: (B, H, W, C) vs PyTorch (B, C, H, W)
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .block import DFL
from .conv import Conv, DWConv


class Sequential(nn.Module):
    """Sequential container that executes a list of layers in order."""

    def __init__(self, *modules):
        """Initialize Sequential with the given layers.

        Args:
            *modules: Variable number of nn.Module layers to execute sequentially.
        """
        super().__init__()
        self.layers = list(modules)

    def __call__(self, x):
        """Forward pass through all layers sequentially.

        Args:
            x: Input tensor passed through each layer in order.

        Returns:
            Output tensor after applying all layers sequentially.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class Detect(nn.Module):
    """YOLO Detection head.

    Reference: ultralytics Detect class
    Predicts bounding boxes and class probabilities at multiple scales.

    YOLO26 specifics:
    - reg_max=1 (no DFL distribution)
    - end2end=True for NMS-free detection
    - DWConv-based classification head
    """

    dynamic = False
    export = False
    format = None
    max_det = 300

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 1,
        end2end: bool = True,
        ch: tuple[int, ...] = (),
    ):
        """Initialize Detect head.

        Args:
            nc: Number of classes
            reg_max: DFL channels (1 for YOLO26 = direct regression)
            end2end: Use end-to-end NMS-free detection
            ch: Input channel sizes for each detection layer
        """
        super().__init__()
        self.nc = nc
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max
        self.no = nc + reg_max * 4  # outputs per anchor
        self.end2end = end2end

        # Default strides for YOLO26 (P3/8, P4/16, P5/32)
        self.stride = mx.array([8.0, 16.0, 32.0][: self.nl])

        # Box regression head channels
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))

        # Box regression convolutions (cv2) - stored as dict for MLX tracking
        self.cv2 = {
            f"layer{i}": Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1))
            for i, x in enumerate(ch)
        }

        # Classification convolutions (cv3) - DWConv-based for efficiency
        self.cv3 = {
            f"layer{i}": Sequential(
                DWConv(x, x, 3),
                Conv(x, c3, 1),
                DWConv(c3, c3, 3),
                Conv(c3, c3, 1),
                nn.Conv2d(c3, nc, 1),
            )
            for i, x in enumerate(ch)
        }

        # DFL layer (only used if reg_max > 1)
        if reg_max > 1:
            self.dfl = DFL(reg_max)
        else:
            self.dfl = None

        # End-to-end one-to-one detection heads
        if end2end:
            self.one2one_cv2 = {
                f"layer{i}": Sequential(
                    Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)
                )
                for i, x in enumerate(ch)
            }

            self.one2one_cv3 = {
                f"layer{i}": Sequential(
                    DWConv(x, x, 3),
                    Conv(x, c3, 1),
                    DWConv(c3, c3, 3),
                    Conv(c3, c3, 1),
                    nn.Conv2d(c3, nc, 1),
                )
                for i, x in enumerate(ch)
            }
        else:
            self.one2one_cv2 = None
            self.one2one_cv3 = None

    def _forward_head(self, x: list[mx.array], cv2: dict, cv3: dict) -> dict[str, mx.array]:
        """Forward through head layers.

        Args:
            x: List of multi-scale feature maps [(B, H_i, W_i, C_i), ...] from backbone/neck.
            cv2: Dict of box regression conv sequences keyed by "layer{i}".
            cv3: Dict of classification conv sequences keyed by "layer{i}".

        Returns:
            Dict with "boxes" (B, total_anchors, 4*reg_max), "scores" (B, total_anchors, nc), and "feats" (original feature maps).
        """
        bs = x[0].shape[0]

        boxes_list = []
        scores_list = []

        for i in range(self.nl):
            key = f"layer{i}"
            box = cv2[key](x[i])  # (B, H, W, 4*reg_max)
            cls = cv3[key](x[i])  # (B, H, W, nc)

            b, h, w, _ = box.shape
            boxes_list.append(mx.reshape(box, (bs, h * w, 4 * self.reg_max)))
            scores_list.append(mx.reshape(cls, (bs, h * w, self.nc)))

        boxes = mx.concatenate(boxes_list, axis=1)  # (B, total_anchors, 4*reg_max)
        scores = mx.concatenate(scores_list, axis=1)  # (B, total_anchors, nc)

        return {"boxes": boxes, "scores": scores, "feats": x}

    def __call__(self, x: list[mx.array]) -> Any:
        """Forward pass.

        Args:
            x: List of feature maps from backbone/neck

        Returns:
            Training: dict with boxes and scores
            Inference: processed detections
        """
        # One-to-many predictions
        preds = self._forward_head(x, self.cv2, self.cv3)

        if self.end2end:
            # One-to-one predictions (detached features)
            x_detach = [mx.stop_gradient(xi) for xi in x]
            one2one = self._forward_head(x_detach, self.one2one_cv2, self.one2one_cv3)
            preds = {"one2many": preds, "one2one": one2one}

        if self.training:
            return preds

        # Inference path
        return self._inference(preds)

    def _make_anchors(self, feats: list[mx.array], strides: mx.array) -> tuple[mx.array, mx.array]:
        """Generate anchor points and stride tensors for all feature levels.

        Args:
            feats: List of feature maps (used for shape)
            strides: Stride values for each level

        Returns:
            anchor_points: (total_anchors, 2) grid cell centers
            stride_tensor: (total_anchors, 1) strides
        """
        anchor_points = []
        stride_tensors = []

        for feat, stride in zip(feats, strides, strict=True):
            _, h, w, _ = feat.shape  # NHWC format

            # Create grid of anchor points (center of each cell)
            sx = mx.arange(w, dtype=mx.float32) + 0.5
            sy = mx.arange(h, dtype=mx.float32) + 0.5

            # Meshgrid
            grid_y, grid_x = mx.meshgrid(sy, sx, indexing="ij")
            anchor = mx.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)

            anchor_points.append(anchor)
            stride_tensors.append(mx.full((h * w, 1), float(stride)))

        return mx.concatenate(anchor_points, axis=0), mx.concatenate(stride_tensors, axis=0)

    def _dist2bbox(self, distance: mx.array, anchor_points: mx.array) -> mx.array:
        """Convert distance predictions to bounding boxes.

        Args:
            distance: (B, anchors, 4) distances from anchor to box edges (left, top, right, bottom)
            anchor_points: (anchors, 2) anchor point coordinates

        Returns:
            boxes: (B, anchors, 4) in xywh format
        """
        # distance: [left, top, right, bottom]
        lt = distance[..., :2]  # left, top
        rb = distance[..., 2:]  # right, bottom

        # anchor_points: (anchors, 2) -> expand to (1, anchors, 2)
        anchor_points = mx.expand_dims(anchor_points, axis=0)

        # x1y1 = anchor - left/top, x2y2 = anchor + right/bottom
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb

        # Convert to xywh
        xy = (x1y1 + x2y2) / 2  # center
        wh = x2y2 - x1y1  # width, height

        return mx.concatenate([xy, wh], axis=-1)

    max_det = 300  # Maximum detections per image for end2end mode

    def _inference(self, preds: dict) -> mx.array:
        """Process predictions for inference with proper box decoding.

        Args:
            preds: Predictions dict from _forward_head, or nested dict with "one2one"/"one2many" keys in end2end mode.

        Returns:
            Decoded detection tensor (B, max_det, 6) with [x, y, w, h, conf, class_idx] in end2end mode,
            or (B, anchors, 4+nc) concatenated boxes and scores otherwise.
        """
        if self.end2end:
            data = preds["one2one"]
        else:
            data = preds

        boxes = data["boxes"]  # (B, anchors, 4*reg_max)
        scores = data["scores"]  # (B, anchors, nc)
        feats = data["feats"]  # Feature maps for anchor generation

        # Apply DFL if needed (reg_max > 1)
        if self.dfl is not None:
            boxes = self.dfl(mx.transpose(boxes, (0, 2, 1)))  # (B, 4, anchors)
            boxes = mx.transpose(boxes, (0, 2, 1))  # (B, anchors, 4)

        # Generate anchor points and strides
        anchor_points, stride_tensor = self._make_anchors(feats, self.stride)

        # Decode boxes: dist2bbox then scale by stride
        # boxes are in distance format: [left, top, right, bottom]
        boxes = self._dist2bbox(boxes, anchor_points)  # (B, anchors, 4) in xywh

        # Scale by stride
        stride_tensor = mx.expand_dims(stride_tensor, axis=0)  # (1, anchors, 1)
        boxes = boxes * stride_tensor  # Scale to image coordinates

        # Sigmoid for class scores
        scores = mx.sigmoid(scores)

        if self.end2end:
            # Apply top-k selection (same as PyTorch postprocess)
            return self._postprocess_end2end(boxes, scores)
        else:
            # Return all predictions for NMS-based post-processing
            return mx.concatenate([boxes, scores], axis=-1)

    def _postprocess_end2end(self, boxes: mx.array, scores: mx.array) -> mx.array:
        """Post-process predictions for end-to-end mode using two-stage top-k selection.

        Matches PyTorch's Detect.postprocess() + get_topk_index() behavior.
        Two-stage top-k allows multiple detections from the same anchor
        (different classes), improving recall for overlapping objects.

        PyTorch reference (head.py L219-240):
            Stage 1: ori_index = scores.max(dim=-1)[0].topk(k)[1]  # top-k anchors
                      scores = scores.gather(dim=1, index=ori_index.repeat(1,1,nc))
            Stage 2: scores, index = scores.flatten(1).topk(k)  # top-k across k*nc
                      idx = ori_index[..., index // nc]  # map back to anchor
                      class_idx = index % nc

        Args:
            boxes: (B, anchors, 4) decoded boxes in xywh format
            scores: (B, anchors, nc) class probabilities after sigmoid

        Returns:
            (B, max_det, 6) tensor with [x, y, w, h, conf, class_idx]
        """
        batch_size = boxes.shape[0]
        num_anchors = boxes.shape[1]
        nc = scores.shape[2]
        k = min(self.max_det, num_anchors)

        results = []
        for b in range(batch_size):
            box = boxes[b]  # (anchors, 4)
            score = scores[b]  # (anchors, nc)

            # Stage 1: Find top-k anchors by their max class score
            max_scores = mx.max(score, axis=-1)  # (anchors,)
            ori_index = mx.argsort(-max_scores)[:k]  # (k,) top-k anchor indices

            # Gather ALL nc class scores for these k anchors
            top_scores = score[ori_index]  # (k, nc)

            # Stage 2: Flatten (k * nc) and find top-k across all (anchor, class) pairs
            # This allows the same anchor to appear multiple times with different classes
            flat_scores = top_scores.reshape(-1)  # (k * nc,)
            flat_top_idx = mx.argsort(-flat_scores)[:k]  # (k,) indices into flattened

            # Decode: which anchor and which class does each flat index correspond to?
            anchor_idx = flat_top_idx // nc  # index into ori_index (0..k-1)
            class_idx = flat_top_idx % nc  # class index (0..nc-1)

            # Map anchor_idx back to original anchor indices
            final_anchor_idx = ori_index[anchor_idx]  # (k,) original anchor indices

            # Gather final boxes, scores, and classes
            final_boxes = box[final_anchor_idx]  # (k, 4)
            final_scores = flat_scores[flat_top_idx]  # (k,)
            final_classes = class_idx.astype(mx.float32)  # (k,)

            # Combine: (k, 6) with [x, y, w, h, conf, class]
            result = mx.concatenate(
                [
                    final_boxes,
                    mx.expand_dims(final_scores, axis=-1),
                    mx.expand_dims(final_classes, axis=-1),
                ],
                axis=-1,
            )
            results.append(result)

        return mx.stack(results, axis=0)  # (B, k, 6)


class Segment(Detect):
    """YOLO Segmentation head.

    Reference: ultralytics Segment class
    Adds mask prototype prediction and mask coefficient outputs.
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        reg_max: int = 1,
        end2end: bool = True,
        ch: tuple[int, ...] = (),
    ):
        """Initialize Segment head.

        Args:
            nc: Number of classes
            nm: Number of mask prototypes
            npr: Prototype network channels
            reg_max: DFL channels
            end2end: End-to-end mode
            ch: Input channel sizes
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr

        # Mask coefficient head - stored as dict for MLX tracking
        c4 = max(ch[0] // 4, nm)
        self.cv4 = {
            f"layer{i}": Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, nm, 1))
            for i, x in enumerate(ch)
        }

        # Prototype generation network
        self.proto = Sequential(
            Conv(ch[0], npr, 3),
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Conv(npr, npr, 3),
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Conv(npr, nm, 1),
        )

    def __call__(self, x: list[mx.array]) -> tuple:
        """Forward pass for segmentation.

        Args:
            x: List of multi-scale feature maps [(B, H_i, W_i, C_i), ...] from backbone/neck.

        Returns:
            Tuple of (detection output with mask coefficients, mask prototypes (B, H_proto, W_proto, nm)).
        """
        # Generate prototypes from highest resolution feature
        p = self.proto(x[0])

        # Detection + mask coefficients
        bs = x[0].shape[0]
        mc_list = []  # mask coefficients

        for i in range(self.nl):
            mc = self.cv4[f"layer{i}"](x[i])
            b, h, w, _ = mc.shape
            mc_list.append(mx.reshape(mc, (bs, h * w, self.nm)))

        mc = mx.concatenate(mc_list, axis=1)  # (B, anchors, nm)

        # Get detection outputs
        preds = super().__call__(x)

        if self.training:
            preds["mask_coeff"] = mc
            return preds, p

        # Inference: append mask coefficients
        return mx.concatenate([preds, mc], axis=-1), p


class Pose(Detect):
    """YOLO Pose estimation head.

    Reference: ultralytics Pose class
    Adds keypoint prediction outputs.
    """

    def __init__(
        self,
        nc: int = 1,
        kpt_shape: tuple[int, int] = (17, 3),
        reg_max: int = 1,
        end2end: bool = True,
        ch: tuple[int, ...] = (),
    ):
        """Initialize Pose head.

        Args:
            nc: Number of classes (usually 1 for person)
            kpt_shape: (num_keypoints, dims) e.g. (17, 3) for COCO
            reg_max: DFL channels
            end2end: End-to-end mode
            ch: Input channel sizes
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]  # total keypoint values

        # Keypoint head - stored as dict for MLX tracking
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = {
            f"layer{i}": Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1))
            for i, x in enumerate(ch)
        }

    def __call__(self, x: list[mx.array]) -> Any:
        """Forward pass for pose estimation.

        Args:
            x: List of multi-scale feature maps [(B, H_i, W_i, C_i), ...] from backbone/neck.

        Returns:
            Training: dict with detection outputs and "keypoints" (B, anchors, nk).
            Inference: (B, anchors, 4+nc+nk) concatenated detections and keypoints.
        """
        bs = x[0].shape[0]
        kpt_list = []

        for i in range(self.nl):
            kpt = self.cv4[f"layer{i}"](x[i])
            b, h, w, _ = kpt.shape
            kpt_list.append(mx.reshape(kpt, (bs, h * w, self.nk)))

        kpts = mx.concatenate(kpt_list, axis=1)  # (B, anchors, nk)

        # Get detection outputs
        preds = super().__call__(x)

        if self.training:
            preds["keypoints"] = kpts
            return preds

        # Inference: append keypoints
        return mx.concatenate([preds, kpts], axis=-1)


class OBB(Detect):
    """YOLO Oriented Bounding Box head.

    Reference: ultralytics OBB class
    Adds rotation angle prediction for oriented boxes.
    """

    def __init__(
        self,
        nc: int = 80,
        ne: int = 1,
        reg_max: int = 1,
        end2end: bool = True,
        ch: tuple[int, ...] = (),
    ):
        """Initialize OBB head.

        Args:
            nc: Number of classes
            ne: Number of angle outputs (1 for rotation)
            reg_max: DFL channels
            end2end: End-to-end mode
            ch: Input channel sizes
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne

        # Angle prediction head - stored as dict for MLX tracking
        c4 = max(ch[0] // 4, ne)
        self.cv4 = {
            f"layer{i}": Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, ne, 1))
            for i, x in enumerate(ch)
        }

    def __call__(self, x: list[mx.array]) -> Any:
        """Forward pass for oriented bounding box detection.

        Args:
            x: List of multi-scale feature maps [(B, H_i, W_i, C_i), ...] from backbone/neck.

        Returns:
            Training: dict with detection outputs and "angles" (B, anchors, ne).
            Inference: (B, anchors, 4+nc+ne) concatenated detections and rotation angles.
        """
        bs = x[0].shape[0]
        angle_list = []

        for i in range(self.nl):
            angle = self.cv4[f"layer{i}"](x[i])
            b, h, w, _ = angle.shape
            angle_list.append(mx.reshape(angle, (bs, h * w, self.ne)))

        angles = mx.concatenate(angle_list, axis=1)  # (B, anchors, ne)

        # Get detection outputs
        preds = super().__call__(x)

        if self.training:
            preds["angles"] = angles
            return preds

        # Inference: append angles
        return mx.concatenate([preds, angles], axis=-1)
