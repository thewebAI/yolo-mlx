# Copyright (c) 2026 webAI, Inc.
"""Unit tests for segmentation modules and utilities.

Tests Proto, Proto26, Segment26, crop_mask, process_mask, polygon2mask,
mask_iou, and SegmentationMetrics with synthetic inputs.

Run:
    python -m pytest tests/test_segmentation.py -v
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX requires Apple Silicon")
nn = pytest.importorskip("mlx.nn", reason="MLX requires Apple Silicon")

from yolo26mlx.nn.modules.block import Proto, Proto26
from yolo26mlx.nn.modules.head import Segment, Segment26
from yolo26mlx.utils.metrics import SegmentationMetrics, mask_iou
from yolo26mlx.utils.ops import crop_mask, process_mask, process_mask_upsample

# ---------------------------------------------------------------------------
# Proto
# ---------------------------------------------------------------------------


class TestProto:
    """Verify Proto forward pass shapes and learned upsampling."""

    def test_output_shape(self):
        """Proto should double spatial dims via ConvTranspose2d."""
        c1, c_, c2 = 64, 128, 32
        proto = Proto(c1, c_, c2)
        mx.eval(proto.parameters())

        x = mx.random.normal((2, 20, 20, c1))
        out = proto(x)
        mx.eval(out)
        assert out.shape == (2, 40, 40, c2)

    def test_single_batch(self):
        proto = Proto(32, 64, 16)
        mx.eval(proto.parameters())

        x = mx.random.normal((1, 10, 10, 32))
        out = proto(x)
        mx.eval(out)
        assert out.shape == (1, 20, 20, 16)

    def test_has_learnable_upsample(self):
        """ConvTranspose2d should have weight and bias parameters."""
        proto = Proto(64, 128, 32)
        assert hasattr(proto.upsample, "weight")
        assert proto.upsample.weight is not None


# ---------------------------------------------------------------------------
# Proto26
# ---------------------------------------------------------------------------


class TestProto26:
    """Verify Proto26 multi-scale fusion and training/inference modes."""

    @pytest.fixture()
    def proto26(self):
        ch = (64, 128, 256)
        p = Proto26(ch, c_=64, c2=32, nc=80)
        mx.eval(p.parameters())
        return p

    def _make_features(self, batch: int = 2):
        """Create synthetic P3/P4/P5 feature maps."""
        p3 = mx.random.normal((batch, 80, 80, 64))
        p4 = mx.random.normal((batch, 40, 40, 128))
        p5 = mx.random.normal((batch, 20, 20, 256))
        return [p3, p4, p5]

    def test_inference_shape(self, proto26):
        """Inference should return prototypes only (not a tuple)."""
        proto26.eval()
        x = self._make_features()
        out = proto26(x)
        mx.eval(out)
        assert isinstance(out, mx.array)
        assert out.shape == (2, 160, 160, 32)

    def test_training_returns_tuple(self, proto26):
        """Training should return (prototypes, semseg_logits)."""
        proto26.train()
        x = self._make_features()
        out = proto26(x)
        assert isinstance(out, tuple)
        assert len(out) == 2
        protos, semseg = out
        mx.eval(protos, semseg)
        assert protos.shape == (2, 160, 160, 32)
        # semseg should have 80 channels (nc=80)
        assert semseg.shape[0] == 2
        assert semseg.shape[-1] == 80

    def test_fuse_strips_semseg(self, proto26):
        """After fuse(), semseg should be None and output should be array."""
        proto26.fuse()
        proto26.eval()
        x = self._make_features()
        out = proto26(x)
        mx.eval(out)
        assert isinstance(out, mx.array)
        assert proto26.semseg is None

    def test_feat_refine_count(self, proto26):
        """Should have one refinement layer per non-P3 scale."""
        assert len(proto26.feat_refine) == 2  # for P4 and P5


# ---------------------------------------------------------------------------
# Segment / Segment26
# ---------------------------------------------------------------------------


class TestSegment:
    """Verify Segment base class output formats."""

    @pytest.fixture()
    def segment(self):
        ch = (64, 128, 256)
        s = Segment(nc=80, nm=32, npr=64, reg_max=1, end2end=True, ch=ch)
        mx.eval(s.parameters())
        return s

    def _make_features(self, batch: int = 1):
        p3 = mx.random.normal((batch, 80, 80, 64))
        p4 = mx.random.normal((batch, 40, 40, 128))
        p5 = mx.random.normal((batch, 20, 20, 256))
        return [p3, p4, p5]

    def test_inference_returns_tuple(self, segment):
        """Inference should return (det_with_mc, protos) tuple."""
        segment.eval()
        x = self._make_features()
        out = segment(x)
        assert isinstance(out, tuple)
        assert len(out) == 2
        det_mc, protos = out
        mx.eval(det_mc, protos)
        # det_mc: (B, max_det, 6+nm)
        assert det_mc.ndim == 3
        assert det_mc.shape[0] == 1
        assert det_mc.shape[1] == 300  # max_det
        assert det_mc.shape[2] == 6 + 32  # 6 det + 32 mask coeffs

    def test_training_returns_dict(self, segment):
        """Training should return (preds_dict, protos) with mask_coeff key."""
        segment.train()
        x = self._make_features()
        out = segment(x)
        assert isinstance(out, tuple)
        preds, protos = out
        mx.eval(protos)
        assert isinstance(preds, dict)
        assert "mask_coeff" in preds
        mc = preds["mask_coeff"]
        mx.eval(mc)
        assert mc.shape[2] == 32  # nm


class TestSegment26:
    """Verify Segment26 head output formats and mask coefficients."""

    @pytest.fixture()
    def segment26(self):
        ch = (64, 128, 256)
        s = Segment26(nc=80, nm=32, npr=64, reg_max=1, end2end=True, ch=ch)
        mx.eval(s.parameters())
        return s

    def _make_features(self, batch: int = 1):
        p3 = mx.random.normal((batch, 80, 80, 64))
        p4 = mx.random.normal((batch, 40, 40, 128))
        p5 = mx.random.normal((batch, 20, 20, 256))
        return [p3, p4, p5]

    def test_inference_returns_tuple(self, segment26):
        """Inference: (det_with_mc, protos) tuple."""
        segment26.eval()
        x = self._make_features()
        out = segment26(x)
        assert isinstance(out, tuple)
        assert len(out) == 2
        det_mc, protos = out
        mx.eval(det_mc, protos)
        assert det_mc.shape == (1, 300, 6 + 32)
        assert protos.ndim == 4
        assert protos.shape[-1] == 32

    def test_training_returns_dict_with_mask_coeff(self, segment26):
        """Training: preds dict must have mask_coeff, proto, semseg keys."""
        segment26.train()
        x = self._make_features()
        out = segment26(x)
        assert isinstance(out, tuple)
        preds, proto_raw = out
        assert isinstance(preds, dict)
        assert "mask_coeff" in preds
        assert "proto" in preds
        assert "semseg" in preds
        mx.eval(preds["mask_coeff"], preds["proto"], preds["semseg"])
        assert preds["mask_coeff"].shape[2] == 32

    def test_uses_proto26(self, segment26):
        """Segment26 should use Proto26, not base Proto."""
        assert isinstance(segment26.proto, Proto26)

    def test_fuse_strips_heads(self, segment26):
        """fuse() should strip one2many heads and semseg."""
        segment26.fuse()
        assert segment26.cv2 is None
        assert segment26.cv3 is None
        assert segment26.cv4 is None
        assert segment26.proto.semseg is None

    def test_npr_scales_with_width(self):
        """npr should scale with channel width: n-scale has npr=64."""
        ch_n = (64, 128, 256)
        s = Segment26(nc=80, nm=32, npr=64, ch=ch_n)
        assert s.npr == 64
        assert s.proto.cv1.conv.weight.shape[-1] == 64  # c_ = npr for Proto26

        ch_l = (256, 512, 1024)
        s2 = Segment26(nc=80, nm=32, npr=256, ch=ch_l)
        assert s2.npr == 256


# ---------------------------------------------------------------------------
# crop_mask
# ---------------------------------------------------------------------------


class TestCropMask:
    """Verify crop_mask zeroes out regions outside bounding boxes."""

    def test_basic_crop(self):
        """Pixels outside box should be zero, inside should be preserved."""
        masks = mx.ones((1, 10, 10))
        boxes = mx.array([[2.0, 3.0, 7.0, 8.0]])  # x1, y1, x2, y2
        result = crop_mask(masks, boxes)
        mx.eval(result)
        result_np = np.array(result)

        # Inside the box
        assert result_np[0, 4, 3] == 1.0
        assert result_np[0, 5, 5] == 1.0

        # Outside the box
        assert result_np[0, 0, 0] == 0.0
        assert result_np[0, 9, 9] == 0.0
        assert result_np[0, 2, 0] == 0.0

    def test_full_image_box(self):
        """Box covering entire mask should preserve all values."""
        masks = mx.ones((1, 8, 8))
        boxes = mx.array([[0.0, 0.0, 8.0, 8.0]])
        result = crop_mask(masks, boxes)
        mx.eval(result)
        np.testing.assert_array_equal(np.array(result), np.ones((1, 8, 8)))

    def test_zero_area_box(self):
        """Zero-area box should zero out everything."""
        masks = mx.ones((1, 8, 8))
        boxes = mx.array([[3.0, 3.0, 3.0, 3.0]])
        result = crop_mask(masks, boxes)
        mx.eval(result)
        assert np.array(result).sum() == 0.0

    def test_multiple_masks(self):
        """Each mask should be cropped by its corresponding box."""
        masks = mx.ones((2, 8, 8))
        boxes = mx.array([[0.0, 0.0, 4.0, 4.0], [4.0, 4.0, 8.0, 8.0]])
        result = crop_mask(masks, boxes)
        mx.eval(result)
        r = np.array(result)

        # Mask 0: only top-left quadrant
        assert r[0, 0, 0] == 1.0
        assert r[0, 6, 6] == 0.0

        # Mask 1: only bottom-right quadrant
        assert r[1, 0, 0] == 0.0
        assert r[1, 6, 6] == 1.0


# ---------------------------------------------------------------------------
# process_mask
# ---------------------------------------------------------------------------


class TestProcessMask:
    """Verify mask generation from prototypes + coefficients."""

    def test_output_shape(self):
        """Output should be (N, mh, mw) binary masks at proto resolution."""
        mh, mw, c = 40, 40, 32
        protos = mx.random.normal((mh, mw, c))
        coeffs = mx.random.normal((3, c))
        boxes = mx.array(
            [
                [10.0, 10.0, 300.0, 300.0],
                [50.0, 50.0, 400.0, 400.0],
                [0.0, 0.0, 640.0, 480.0],
            ]
        )
        result = process_mask(protos, coeffs, boxes, (480, 640))
        mx.eval(result)
        assert result.shape == (3, mh, mw)

    def test_binary_output(self):
        """Result should be binary (0 or 1)."""
        protos = mx.random.normal((20, 20, 16))
        coeffs = mx.random.normal((2, 16))
        boxes = mx.array([[0.0, 0.0, 100.0, 100.0], [50.0, 50.0, 200.0, 200.0]])
        result = process_mask(protos, coeffs, boxes, (200, 200))
        mx.eval(result)
        r = np.array(result)
        assert set(np.unique(r)).issubset({0, 1})


class TestProcessMaskUpsample:
    """Verify upsampled mask generation."""

    def test_output_at_original_resolution(self):
        """Output should be at the target shape."""
        protos = mx.random.normal((40, 40, 32))
        coeffs = mx.random.normal((2, 32))
        boxes = mx.array([[10.0, 10.0, 300.0, 300.0], [50.0, 50.0, 400.0, 400.0]])
        shape = (480, 640)
        result = process_mask_upsample(protos, coeffs, boxes, shape)
        mx.eval(result)
        assert result.shape[0] == 2
        assert result.shape[1] == shape[0]
        assert result.shape[2] == shape[1]


# ---------------------------------------------------------------------------
# polygon2mask
# ---------------------------------------------------------------------------


class TestPolygon2Mask:
    """Verify polygon rasterization into binary masks."""

    def test_rectangle(self):
        """A rectangle polygon should fill the expected region."""
        cv2 = pytest.importorskip("cv2", reason="polygon2mask requires OpenCV")  # noqa: F841
        from yolo26mlx.data.coco_dataset import polygon2mask

        poly = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        mask = polygon2mask((100, 100), poly, downsample_ratio=1)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        # Center should be filled
        assert mask[50, 50] == 1
        # Corners outside polygon should be empty
        assert mask[0, 0] == 0
        assert mask[99, 99] == 0

    def test_triangle(self):
        """A triangle polygon should produce a triangular mask."""
        cv2 = pytest.importorskip("cv2", reason="polygon2mask requires OpenCV")  # noqa: F841
        from yolo26mlx.data.coco_dataset import polygon2mask

        poly = np.array([[50, 10], [90, 90], [10, 90]])
        mask = polygon2mask((100, 100), poly, downsample_ratio=1)
        assert mask[70, 50] == 1
        assert mask[5, 5] == 0

    def test_downsample(self):
        """Downsampled mask should have smaller spatial dims."""
        cv2 = pytest.importorskip("cv2", reason="polygon2mask requires OpenCV")  # noqa: F841
        from yolo26mlx.data.coco_dataset import polygon2mask

        poly = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        mask = polygon2mask((100, 100), poly, downsample_ratio=4)
        assert mask.shape == (25, 25)
        # Center should still be filled
        assert mask[12, 12] == 1


# ---------------------------------------------------------------------------
# polygons2masks_overlap
# ---------------------------------------------------------------------------


class TestPolygons2MasksOverlap:
    """Verify overlap mask with instance IDs."""

    def test_two_non_overlapping(self):
        """Two non-overlapping polygons get distinct IDs."""
        cv2 = pytest.importorskip("cv2", reason="requires OpenCV")  # noqa: F841
        from yolo26mlx.data.coco_dataset import polygons2masks_overlap

        seg1 = np.array([[10, 10], [40, 10], [40, 40], [10, 40]])
        seg2 = np.array([[60, 60], [90, 60], [90, 90], [60, 90]])
        masks, sort_idx = polygons2masks_overlap((100, 100), [seg1, seg2])
        assert masks.shape == (100, 100)
        assert masks[25, 25] == 1  # first polygon → ID 1
        assert masks[75, 75] == 2  # second polygon → ID 2
        assert masks[50, 50] == 0  # gap between

    def test_returns_sort_index(self):
        cv2 = pytest.importorskip("cv2", reason="requires OpenCV")  # noqa: F841
        from yolo26mlx.data.coco_dataset import polygons2masks_overlap

        seg1 = np.array([[10, 10], [40, 10], [40, 40], [10, 40]])
        masks, sort_idx = polygons2masks_overlap((100, 100), [seg1])
        assert isinstance(sort_idx, np.ndarray)
        assert len(sort_idx) == 1


# ---------------------------------------------------------------------------
# mask_iou
# ---------------------------------------------------------------------------


class TestMaskIoU:
    """Verify mask IoU computation."""

    def test_identical_masks(self):
        """Identical masks should have IoU = 1.0."""
        mask = np.ones((1, 10, 10), dtype=np.uint8)
        iou = mask_iou(mask, mask)
        np.testing.assert_allclose(iou, [[1.0]], atol=1e-5)

    def test_no_overlap(self):
        """Non-overlapping masks should have IoU = 0.0."""
        m1 = np.zeros((1, 10, 10), dtype=np.uint8)
        m2 = np.zeros((1, 10, 10), dtype=np.uint8)
        m1[0, :5, :] = 1
        m2[0, 5:, :] = 1
        iou = mask_iou(m1, m2)
        np.testing.assert_allclose(iou, [[0.0]], atol=1e-5)

    def test_partial_overlap(self):
        """50% overlap should give IoU = 1/3."""
        m1 = np.zeros((1, 10, 10), dtype=np.uint8)
        m2 = np.zeros((1, 10, 10), dtype=np.uint8)
        m1[0, :5, :] = 1  # top half
        m2[0, 2:7, :] = 1  # middle strip
        # intersection: rows 2-4 = 3 rows × 10 cols = 30
        # m1 area: 5 × 10 = 50, m2 area: 5 × 10 = 50
        # union: 50 + 50 - 30 = 70
        iou = mask_iou(m1, m2)
        expected = 30.0 / 70.0
        np.testing.assert_allclose(iou[0, 0], expected, atol=1e-5)

    def test_multi_mask_matrix(self):
        """IoU should return (N, M) matrix."""
        m1 = np.random.randint(0, 2, (3, 8, 8), dtype=np.uint8)
        m2 = np.random.randint(0, 2, (5, 8, 8), dtype=np.uint8)
        iou = mask_iou(m1, m2)
        assert iou.shape == (3, 5)
        assert (iou >= 0).all()
        assert (iou <= 1.0 + 1e-5).all()

    def test_empty_mask(self):
        """Empty mask vs non-empty should give IoU ~ 0."""
        m1 = np.zeros((1, 10, 10), dtype=np.uint8)
        m2 = np.ones((1, 10, 10), dtype=np.uint8)
        iou = mask_iou(m1, m2)
        np.testing.assert_allclose(iou, [[0.0]], atol=1e-5)


# ---------------------------------------------------------------------------
# SegmentationMetrics
# ---------------------------------------------------------------------------


class TestSegmentationMetrics:
    """Verify mask/box mAP accumulation and computation."""

    def test_perfect_predictions(self):
        """All predictions match GT → mAP should be 1.0."""
        metrics = SegmentationMetrics(num_classes=2)
        boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float32)
        labels = np.array([0, 1])
        masks = np.zeros((2, 20, 20), dtype=np.uint8)
        masks[0, 2:8, 2:8] = 1
        masks[1, 12:18, 12:18] = 1

        metrics.update(
            pred_boxes=boxes,
            pred_scores=np.array([0.95, 0.90]),
            pred_labels=labels,
            pred_masks=masks,
            gt_boxes=boxes,
            gt_labels=labels,
            gt_masks=masks,
        )
        result = metrics.compute()
        # Perfect predictions yield ~1.0 (101-point COCO AP interpolation).
        assert result["mAP50_mask"] >= 0.99
        assert result["mAP50_box"] >= 0.99

    def test_no_predictions(self):
        """No predictions → all zeros."""
        metrics = SegmentationMetrics(num_classes=2)
        metrics.update(
            pred_boxes=np.empty((0, 4)),
            pred_scores=np.empty(0),
            pred_labels=np.empty(0, dtype=int),
            pred_masks=None,
            gt_boxes=np.array([[10, 10, 50, 50]], dtype=np.float32),
            gt_labels=np.array([0]),
            gt_masks=np.ones((1, 20, 20), dtype=np.uint8),
        )
        result = metrics.compute()
        assert result["mAP50_mask"] == 0.0
        assert result["mAP50_box"] == 0.0

    def test_wrong_class_predictions(self):
        """Predictions with wrong class should not contribute to mAP."""
        metrics = SegmentationMetrics(num_classes=3)
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        pred_boxes = gt_boxes.copy()
        gt_masks = np.ones((1, 20, 20), dtype=np.uint8)

        metrics.update(
            pred_boxes=pred_boxes,
            pred_scores=np.array([0.99]),
            pred_labels=np.array([2]),  # wrong class
            pred_masks=gt_masks.copy(),
            gt_boxes=gt_boxes,
            gt_labels=np.array([0]),  # true class is 0
            gt_masks=gt_masks,
        )
        result = metrics.compute()
        assert result["mAP50_mask"] == 0.0
        assert result["mAP50_box"] == 0.0

    def test_multiple_images(self):
        """Metrics should accumulate across multiple update calls."""
        metrics = SegmentationMetrics(num_classes=2)
        box = np.array([[10, 10, 50, 50]], dtype=np.float32)
        mask = np.ones((1, 20, 20), dtype=np.uint8)

        for _ in range(5):
            metrics.update(
                pred_boxes=box,
                pred_scores=np.array([0.9]),
                pred_labels=np.array([0]),
                pred_masks=mask,
                gt_boxes=box,
                gt_labels=np.array([0]),
                gt_masks=mask,
            )
        result = metrics.compute()
        assert result["mAP50_mask"] > 0.0
        assert result["mAP50_box"] > 0.0
