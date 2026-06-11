# Copyright (c) 2026 webAI, Inc.
"""Unit tests for pose estimation modules and utilities.

Tests the Pose / Pose26 heads, keypoint decode, KeypointLoss, v8PoseLoss sigma
selection, oks_iou, PoseMetrics, and the YOLO-pose data-loader parsing/flip.

Run:
    python -m pytest tests/test_pose.py -v
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX requires Apple Silicon")
nn = pytest.importorskip("mlx.nn", reason="MLX requires Apple Silicon")

from yolo26mlx.nn.modules.head import Pose, Pose26
from yolo26mlx.utils.loss import KeypointLoss, v8PoseLoss
from yolo26mlx.utils.metrics import OKS_SIGMA, PoseMetrics, oks_iou

NK = 17 * 3  # 51 keypoint values per anchor
NK_SIGMA = 17 * 2  # 34 sigma values per anchor
CH = (64, 128, 256)


def _features(batch: int = 1):
    """Synthetic P3/P4/P5 feature maps for a 640x640 input (NHWC)."""
    p3 = mx.random.normal((batch, 80, 80, CH[0]))
    p4 = mx.random.normal((batch, 40, 40, CH[1]))
    p5 = mx.random.normal((batch, 20, 20, CH[2]))
    return [p3, p4, p5]


# ---------------------------------------------------------------------------
# Pose head (base)
# ---------------------------------------------------------------------------


class TestPoseHead:
    """Verify the base Pose head output formats and one2one keypoint branch."""

    @pytest.fixture()
    def pose(self):
        p = Pose(nc=1, kpt_shape=(17, 3), reg_max=1, end2end=True, ch=CH)
        mx.eval(p.parameters())
        return p

    def test_one2one_cv4_present(self, pose):
        """End2end Pose must build a one2one keypoint head (used at inference)."""
        assert pose.one2one_cv4 is not None
        assert pose.nk == NK

    def test_inference_shape(self, pose):
        """Inference (end2end) returns (B, max_det, 6 + nk)."""
        pose.eval()
        out = pose(_features())
        mx.eval(out)
        assert out.shape == (1, 300, 6 + NK)

    def test_training_dict_has_keypoints(self, pose):
        """Training returns one2many/one2one dicts, each with raw keypoints."""
        pose.train()
        out = pose(_features())
        assert isinstance(out, dict)
        assert "one2many" in out and "one2one" in out
        for branch in ("one2many", "one2one"):
            assert "keypoints" in out[branch]
            kpts = out[branch]["keypoints"]
            mx.eval(kpts)
            assert kpts.shape[0] == 1
            assert kpts.shape[2] == NK

    def test_fuse_strips_heads(self, pose):
        """fuse() should drop the one2many heads used only in training."""
        pose.fuse()
        assert pose.cv2 is None
        assert pose.cv3 is None
        assert pose.cv4 is None


# ---------------------------------------------------------------------------
# Pose26 head (flow-based)
# ---------------------------------------------------------------------------


class TestPose26Head:
    """Verify the YOLO26 Pose26 head split sub-heads and outputs."""

    @pytest.fixture()
    def pose26(self):
        p = Pose26(nc=1, kpt_shape=(17, 3), reg_max=1, end2end=True, ch=CH)
        mx.eval(p.parameters())
        return p

    def test_split_subheads_present(self, pose26):
        """Pose26 splits keypoint + sigma heads and owns a flow model."""
        assert pose26.cv4_kpts is not None
        assert pose26.cv4_sigma is not None
        assert pose26.flow_model is not None
        assert pose26.one2one_cv4_kpts is not None
        assert pose26.nk_sigma == NK_SIGMA

    def test_inference_shape(self, pose26):
        """Inference (end2end) returns (B, max_det, 6 + nk)."""
        pose26.eval()
        out = pose26(_features())
        mx.eval(out)
        assert out.shape == (1, 300, 6 + NK)

    def test_training_dict_has_keypoints_and_sigma(self, pose26):
        """Training emits keypoints + kpts_sigma for both branches."""
        pose26.train()
        out = pose26(_features(batch=2))
        assert isinstance(out, dict)
        for branch in ("one2many", "one2one"):
            assert "keypoints" in out[branch]
            assert "kpts_sigma" in out[branch]
            kpts = out[branch]["keypoints"]
            sigma = out[branch]["kpts_sigma"]
            mx.eval(kpts, sigma)
            assert kpts.shape[0] == 2 and kpts.shape[2] == NK
            assert sigma.shape[0] == 2 and sigma.shape[2] == NK_SIGMA

    def test_fuse_strips_training_only_heads(self, pose26):
        """fuse() drops cv4_kpts/cv4_sigma/flow_model (training-only)."""
        pose26.fuse()
        assert pose26.cv4_kpts is None
        assert pose26.cv4_sigma is None
        assert pose26.flow_model is None
        assert pose26.one2one_cv4_sigma is None


# ---------------------------------------------------------------------------
# Keypoint decode (head, inference)
# ---------------------------------------------------------------------------


class TestKptsDecode:
    """Verify the head-side keypoint decode formulas and visibility sigmoid."""

    def _raw(self, val: float):
        """Single-anchor raw keypoints (1, 1, nk) with x=y=val, v=0."""
        k = np.zeros((1, 1, 17, 3), dtype=np.float32)
        k[..., 0] = val
        k[..., 1] = val
        return mx.array(k.reshape(1, 1, NK))

    def test_base_pose_decode(self):
        """Base Pose: (d*2 + (anchor-0.5))*stride, visibility via sigmoid."""
        pose = Pose(nc=1, kpt_shape=(17, 3), reg_max=1, end2end=True, ch=CH)
        anchor = mx.array([[10.0, 20.0]])
        stride = mx.array([[8.0]])
        out = pose.kpts_decode(self._raw(0.25), anchor, stride)
        mx.eval(out)
        o = np.array(out).reshape(17, 3)
        # x = (0.25*2 + (10 - 0.5)) * 8 = 80 ; y = (0.5 + 19.5) * 8 = 160
        np.testing.assert_allclose(o[0, 0], 80.0, atol=1e-4)
        np.testing.assert_allclose(o[0, 1], 160.0, atol=1e-4)
        np.testing.assert_allclose(o[0, 2], 0.5, atol=1e-5)  # sigmoid(0)

    def test_pose26_decode(self):
        """Pose26: (d + anchor)*stride (no x2, no -0.5)."""
        pose26 = Pose26(nc=1, kpt_shape=(17, 3), reg_max=1, end2end=True, ch=CH)
        anchor = mx.array([[10.0, 20.0]])
        stride = mx.array([[8.0]])
        out = pose26.kpts_decode(self._raw(0.25), anchor, stride)
        mx.eval(out)
        o = np.array(out).reshape(17, 3)
        # x = (0.25 + 10) * 8 = 82 ; y = (0.25 + 20) * 8 = 162
        np.testing.assert_allclose(o[0, 0], 82.0, atol=1e-4)
        np.testing.assert_allclose(o[0, 1], 162.0, atol=1e-4)
        np.testing.assert_allclose(o[0, 2], 0.5, atol=1e-5)

    def test_visibility_in_unit_range(self):
        """Decoded visibility channel must lie in [0, 1] for any logit."""
        pose26 = Pose26(nc=1, kpt_shape=(17, 3), reg_max=1, end2end=True, ch=CH)
        raw = mx.random.normal((1, 5, NK)) * 10.0
        anchor = mx.random.uniform(shape=(5, 2)) * 50.0
        stride = mx.full((5, 1), 8.0)
        out = np.array(pose26.kpts_decode(raw, anchor, stride)).reshape(1, 5, 17, 3)
        vis = out[..., 2]
        assert (vis >= 0.0).all() and (vis <= 1.0).all()


# ---------------------------------------------------------------------------
# KeypointLoss (OKS)
# ---------------------------------------------------------------------------


class TestKeypointLoss:
    """Verify OKS-based keypoint location loss behavior."""

    def test_identical_keypoints_zero_loss(self):
        """Identical predicted/GT keypoints → ~0 loss."""
        loss = KeypointLoss(sigmas=mx.array(OKS_SIGMA))
        gt = mx.random.uniform(shape=(3, 17, 3)) * 100.0
        pred = gt
        mask = mx.ones((3, 17))
        area = mx.full((3,), 1000.0)
        val = float(loss(pred, gt, mask, area))
        assert val == pytest.approx(0.0, abs=1e-5)

    def test_masked_points_ignored(self):
        """A masked-out keypoint should not contribute even if far off."""
        loss = KeypointLoss(sigmas=mx.array(OKS_SIGMA))
        gt = np.zeros((1, 17, 3), dtype=np.float32)
        pred = gt.copy()
        # Move keypoint 0 far away, but mask it out.
        pred[0, 0, 0] = 500.0
        mask = np.ones((1, 17), dtype=np.float32)
        mask[0, 0] = 0.0
        area = mx.full((1,), 1000.0)
        masked = float(loss(mx.array(pred), mx.array(gt), mx.array(mask), area))
        assert masked == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# v8PoseLoss: sigma selection (regression for the tuple/list bug)
# ---------------------------------------------------------------------------


class TestV8PoseLossSigmas:
    """Regression tests for OKS-sigma selection and component count."""

    def test_coco_sigmas_for_list_kpt_shape(self):
        """kpt_shape=[17, 3] (list, as YAML) must select COCO OKS sigmas."""
        head = Pose26(nc=1, kpt_shape=[17, 3], reg_max=1, end2end=True, ch=CH)
        loss = v8PoseLoss(head)
        assert loss.sigmas.shape == (17,)
        # COCO sigmas are non-uniform; uniform fallback would sum to ~1.0.
        assert float(mx.sum(loss.sigmas)) != pytest.approx(1.0, abs=1e-3)

    def test_pose26_loss_is_six_component(self):
        """Pose26 owns a flow model → RLE term present (6 components)."""
        head = Pose26(nc=1, kpt_shape=[17, 3], reg_max=1, end2end=True, ch=CH)
        loss = v8PoseLoss(head)
        assert loss.flow_model is not None

    def test_uniform_sigmas_for_non_coco_shape(self):
        """A non-17 keypoint shape falls back to uniform sigmas summing to 1."""
        head = Pose(nc=1, kpt_shape=(5, 3), reg_max=1, end2end=True, ch=CH)
        loss = v8PoseLoss(head)
        assert loss.sigmas.shape == (5,)
        assert float(mx.sum(loss.sigmas)) == pytest.approx(1.0, abs=1e-5)
        assert loss.flow_model is None


# ---------------------------------------------------------------------------
# oks_iou
# ---------------------------------------------------------------------------


class TestOksIou:
    """Verify Object Keypoint Similarity computation."""

    def _kpts(self, x: float, y: float):
        k = np.zeros((1, 17, 3), dtype=np.float32)
        k[0, :, 0] = x
        k[0, :, 1] = y
        k[0, :, 2] = 2  # visible
        return k

    def test_identical(self):
        """Identical keypoints → OKS = 1.0."""
        gt = self._kpts(50.0, 50.0)
        oks = oks_iou(gt, gt.copy(), areas=np.array([1000.0]), sigmas=OKS_SIGMA)
        assert oks.shape == (1, 1)
        np.testing.assert_allclose(oks[0, 0], 1.0, atol=1e-5)

    def test_no_overlap(self):
        """Far-apart keypoints → OKS ≈ 0."""
        gt = self._kpts(10.0, 10.0)
        pred = self._kpts(1000.0, 1000.0)
        oks = oks_iou(gt, pred, areas=np.array([1000.0]), sigmas=OKS_SIGMA)
        assert oks[0, 0] < 1e-3

    def test_partial_overlap_monotonic(self):
        """Closer predictions yield strictly higher OKS."""
        gt = self._kpts(50.0, 50.0)
        near = self._kpts(52.0, 50.0)
        far = self._kpts(70.0, 50.0)
        area = np.array([1000.0])
        oks_near = oks_iou(gt, near, area, OKS_SIGMA)[0, 0]
        oks_far = oks_iou(gt, far, area, OKS_SIGMA)[0, 0]
        assert 0.0 < oks_far < oks_near < 1.0

    def test_empty_inputs(self):
        """Empty GT or pred returns a correctly-shaped zero matrix."""
        empty = np.zeros((0, 17, 3), dtype=np.float32)
        pred = self._kpts(1.0, 1.0)
        oks = oks_iou(empty, pred, areas=np.zeros(0), sigmas=OKS_SIGMA)
        assert oks.shape == (0, 1)


# ---------------------------------------------------------------------------
# PoseMetrics
# ---------------------------------------------------------------------------


class TestPoseMetrics:
    """Verify keypoint/box mAP accumulation."""

    def _gt(self):
        boxes = np.array([[10, 10, 60, 110]], dtype=np.float32)
        labels = np.array([0])
        kpts = np.zeros((1, 17, 3), dtype=np.float32)
        kpts[0, :, 0] = 35.0
        kpts[0, :, 1] = 60.0
        kpts[0, :, 2] = 2
        return boxes, labels, kpts

    def test_perfect_predictions(self):
        """Predictions identical to GT → mAP = 1.0."""
        m = PoseMetrics(num_classes=1)
        boxes, labels, kpts = self._gt()
        m.update(
            pred_boxes=boxes.copy(),
            pred_scores=np.array([0.95]),
            pred_labels=labels.copy(),
            pred_kpts=kpts.copy(),
            gt_boxes=boxes,
            gt_labels=labels,
            gt_kpts=kpts,
        )
        result = m.compute()
        # A single perfect detection yields ~1.0 (101-point AP interpolation).
        assert result["mAP50_pose"] >= 0.99
        assert result["mAP50_box"] >= 0.99

    def test_no_predictions(self):
        """No predictions → zero mAP."""
        m = PoseMetrics(num_classes=1)
        boxes, labels, kpts = self._gt()
        m.update(
            pred_boxes=np.empty((0, 4), dtype=np.float32),
            pred_scores=np.empty(0),
            pred_labels=np.empty(0, dtype=int),
            pred_kpts=None,
            gt_boxes=boxes,
            gt_labels=labels,
            gt_kpts=kpts,
        )
        result = m.compute()
        assert result["mAP50_pose"] == 0.0
        assert result["mAP50_box"] == 0.0

    def test_multiple_images_accumulate(self):
        """Metrics accumulate across update() calls."""
        m = PoseMetrics(num_classes=1)
        boxes, labels, kpts = self._gt()
        for _ in range(4):
            m.update(
                pred_boxes=boxes.copy(),
                pred_scores=np.array([0.9]),
                pred_labels=labels.copy(),
                pred_kpts=kpts.copy(),
                gt_boxes=boxes,
                gt_labels=labels,
                gt_kpts=kpts,
            )
        result = m.compute()
        assert result["mAP50_pose"] > 0.0
        assert result["mAP50-95_pose"] > 0.0


# ---------------------------------------------------------------------------
# Data loader: YOLO-pose parsing + flip_idx swap
# ---------------------------------------------------------------------------


class TestPoseDataLoader:
    """Verify YOLO-pose label parsing and horizontal-flip keypoint handling."""

    def _dataset(self, tmp_path, flip_idx=None, with_label=False):
        from yolo26mlx.data.coco_dataset import COCODataset

        (tmp_path / "images" / "val2017").mkdir(parents=True)
        (tmp_path / "labels" / "val2017").mkdir(parents=True)
        if with_label:
            from PIL import Image

            Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(
                str(tmp_path / "images" / "val2017" / "1.jpg")
            )
            # cls cx cy w h + 17*(x, y, v)
            vals = ["0", "0.5", "0.5", "0.2", "0.4"]
            for _ in range(17):
                vals += ["0.5", "0.5", "2"]
            (tmp_path / "labels" / "val2017" / "1.txt").write_text(" ".join(vals) + "\n")
        return COCODataset(root=str(tmp_path), split="val2017", task="pose", flip_idx=flip_idx)

    def test_yolo_pose_line_parse(self, tmp_path):
        """A YOLO-pose label line parses into (17, 3) keypoints."""
        ds = self._dataset(tmp_path, with_label=True)
        anns = ds.annotations.get(1)
        assert anns is not None and len(anns) == 1
        kpts = anns[0]["keypoints_raw"]
        assert kpts.shape == (17, 3)
        np.testing.assert_allclose(kpts[0], [0.5, 0.5, 2.0], atol=1e-6)

    def test_letterbox_keypoint_transform(self, tmp_path):
        """__getitem__ runs keypoints through the letterbox pipeline → (n, 17, 3)."""
        ds = self._dataset(tmp_path, with_label=True)
        _, ann = ds[0]
        kpts = ann["keypoints"]
        assert kpts.shape == (1, 17, 3)
        # Centered keypoints stay finite and within the normalized frame.
        xy = kpts[..., :2]
        assert np.isfinite(xy).all()
        assert (xy >= -1e-6).all() and (xy <= 1.0 + 1e-6).all()

    def test_flip_idx_swaps_left_right(self, tmp_path):
        """Horizontal flip mirrors x and reorders keypoints via flip_idx."""
        ds = self._dataset(tmp_path, flip_idx=[0, 2, 1])
        ann = {
            "boxes": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            "keypoints": np.array(
                [[[0.10, 0.5, 2.0], [0.20, 0.5, 2.0], [0.80, 0.5, 2.0]]],
                dtype=np.float32,
            ),
        }
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, out = ds._random_fliplr(img, ann, p=1.0)
        k = out["keypoints"]
        # x flipped (1 - x) → [0.90, 0.80, 0.20], then reorder [0, 2, 1].
        np.testing.assert_allclose(k[0, 0, 0], 0.90, atol=1e-6)
        np.testing.assert_allclose(k[0, 1, 0], 0.20, atol=1e-6)
        np.testing.assert_allclose(k[0, 2, 0], 0.80, atol=1e-6)

    def test_flip_disabled_without_flip_idx(self, tmp_path):
        """Pose flipping is disabled when flip_idx is absent."""
        ds = self._dataset(tmp_path, flip_idx=None)
        ann = {
            "boxes": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            "keypoints": np.array([[[0.1, 0.5, 2.0]]], dtype=np.float32),
        }
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, out = ds._random_fliplr(img, ann, p=1.0)
        np.testing.assert_allclose(out["keypoints"][0, 0, 0], 0.1, atol=1e-6)
