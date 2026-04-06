# Copyright (c) 2026 webAI, Inc.
"""Tests for built-in MOT metrics (utils/mot_metrics.py).

Verifies MOTA, IDF1, MT, ML, FP, FN, IDSW, Frag against
hand-computed values on synthetic sequences.

Run:
    python -m pytest tests/test_mot_metrics.py -v
"""

import os
import tempfile

import numpy as np

from yolo26mlx.utils.mot_metrics import MOTAccumulator, load_mot_gt


class TestMOTAccumulatorPerfectTracking:
    """All predictions perfectly match ground truth — ideal metrics."""

    def test_perfect(self):
        acc = MOTAccumulator(iou_threshold=0.5)

        # 5 frames, 2 objects, perfect overlap and consistent IDs
        for _f in range(5):
            gt_ids = np.array([1, 2])
            gt_boxes = np.array(
                [
                    [10, 10, 50, 50],
                    [100, 100, 150, 150],
                ],
                dtype=np.float32,
            )
            pred_ids = np.array([1, 2])
            pred_boxes = gt_boxes.copy()
            acc.update(gt_ids, gt_boxes, pred_ids, pred_boxes)

        m = acc.compute()
        assert m["MOTA"] == 100.0
        assert m["FP"] == 0
        assert m["FN"] == 0
        assert m["IDSW"] == 0
        assert m["IDF1"] == 100.0
        assert m["MT"] == 100.0
        assert m["ML"] == 0.0


class TestMOTAccumulatorNoDetections:
    """No predictions at all — everything is a false negative."""

    def test_all_fn(self):
        acc = MOTAccumulator(iou_threshold=0.5)

        for _ in range(5):
            gt_ids = np.array([1, 2])
            gt_boxes = np.array(
                [
                    [10, 10, 50, 50],
                    [100, 100, 150, 150],
                ],
                dtype=np.float32,
            )
            acc.update(gt_ids, gt_boxes, np.array([]), np.empty((0, 4)))

        m = acc.compute()
        assert m["FN"] == 10  # 2 GT × 5 frames
        assert m["FP"] == 0
        assert m["IDSW"] == 0
        assert m["MOTA"] == 0.0  # (1 - 10/10) * 100 = 0
        assert m["ML"] == 100.0  # both tracks mostly lost


class TestMOTAccumulatorAllFP:
    """No ground truth but predictions exist — all false positives."""

    def test_all_fp(self):
        acc = MOTAccumulator(iou_threshold=0.5)

        for _ in range(3):
            pred_ids = np.array([1])
            pred_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
            acc.update(np.array([]), np.empty((0, 4)), pred_ids, pred_boxes)

        m = acc.compute()
        assert m["FP"] == 3
        assert m["FN"] == 0
        assert m["num_gt_tracks"] == 0


class TestMOTAccumulatorIDSwitches:
    """Simulate an ID switch: GT object matched to different pred IDs."""

    def test_id_switch(self):
        acc = MOTAccumulator(iou_threshold=0.5)

        gt_ids = np.array([1])
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)

        # Frame 1: GT-1 matched to Pred-10
        acc.update(gt_ids, gt_boxes, np.array([10]), gt_boxes.copy())

        # Frame 2: GT-1 matched to Pred-20 (ID switch!)
        acc.update(gt_ids, gt_boxes, np.array([20]), gt_boxes.copy())

        # Frame 3: GT-1 matched to Pred-20 (no switch)
        acc.update(gt_ids, gt_boxes, np.array([20]), gt_boxes.copy())

        m = acc.compute()
        assert m["IDSW"] == 1
        assert m["FP"] == 0
        assert m["FN"] == 0

    def test_multiple_switches(self):
        acc = MOTAccumulator(iou_threshold=0.5)

        gt_ids = np.array([1])
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)

        pred_id_sequence = [10, 20, 10, 30]  # 3 switches
        for pid in pred_id_sequence:
            acc.update(gt_ids, gt_boxes, np.array([pid]), gt_boxes.copy())

        m = acc.compute()
        assert m["IDSW"] == 3


class TestMOTAccumulatorFragmentation:
    """Test fragmentation: track disappears for a frame then reappears."""

    def test_fragment(self):
        acc = MOTAccumulator(iou_threshold=0.5)

        gt_ids = np.array([1])
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)

        # Frame 1: matched
        acc.update(gt_ids, gt_boxes, np.array([10]), gt_boxes.copy())
        # Frame 2: missing (FN)
        acc.update(gt_ids, gt_boxes, np.array([]), np.empty((0, 4)))
        # Frame 3: matched again (fragmentation)
        acc.update(gt_ids, gt_boxes, np.array([10]), gt_boxes.copy())

        m = acc.compute()
        assert m["Frag"] >= 1
        assert m["FN"] == 1


class TestMOTAccumulatorMTML:
    """Test Mostly Tracked / Mostly Lost computation."""

    def test_mt(self):
        """Track matched >= 80% of frames → Mostly Tracked."""
        acc = MOTAccumulator(iou_threshold=0.5)
        gt_ids = np.array([1])
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)

        # 10 frames, matched in 9 (90%)
        for i in range(10):
            if i == 5:  # miss one frame
                acc.update(gt_ids, gt_boxes, np.array([]), np.empty((0, 4)))
            else:
                acc.update(gt_ids, gt_boxes, np.array([10]), gt_boxes.copy())

        m = acc.compute()
        assert m["MT"] == 100.0  # 90% > 80% → MT

    def test_ml(self):
        """Track matched <= 20% of frames → Mostly Lost."""
        acc = MOTAccumulator(iou_threshold=0.5)
        gt_ids = np.array([1])
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)

        # 10 frames, matched in only 1 (10%)
        for i in range(10):
            if i == 0:
                acc.update(gt_ids, gt_boxes, np.array([10]), gt_boxes.copy())
            else:
                acc.update(gt_ids, gt_boxes, np.array([]), np.empty((0, 4)))

        m = acc.compute()
        assert m["ML"] == 100.0  # 10% <= 20% → ML


class TestMOTAccumulatorIoUThreshold:
    """Predictions with low IoU should not match."""

    def test_low_iou_rejected(self):
        acc = MOTAccumulator(iou_threshold=0.5)
        gt_ids = np.array([1])
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)

        # Far-away prediction → IoU ≈ 0
        pred_boxes = np.array([[200, 200, 250, 250]], dtype=np.float32)
        acc.update(gt_ids, gt_boxes, np.array([1]), pred_boxes)

        m = acc.compute()
        assert m["FP"] == 1
        assert m["FN"] == 1


class TestMOTAccumulatorMixed:
    """Mixed scenario with multiple objects, some matched, some not."""

    def test_mixed(self):
        acc = MOTAccumulator(iou_threshold=0.5)

        # Frame 1: 3 GTs, 2 matched, 1 FN, 0 FP
        gt_ids = np.array([1, 2, 3])
        gt_boxes = np.array(
            [
                [10, 10, 50, 50],
                [100, 100, 150, 150],
                [200, 200, 250, 250],
            ],
            dtype=np.float32,
        )
        pred_ids = np.array([10, 20])
        pred_boxes = np.array(
            [
                [10, 10, 50, 50],
                [100, 100, 150, 150],
            ],
            dtype=np.float32,
        )
        acc.update(gt_ids, gt_boxes, pred_ids, pred_boxes)

        m = acc.compute()
        assert m["FN"] == 1
        assert m["FP"] == 0


class TestMOTAccumulatorIDF1:
    """Test IDF1 computation."""

    def test_perfect_idf1(self):
        acc = MOTAccumulator(iou_threshold=0.5)
        gt_ids = np.array([1])
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)

        for _ in range(10):
            acc.update(gt_ids, gt_boxes, np.array([1]), gt_boxes.copy())

        m = acc.compute()
        assert m["IDF1"] == 100.0

    def test_half_matched_idf1(self):
        acc = MOTAccumulator(iou_threshold=0.5)
        gt_ids = np.array([1])
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)

        # 10 frames, matched in 5
        for i in range(10):
            if i < 5:
                acc.update(gt_ids, gt_boxes, np.array([1]), gt_boxes.copy())
            else:
                acc.update(gt_ids, gt_boxes, np.array([]), np.empty((0, 4)))

        m = acc.compute()
        # IDTP = 5, IDFN = 5, IDFP = 0 → IDF1 = 2*5/(2*5 + 0 + 5) = 66.67
        assert 60.0 < m["IDF1"] < 70.0


class TestMOTAccumulatorEmpty:
    """Edge case: no frames at all."""

    def test_empty(self):
        acc = MOTAccumulator()
        m = acc.compute()
        assert m["MOTA"] == 0.0
        assert m["IDF1"] == 0.0
        assert m["FP"] == 0
        assert m["FN"] == 0


class TestMOTAccumulatorMultiObject:
    """Two objects tracked across frames with one ID switch."""

    def test_two_objects_one_switch(self):
        acc = MOTAccumulator(iou_threshold=0.5)

        gt_ids = np.array([1, 2])
        gt_boxes = np.array(
            [
                [10, 10, 50, 50],
                [100, 100, 150, 150],
            ],
            dtype=np.float32,
        )

        # Frame 1-3: correct matching
        for _ in range(3):
            acc.update(gt_ids, gt_boxes, np.array([10, 20]), gt_boxes.copy())

        # Frame 4: ID switch on object 1 (pred 30 instead of 10)
        acc.update(gt_ids, gt_boxes, np.array([30, 20]), gt_boxes.copy())

        # Frame 5: back to normal but another switch
        acc.update(gt_ids, gt_boxes, np.array([10, 20]), gt_boxes.copy())

        m = acc.compute()
        assert m["IDSW"] == 2  # switch at frame 4, switch back at frame 5
        assert m["FP"] == 0
        assert m["FN"] == 0


# ------------------------------------------------------------------
# load_mot_gt
# ------------------------------------------------------------------


class TestLoadMotGT:
    def test_basic(self):
        content = (
            "1,1,10,10,40,40,1,1,1.0\n"
            "1,2,100,100,50,50,1,1,0.8\n"
            "2,1,12,12,40,40,1,1,1.0\n"
            "2,3,200,200,30,30,1,7,0.5\n"  # class 7 → filtered out
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            gt = load_mot_gt(path, eval_class=1)
            assert 1 in gt
            assert 2 in gt
            assert len(gt[1]) == 2  # two objects in frame 1
            assert len(gt[2]) == 1  # only one class-1 object in frame 2

            # Check xyxy conversion: tlwh(10,10,40,40) → xyxy(10,10,50,50)
            tid, box = gt[1][0]
            assert tid == 1
            np.testing.assert_allclose(box, [10, 10, 50, 50])
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            path = f.name
        try:
            gt = load_mot_gt(path)
            assert gt == {}
        finally:
            os.unlink(path)

    def test_filter_zero_conf(self):
        content = "1,1,10,10,40,40,0,1,1.0\n"  # conf=0 → filtered
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            gt = load_mot_gt(path, eval_class=1)
            assert gt == {}
        finally:
            os.unlink(path)
