# Copyright (c) 2026 webAI, Inc.
"""Unit tests for YOLO26 MLX tracking pipeline.

Covers: Kalman filter, matching/assignment, STrack lifecycle,
BYTETracker single/multi/occlusion/new-object, TrackerManager reset,
Results with track IDs, and YOLO.track() integration.

Run:
    python -m pytest tests/test_tracking.py -v
"""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV required for tracking tests")
mx = pytest.importorskip("mlx.core", reason="MLX requires Apple Silicon")

from yolo26mlx.engine.results import Boxes, Results
from yolo26mlx.trackers.basetrack import BaseTrack, TrackState
from yolo26mlx.trackers.byte_tracker import BYTETracker, STrack
from yolo26mlx.trackers.kalman_filter import KalmanFilterXYAH, KalmanFilterXYWH
from yolo26mlx.trackers.matching import (
    iou_distance,
    linear_assignment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Create default BYTETracker args namespace."""
    defaults = dict(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_results(boxes_data, orig_shape=(480, 640), names=None):
    """Build a Results object from an Nx6 numpy array."""
    if names is None:
        names = {0: "person", 1: "car"}
    img = np.zeros((*orig_shape, 3), dtype=np.uint8)
    boxes = Boxes(np.asarray(boxes_data, dtype=np.float32), orig_shape=orig_shape)
    return Results(orig_img=img, path="test.jpg", names=names, boxes=boxes)


# ===================================================================
# 1. Kalman Filter
# ===================================================================


class TestKalmanFilterPredict:
    def test_shape_consistency(self):
        kf = KalmanFilterXYAH()
        mean, cov = kf.initiate(mx.array([100.0, 100.0, 1.0, 50.0]))
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)
        mean2, cov2 = kf.predict(mean, cov)
        assert mean2.shape == (8,)
        assert cov2.shape == (8, 8)

    def test_deterministic(self):
        kf = KalmanFilterXYAH()
        mean, cov = kf.initiate(mx.array([100.0, 100.0, 1.0, 50.0]))
        m1, c1 = kf.predict(mean, cov)
        m2, c2 = kf.predict(mean, cov)
        mx.eval(m1, c1, m2, c2)
        np.testing.assert_allclose(np.array(m1), np.array(m2), atol=1e-5)
        np.testing.assert_allclose(np.array(c1), np.array(c2), atol=1e-5)

    def test_covariance_grows(self):
        kf = KalmanFilterXYAH()
        mean, cov = kf.initiate(mx.array([100.0, 100.0, 1.0, 50.0]))
        _, cov2 = kf.predict(mean, cov)
        mx.eval(cov, cov2)
        diag_before = np.diag(np.array(cov))
        diag_after = np.diag(np.array(cov2))
        assert np.all(diag_after >= diag_before - 1e-6)


class TestKalmanFilterUpdate:
    def test_correction_toward_measurement(self):
        kf = KalmanFilterXYAH()
        mean, cov = kf.initiate(mx.array([100.0, 100.0, 1.0, 50.0]))
        mean_pred, cov_pred = kf.predict(mean, cov)
        measurement = mx.array([120.0, 120.0, 1.0, 50.0])
        mean_upd, cov_upd = kf.update(mean_pred, cov_pred, measurement)
        mx.eval(mean_upd, cov_upd)
        # Updated position should be closer to measurement than prediction
        m_arr = np.array(mean_upd)[:4]
        pred_arr = np.array(mean_pred)[:4]
        meas_arr = np.array(measurement)
        dist_upd = np.linalg.norm(m_arr - meas_arr)
        dist_pred = np.linalg.norm(pred_arr - meas_arr)
        assert dist_upd < dist_pred

    def test_covariance_shrinks(self):
        kf = KalmanFilterXYAH()
        mean, cov = kf.initiate(mx.array([100.0, 100.0, 1.0, 50.0]))
        mean_pred, cov_pred = kf.predict(mean, cov)
        measurement = mx.array([100.0, 100.0, 1.0, 50.0])
        _, cov_upd = kf.update(mean_pred, cov_pred, measurement)
        mx.eval(cov_pred, cov_upd)
        assert np.trace(np.array(cov_upd)) < np.trace(np.array(cov_pred))


class TestKalmanFilterXYWH:
    def test_basic(self):
        kf = KalmanFilterXYWH()
        mean, cov = kf.initiate(mx.array([100.0, 100.0, 50.0, 80.0]))
        assert mean.shape == (8,)
        mean2, cov2 = kf.predict(mean, cov)
        assert mean2.shape == (8,)
        measurement = mx.array([102.0, 102.0, 50.0, 80.0])
        mean3, cov3 = kf.update(mean2, cov2, measurement)
        mx.eval(mean3, cov3)
        assert mean3.shape == (8,)


# ===================================================================
# 2. Matching / Linear Assignment
# ===================================================================


class TestLinearAssignment:
    def test_basic(self):
        cost = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        matches, u_a, u_b = linear_assignment(cost, thresh=0.5)
        matched_pairs = set(map(tuple, matches))
        assert (0, 0) in matched_pairs
        assert (1, 1) in matched_pairs
        assert len(u_a) == 0
        assert len(u_b) == 0

    def test_threshold_filtering(self):
        cost = np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32)
        matches, u_a, u_b = linear_assignment(cost, thresh=0.5)
        assert len(matches) == 0
        assert set(u_a) == {0, 1}
        assert set(u_b) == {0, 1}

    def test_empty(self):
        cost = np.empty((0, 0), dtype=np.float32)
        matches, u_a, u_b = linear_assignment(cost, thresh=0.5)
        assert len(matches) == 0


class TestIoUDistance:
    def test_known_boxes(self):
        """Two overlapping tracks/dets → low cost; non-overlapping → high cost."""
        BaseTrack._count = 0
        kf = KalmanFilterXYAH()
        _args = _make_args()

        # Track A at xywh = (75,75,50,50)
        t1 = STrack(np.array([75, 75, 50, 50, 0]), 0.9, 0)
        t1.activate(kf, 1)

        # Det B overlapping with A: xywh=(85,85,50,50)
        d1 = STrack(np.array([85, 85, 50, 50, 0]), 0.9, 0)
        d1.activate(kf, 1)

        # Det C far away: xywh=(325,325,50,50)
        d2 = STrack(np.array([325, 325, 50, 50, 0]), 0.9, 0)
        d2.activate(kf, 1)

        cost = iou_distance([t1], [d1, d2])
        assert cost.shape == (1, 2)
        # d1 overlaps t1 → lower cost; d2 doesn't → high cost
        assert cost[0, 0] < 0.7
        assert cost[0, 1] > 0.8


# ===================================================================
# 3. STrack Lifecycle
# ===================================================================


class TestSTrackLifecycle:
    def setup_method(self):
        BaseTrack._count = 0

    def test_full_lifecycle(self):
        kf = KalmanFilterXYAH()
        track = STrack(np.array([100, 100, 50, 50, 0]), score=0.9, cls=0)

        # Activate
        track.activate(kf, frame_id=1)
        assert track.state == TrackState.Tracked
        assert track.is_activated is True
        tid = track.track_id
        assert tid > 0

        # Predict
        track.predict()
        # Should not crash, state remains Tracked
        assert track.state == TrackState.Tracked

        # Update with new detection
        det = STrack(np.array([105, 105, 50, 50, 0]), score=0.85, cls=0)
        track.update(det, frame_id=2)
        assert track.track_id == tid  # same ID
        assert track.state == TrackState.Tracked

        # Mark lost
        track.mark_lost()
        assert track.state == TrackState.Lost

        # Mark removed
        track.mark_removed()
        assert track.state == TrackState.Removed

    def test_id_increments(self):
        kf = KalmanFilterXYAH()
        t1 = STrack(np.array([50, 50, 30, 30, 0]), 0.9, 0)
        t1.activate(kf, 1)
        t2 = STrack(np.array([200, 200, 40, 40, 0]), 0.8, 1)
        t2.activate(kf, 1)
        assert t2.track_id == t1.track_id + 1


# ===================================================================
# 4. BYTETracker
# ===================================================================


class TestBYTETrackerSingleObject:
    """Track one object moving diagonally across 10 frames."""

    def setup_method(self):
        BaseTrack._count = 0

    def test_consistent_id(self):
        args = _make_args()
        tracker = BYTETracker(args, frame_rate=30)
        track_ids = []

        for f in range(10):
            offset = f * 5
            data = np.array(
                [[50 + offset, 50 + offset, 100 + offset, 100 + offset, 0.9, 0]], dtype=np.float32
            )
            results = _make_results(data)
            output = tracker.update(results)
            if len(output) > 0:
                track_ids.append(int(output[0, 4]))

        # After warm-up, all track IDs should be the same
        assert len(set(track_ids)) == 1


class TestBYTETrackerMultiObject:
    """Track two objects simultaneously."""

    def setup_method(self):
        BaseTrack._count = 0

    def test_two_objects(self):
        args = _make_args()
        tracker = BYTETracker(args, frame_rate=30)

        for f in range(10):
            off = f * 3
            data = np.array(
                [
                    [50 + off, 50 + off, 100 + off, 100 + off, 0.9, 0],
                    [300 + off, 300 + off, 350 + off, 350 + off, 0.85, 1],
                ],
                dtype=np.float32,
            )
            results = _make_results(data)
            output = tracker.update(results)

        # Last frame should have 2 tracks with distinct IDs
        assert output.shape[0] == 2
        ids = set(int(x) for x in output[:, 4])
        assert len(ids) == 2


class TestBYTETrackerOcclusion:
    """Object disappears for a few frames then reappears."""

    def setup_method(self):
        BaseTrack._count = 0

    def test_reappearance(self):
        args = _make_args(track_buffer=30)
        tracker = BYTETracker(args, frame_rate=30)
        first_id = None

        # Frames 1-5: object present
        for f in range(5):
            off = f * 3
            data = np.array(
                [[100 + off, 100 + off, 150 + off, 150 + off, 0.9, 0]], dtype=np.float32
            )
            results = _make_results(data)
            output = tracker.update(results)
            if len(output) > 0 and first_id is None:
                first_id = int(output[0, 4])

        assert first_id is not None

        # Frames 6-7: object missing (empty detections)
        for _ in range(2):
            empty = _make_results(np.empty((0, 6), dtype=np.float32))
            tracker.update(empty)

        # Frames 8-10: object reappears at similar location
        for f in range(3):
            off = (5 + f) * 3
            data = np.array(
                [[100 + off, 100 + off, 150 + off, 150 + off, 0.9, 0]], dtype=np.float32
            )
            results = _make_results(data)
            output = tracker.update(results)

        # Should recover the same ID
        if len(output) > 0:
            recovered_id = int(output[0, 4])
            assert recovered_id == first_id


class TestBYTETrackerNewObject:
    """New object entering gets a new track ID."""

    def setup_method(self):
        BaseTrack._count = 0

    def test_new_id(self):
        args = _make_args()
        tracker = BYTETracker(args, frame_rate=30)
        all_ids = set()

        # First 5 frames: one object
        for f in range(5):
            off = f * 3
            data = np.array([[50 + off, 50 + off, 100 + off, 100 + off, 0.9, 0]], dtype=np.float32)
            results = _make_results(data)
            output = tracker.update(results)
            for row in output:
                all_ids.add(int(row[4]))

        # Frames 6-10: add a second object far away
        for f in range(5, 10):
            off = f * 3
            data = np.array(
                [
                    [50 + off, 50 + off, 100 + off, 100 + off, 0.9, 0],
                    [400, 400, 450, 450, 0.85, 1],
                ],
                dtype=np.float32,
            )
            results = _make_results(data)
            output = tracker.update(results)
            for row in output:
                all_ids.add(int(row[4]))

        # Should have at least 2 distinct IDs
        assert len(all_ids) >= 2


# ===================================================================
# 5. Tracker Reset
# ===================================================================


class TestTrackerReset:
    def setup_method(self):
        BaseTrack._count = 0

    def test_reset(self):
        from yolo26mlx.engine.tracker import TrackerManager

        tm = TrackerManager("bytetrack.yaml", frame_rate=30)

        # Run a few frames
        for f in range(3):
            data = np.array([[50 + f * 5, 50, 100 + f * 5, 100, 0.9, 0]], dtype=np.float32)
            results = _make_results(data)
            tm.update(results)

        assert tm.tracker.frame_id > 0

        tm.reset()
        assert tm.tracker.frame_id == 0
        assert len(tm.tracker.tracked_stracks) == 0
        assert len(tm.tracker.lost_stracks) == 0


# ===================================================================
# 6. Results with Track IDs
# ===================================================================


class TestResultsWithTrackIDs:
    def setup_method(self):
        BaseTrack._count = 0

    def test_boxes_id_populated(self):
        from yolo26mlx.engine.tracker import TrackerManager

        tm = TrackerManager("bytetrack.yaml", frame_rate=30)
        data = np.array(
            [
                [50, 50, 100, 100, 0.9, 0],
                [200, 200, 280, 280, 0.8, 1],
            ],
            dtype=np.float32,
        )
        results = _make_results(data)

        # Run 2 frames to get tracks
        tm.update(results)
        tracked = tm.update(results)

        assert tracked.boxes is not None
        assert tracked.boxes.is_track is True
        assert len(tracked.boxes.id) == len(tracked.boxes)
        assert all(tracked.boxes.id > 0)

    def test_empty_results(self):
        from yolo26mlx.engine.tracker import TrackerManager

        tm = TrackerManager("bytetrack.yaml", frame_rate=30)
        empty = _make_results(np.empty((0, 6), dtype=np.float32))
        tracked = tm.update(empty)
        assert tracked.boxes is not None
        assert tracked.boxes.is_track is True
        assert len(tracked.boxes) == 0


# ===================================================================
# 7. Integration: YOLO.track()
# ===================================================================


class TestTrackMethodVideo:
    """Integration test using mocked predict() on a short synthetic video."""

    def setup_method(self):
        BaseTrack._count = 0

    def _make_yolo(self):
        from yolo26mlx.engine.model import YOLO

        yolo = YOLO.__new__(YOLO)
        yolo.model_path = None
        yolo.task = "detect"
        yolo.verbose = False
        yolo.model = MagicMock()
        yolo.predictor = None
        yolo.trainer = None
        yolo.validator = None
        yolo._tracker = None
        yolo._tracker_type = None
        yolo.names = {0: "person"}
        yolo.nc = 1
        yolo.stride = None
        self._frame_idx = 0
        return yolo

    def _fake_predict(self, source, conf=0.25, imgsz=640, **kwargs):
        self._frame_idx += 1
        img = source if isinstance(source, np.ndarray) else np.zeros((480, 640, 3), dtype=np.uint8)
        off = self._frame_idx * 3
        data = np.array([[50 + off, 50 + off, 100 + off, 100 + off, 0.9, 0]], dtype=np.float32)
        boxes = Boxes(data, orig_shape=img.shape[:2])
        return [Results(orig_img=img, path="f.jpg", names={0: "person"}, boxes=boxes)]

    def test_track_video(self):
        yolo = self._make_yolo()
        yolo.predict = self._fake_predict

        # Create a small video
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(tmp.name, fourcc, 25, (640, 480))
        for _ in range(10):
            w.write(np.zeros((480, 640, 3), dtype=np.uint8))
        w.release()

        try:
            results = yolo.track(tmp.name)
            assert isinstance(results, list)
            assert len(results) == 10
            for r in results:
                assert isinstance(r, Results)
                assert r.boxes.is_track is True
        finally:
            os.unlink(tmp.name)

    def test_track_numpy_frame(self):
        yolo = self._make_yolo()
        yolo.predict = self._fake_predict
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = yolo.track(frame)
        assert len(results) == 1
        assert results[0].boxes.is_track is True
