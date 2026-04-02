# Copyright (c) 2026 webAI, Inc.
"""Unit tests for BoT-SORT tracker (BOTrack + BOTSORT + GMC).

Run:
    python -m pytest tests/test_botsort.py -v
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from yolo26mlx.engine.results import Boxes, Results
from yolo26mlx.trackers.basetrack import BaseTrack, TrackState
from yolo26mlx.trackers.bot_sort import BOTSORT, BOTrack
from yolo26mlx.trackers.kalman_filter import KalmanFilterXYWH
from yolo26mlx.trackers.utils.gmc import GMC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _botsort_args(**overrides):
    defaults = dict(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=False,
        gmc_method="none",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_results(boxes_data, orig_shape=(480, 640)):
    img = np.zeros((*orig_shape, 3), dtype=np.uint8)
    boxes = Boxes(np.asarray(boxes_data, dtype=np.float32), orig_shape=orig_shape)
    return Results(orig_img=img, path="test.jpg", names={0: "person", 1: "car"}, boxes=boxes)


# ===================================================================
# 1. BOTrack
# ===================================================================


class TestBOTrackFeatures:
    def setup_method(self):
        BaseTrack._count = 0

    def test_feature_ema_smoothing(self):
        feat1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        feat2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        track = BOTrack(np.array([100, 100, 50, 50, 0]), 0.9, 0, feat=feat1)

        # First feature → smooth_feat == normalized feat1
        sf1 = np.array(track.smooth_feat)
        assert sf1[0] > 0.9, "First feature should dominate"

        # Update with second feature
        track.update_features(feat2)
        sf2 = np.array(track.smooth_feat)
        # After EMA: alpha=0.9 → still mostly feat1-direction, but some feat2
        assert sf2[0] > sf2[1], "EMA should retain feat1 dominance (alpha=0.9)"
        assert sf2[1] > 0.0, "feat2 component should be present"

    def test_feature_normalization(self):
        feat = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        track = BOTrack(np.array([100, 100, 50, 50, 0]), 0.9, 0, feat=feat)
        norm = float(mx.linalg.norm(track.smooth_feat))
        np.testing.assert_allclose(norm, 1.0, atol=1e-4)

    def test_no_features(self):
        track = BOTrack(np.array([100, 100, 50, 50, 0]), 0.9, 0)
        assert track.smooth_feat is None
        assert track.curr_feat is None

    def test_feature_history(self):
        track = BOTrack(np.array([50, 50, 30, 30, 0]), 0.9, 0, feat_history=5)
        for i in range(10):
            feat = np.zeros(4, dtype=np.float32)
            feat[i % 4] = 1.0
            track.update_features(feat)
        # Only last 5 features retained
        assert len(track.features) == 5


class TestBOTrackKalman:
    def setup_method(self):
        BaseTrack._count = 0

    def test_uses_xywh_filter(self):
        assert isinstance(BOTrack.shared_kalman, KalmanFilterXYWH)

    def test_activate_and_predict(self):
        kf = KalmanFilterXYWH()
        track = BOTrack(np.array([100, 100, 50, 80, 0]), 0.9, 0)
        track.activate(kf, frame_id=1)
        assert track.state == TrackState.Tracked
        track.predict()
        # Should not crash; mean is updated
        assert track.mean is not None

    def test_multi_predict(self):
        kf = KalmanFilterXYWH()
        t1 = BOTrack(np.array([100, 100, 50, 50, 0]), 0.9, 0)
        t2 = BOTrack(np.array([200, 200, 40, 60, 0]), 0.8, 1)
        t1.activate(kf, 1)
        t2.activate(kf, 1)
        BOTrack.multi_predict([t1, t2])
        # Should not crash, means updated
        assert t1.mean is not None
        assert t2.mean is not None

    def test_tlwh_property(self):
        """After activation, tlwh should match XYWH-based reconstruction."""
        kf = KalmanFilterXYWH()
        track = BOTrack(np.array([100, 100, 50, 80, 0]), 0.9, 0)
        track.activate(kf, 1)
        tlwh = np.array(track.tlwh)
        # cx=100, cy=100, w=50, h=80 → tl_x=75, tl_y=60, w=50, h=80
        np.testing.assert_allclose(tlwh, [75, 60, 50, 80], atol=1.0)


class TestBOTrackGMC:
    def setup_method(self):
        BaseTrack._count = 0

    def test_multi_gmc_identity(self):
        """Identity affine should not change track positions."""
        kf = KalmanFilterXYWH()
        track = BOTrack(np.array([100, 100, 50, 50, 0]), 0.9, 0)
        track.activate(kf, 1)
        mean_before = np.array(track.mean).copy()
        H = np.eye(2, 3, dtype=np.float64)
        BOTrack.multi_gmc([track], H)
        mean_after = np.array(track.mean)
        np.testing.assert_allclose(mean_after, mean_before, atol=1e-4)

    def test_multi_gmc_translation(self):
        """Translation affine should shift track center."""
        kf = KalmanFilterXYWH()
        track = BOTrack(np.array([100, 100, 50, 50, 0]), 0.9, 0)
        track.activate(kf, 1)
        H = np.eye(2, 3, dtype=np.float64)
        H[0, 2] = 10.0  # shift x by 10
        H[1, 2] = -5.0  # shift y by -5
        BOTrack.multi_gmc([track], H)
        mean = np.array(track.mean)
        np.testing.assert_allclose(mean[0], 110.0, atol=1e-3)
        np.testing.assert_allclose(mean[1], 95.0, atol=1e-3)

    def test_multi_gmc_empty(self):
        """Empty list should not crash."""
        H = np.eye(2, 3, dtype=np.float64)
        BOTrack.multi_gmc([], H)  # no error


# ===================================================================
# 2. BOTSORT Tracker
# ===================================================================


class TestBOTSORTSingleObject:
    def setup_method(self):
        BaseTrack._count = 0

    def test_consistent_id(self):
        args = _botsort_args()
        tracker = BOTSORT(args, frame_rate=30)
        ids = []
        for f in range(10):
            off = f * 5
            data = np.array([[50 + off, 50 + off, 100 + off, 100 + off, 0.9, 0]], dtype=np.float32)
            results = _make_results(data)
            output = tracker.update(results)
            if len(output) > 0:
                ids.append(int(output[0, 4]))
        assert len(set(ids)) == 1, "Single object should keep same ID"


class TestBOTSORTMultiObject:
    def setup_method(self):
        BaseTrack._count = 0

    def test_two_objects(self):
        args = _botsort_args()
        tracker = BOTSORT(args, frame_rate=30)
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
        assert output.shape[0] == 2
        assert len(set(int(x) for x in output[:, 4])) == 2


class TestBOTSORTOcclusion:
    def setup_method(self):
        BaseTrack._count = 0

    def test_reappearance(self):
        args = _botsort_args(track_buffer=30)
        tracker = BOTSORT(args, frame_rate=30)
        first_id = None

        for f in range(5):
            off = f * 3
            data = np.array(
                [[100 + off, 100 + off, 150 + off, 150 + off, 0.9, 0]], dtype=np.float32
            )
            results = _make_results(data)
            output = tracker.update(results)
            if len(output) > 0 and first_id is None:
                first_id = int(output[0, 4])

        # Gap: 2 empty frames
        for _ in range(2):
            empty = _make_results(np.empty((0, 6), dtype=np.float32))
            tracker.update(empty)

        # Reappear
        for f in range(3):
            off = (5 + f) * 3
            data = np.array(
                [[100 + off, 100 + off, 150 + off, 150 + off, 0.9, 0]], dtype=np.float32
            )
            results = _make_results(data)
            output = tracker.update(results)

        if len(output) > 0:
            assert int(output[0, 4]) == first_id


class TestBOTSORTFactoryMethods:
    def test_get_kalmanfilter(self):
        args = _botsort_args()
        tracker = BOTSORT(args)
        kf = tracker.get_kalmanfilter()
        assert isinstance(kf, KalmanFilterXYWH)

    def test_init_track_creates_botracks(self):
        args = _botsort_args()
        tracker = BOTSORT(args)
        data = np.array(
            [
                [50, 50, 100, 100, 0.9, 0],
                [200, 200, 280, 280, 0.8, 1],
            ],
            dtype=np.float32,
        )
        results = _make_results(data)
        tracks = tracker.init_track(results)
        assert len(tracks) == 2
        assert all(isinstance(t, BOTrack) for t in tracks)

    def test_init_track_with_features(self):
        args = _botsort_args()
        tracker = BOTSORT(args)
        data = np.array([[50, 50, 100, 100, 0.9, 0]], dtype=np.float32)
        results = _make_results(data)
        feats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        tracks = tracker.init_track(results, feats=feats)
        assert tracks[0].smooth_feat is not None

    def test_init_track_empty(self):
        args = _botsort_args()
        tracker = BOTSORT(args)
        empty = _make_results(np.empty((0, 6), dtype=np.float32))
        tracks = tracker.init_track(empty)
        assert len(tracks) == 0


class TestBOTSORTWithReID:
    def setup_method(self):
        BaseTrack._count = 0

    def test_fused_distance(self):
        """With ReID enabled, get_dists should fuse IoU + embedding."""
        args = _botsort_args(with_reid=True)
        tracker = BOTSORT(args, frame_rate=30)
        kf = KalmanFilterXYWH()

        # Two tracks with known features
        t1 = BOTrack(
            np.array([100, 100, 50, 50, 0]), 0.9, 0, feat=np.array([1, 0, 0, 0], dtype=np.float32)
        )
        t1.activate(kf, 1)
        t2 = BOTrack(
            np.array([300, 300, 50, 50, 0]), 0.9, 0, feat=np.array([0, 1, 0, 0], dtype=np.float32)
        )
        t2.activate(kf, 1)

        # Two detections: d1 near t1 with similar feat, d2 near t2 with similar feat
        d1 = BOTrack(
            np.array([105, 105, 50, 50, 0]),
            0.85,
            0,
            feat=np.array([0.9, 0.1, 0, 0], dtype=np.float32),
        )
        d1.activate(kf, 1)
        d2 = BOTrack(
            np.array([305, 305, 50, 50, 0]),
            0.85,
            0,
            feat=np.array([0.1, 0.9, 0, 0], dtype=np.float32),
        )
        d2.activate(kf, 1)

        dists = tracker.get_dists([t1, t2], [d1, d2])
        dists_np = np.array(dists)
        # Diagonal (matched) should be lower than off-diagonal
        assert dists_np[0, 0] < dists_np[0, 1]
        assert dists_np[1, 1] < dists_np[1, 0]


class TestBOTSORTReset:
    def test_reset_clears_gmc(self):
        args = _botsort_args()
        tracker = BOTSORT(args, frame_rate=30)
        # Feed a frame
        data = np.array([[50, 50, 100, 100, 0.9, 0]], dtype=np.float32)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        results = _make_results(data)
        tracker.update(results, img=img)
        tracker.reset()
        assert tracker.frame_id == 0
        assert len(tracker.tracked_stracks) == 0


# ===================================================================
# 3. GMC
# ===================================================================


class TestGMC:
    def test_none_method(self):
        gmc = GMC(method="none")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        H = gmc.apply(img)
        expected = np.eye(2, 3, dtype=np.float64)
        np.testing.assert_allclose(H, expected)

    def test_reset(self):
        gmc = GMC(method="none")
        gmc.apply(np.zeros((480, 640, 3), dtype=np.uint8))
        gmc.reset()
        # Should not crash, can apply again
        H = gmc.apply(np.zeros((480, 640, 3), dtype=np.uint8))
        assert H.shape == (2, 3)


# ===================================================================
# 4. TrackerManager with BoT-SORT
# ===================================================================


class TestTrackerManagerBotsort:
    def setup_method(self):
        BaseTrack._count = 0

    def test_load_botsort(self):
        from yolo26mlx.engine.tracker import TrackerManager

        tm = TrackerManager("botsort.yaml", frame_rate=30)
        assert isinstance(tm.tracker, BOTSORT)

    def test_track_with_botsort(self):
        from yolo26mlx.engine.tracker import TrackerManager

        tm = TrackerManager("botsort.yaml", frame_rate=30)
        ids = set()
        for f in range(8):
            off = f * 5
            data = np.array(
                [
                    [50 + off, 50 + off, 100 + off, 100 + off, 0.9, 0],
                ],
                dtype=np.float32,
            )
            results = _make_results(data)
            tracked = tm.update(results)
            if tracked.boxes is not None and len(tracked.boxes) > 0:
                ids.update(int(i) for i in tracked.boxes.id)
        assert len(ids) == 1, "Single object should keep one ID across frames"
