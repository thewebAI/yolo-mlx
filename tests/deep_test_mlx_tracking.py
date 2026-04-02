#!/usr/bin/env python3
"""Deep MLX 0.30.3 compatibility and correctness tests for the tracking pipeline.

Tests every MLX API call used across all tracking modules:
- mx.array, mx.eval, mx.concatenate, mx.stack, mx.zeros, mx.eye
- mx.expand_dims, mx.maximum, mx.matmul, mx.transpose, mx.zeros_like
- mx.linalg.norm, mx.linalg.cholesky, mx.linalg.solve_triangular, mx.linalg.inv
- mx.diag, mx.reshape
- Kalman filter full cycle (initiate → predict → update → multi_predict)
- Matching (iou_distance, embedding_distance, fuse_score, linear_assignment)
- STrack / BYTETracker / BOTrack / BOTSORT full lifecycle
- GMC module
- MOTAccumulator
- TrackerManager with both configs
- Results + Boxes tracking integration
- YOLO.track() with mocked video
"""

import logging
import os
import sys
import tempfile
import traceback
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import mlx.core as mx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
passed = 0
failed = 0
errors = []


def check(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        logger.info(f"  ✓ {name}")
    except Exception as e:
        failed += 1
        tb = traceback.format_exc()
        errors.append((name, tb))
        logger.error(f"  ✗ {name}: {e}")


# ────────────────────────────────────────────────────────────────────
# 1. MLX Core API Checks
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 1. MLX Core API (v0.30.3) ═══")


def test_mx_version():
    assert mx.__version__ == "0.30.3", f"Expected 0.30.3 got {mx.__version__}"


check("mx version is 0.30.3", test_mx_version)


def test_mx_array_creation():
    a = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    mx.eval(a)
    assert a.shape == (3,)
    assert a.dtype == mx.float32


check("mx.array creation + eval", test_mx_array_creation)


def test_mx_concatenate():
    a = mx.array([1.0, 2.0])
    b = mx.array([3.0, 4.0])
    c = mx.concatenate([a, b])
    mx.eval(c)
    np.testing.assert_allclose(np.array(c), [1, 2, 3, 4])


check("mx.concatenate", test_mx_concatenate)


def test_mx_concatenate_axis1():
    a = mx.eye(4)
    b = mx.zeros((4, 4))
    c = mx.concatenate([a, b], axis=1)
    mx.eval(c)
    assert c.shape == (4, 8)


check("mx.concatenate axis=1", test_mx_concatenate_axis1)


def test_mx_stack():
    vecs = [mx.array([1.0, 2.0]), mx.array([3.0, 4.0])]
    s = mx.stack(vecs)
    mx.eval(s)
    assert s.shape == (2, 2)


check("mx.stack", test_mx_stack)


def test_mx_eye_zeros():
    e = mx.eye(4)
    z = mx.zeros((4, 4))
    mx.eval(e, z)
    assert e.shape == (4, 4)
    assert float(e[0, 0]) == 1.0
    assert float(z[0, 0]) == 0.0


check("mx.eye + mx.zeros", test_mx_eye_zeros)


def test_mx_expand_dims():
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.expand_dims(a, 0)
    mx.eval(b)
    assert b.shape == (1, 3)


check("mx.expand_dims", test_mx_expand_dims)


def test_mx_maximum():
    a = mx.array([0.5, -0.3])
    b = mx.maximum(a, mx.array([0.0]))
    mx.eval(b)
    np.testing.assert_allclose(np.array(b), [0.5, 0.0])


check("mx.maximum", test_mx_maximum)


def test_mx_matmul():
    a = mx.array([[1.0, 0.0], [0.0, 1.0]])
    b = mx.array([[2.0], [3.0]])
    c = mx.matmul(a, b)
    mx.eval(c)
    np.testing.assert_allclose(np.array(c), [[2.0], [3.0]])


check("mx.matmul", test_mx_matmul)


def test_mx_transpose():
    a = mx.array([[1.0, 2.0], [3.0, 4.0]])
    t = mx.transpose(a)
    mx.eval(t)
    np.testing.assert_allclose(np.array(t), [[1, 3], [2, 4]])


check("mx.transpose", test_mx_transpose)


def test_mx_zeros_like():
    a = mx.array([1.0, 2.0, 3.0, 4.0])
    z = mx.zeros_like(a)
    mx.eval(z)
    assert z.shape == a.shape
    assert float(mx.sum(z)) == 0.0


check("mx.zeros_like", test_mx_zeros_like)


def test_mx_diag():
    a = mx.array([1.0, 2.0, 3.0])
    d = mx.diag(a)
    mx.eval(d)
    assert d.shape == (3, 3)
    assert float(d[1, 1]) == 2.0


check("mx.diag", test_mx_diag)


def test_mx_reshape():
    a = mx.array([[1, 2], [3, 4]])
    r = mx.reshape(a, (4,))
    mx.eval(r)
    assert r.shape == (4,)


check("mx.reshape", test_mx_reshape)


# ────────────────────────────────────────────────────────────────────
# 2. MLX Linalg API
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 2. MLX Linalg API ═══")


def test_linalg_norm_1d():
    a = mx.array([3.0, 4.0])
    n = mx.linalg.norm(a)
    mx.eval(n)
    np.testing.assert_allclose(float(n), 5.0, atol=1e-5)


check("mx.linalg.norm (1D)", test_linalg_norm_1d)


def test_linalg_norm_keepdims():
    a = mx.array([[3.0, 4.0], [5.0, 12.0]])
    n = mx.linalg.norm(a, axis=1, keepdims=True)
    mx.eval(n)
    np.testing.assert_allclose(np.array(n), [[5.0], [13.0]], atol=1e-4)


check("mx.linalg.norm axis=1 keepdims=True", test_linalg_norm_keepdims)


def test_linalg_cholesky():
    A = mx.array([[4.0, 2.0], [2.0, 3.0]])
    L = mx.linalg.cholesky(A, stream=mx.cpu)
    mx.eval(L)
    # Reconstruct: L @ L^T ≈ A
    recon = mx.matmul(L, mx.transpose(L))
    mx.eval(recon)
    np.testing.assert_allclose(np.array(recon), np.array(A), atol=1e-5)


check("mx.linalg.cholesky (stream=cpu)", test_linalg_cholesky)


def test_linalg_solve_triangular():
    L = mx.array([[2.0, 0.0], [1.0, 3.0]])
    b = mx.array([[4.0], [7.0]])
    x = mx.linalg.solve_triangular(L, b, upper=False, stream=mx.cpu)
    mx.eval(x)
    # L @ x should ≈ b
    check_b = mx.matmul(L, x)
    mx.eval(check_b)
    np.testing.assert_allclose(np.array(check_b), np.array(b), atol=1e-5)


check("mx.linalg.solve_triangular (stream=cpu)", test_linalg_solve_triangular)


def test_linalg_inv():
    A = mx.array([[1.0, 2.0], [3.0, 4.0]])
    Ainv = mx.linalg.inv(A, stream=mx.cpu)
    mx.eval(Ainv)
    product = mx.matmul(A, Ainv)
    mx.eval(product)
    np.testing.assert_allclose(np.array(product), np.eye(2), atol=1e-4)


check("mx.linalg.inv (stream=cpu)", test_linalg_inv)


# ────────────────────────────────────────────────────────────────────
# 3. Kalman Filter
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 3. Kalman Filters ═══")

from yolo26mlx.trackers.kalman_filter import KalmanFilterXYAH, KalmanFilterXYWH  # noqa: E402


def test_kf_xyah_full_cycle():
    kf = KalmanFilterXYAH()
    meas = mx.array([100.0, 100.0, 1.0, 50.0])
    mean, cov = kf.initiate(meas)
    assert mean.shape == (8,), f"mean shape {mean.shape}"
    assert cov.shape == (8, 8), f"cov shape {cov.shape}"
    # Predict
    mean2, cov2 = kf.predict(mean, cov)
    assert mean2.shape == (8,)
    # Update
    mean3, cov3 = kf.update(mean2, cov2, meas)
    assert mean3.shape == (8,)
    mx.eval(mean3, cov3)
    # Position should stay near original
    np.testing.assert_allclose(float(mean3[0]), 100.0, atol=5.0)


check("KalmanFilterXYAH initiate→predict→update", test_kf_xyah_full_cycle)


def test_kf_xywh_full_cycle():
    kf = KalmanFilterXYWH()
    meas = mx.array([100.0, 100.0, 50.0, 80.0])
    mean, cov = kf.initiate(meas)
    assert mean.shape == (8,)
    mean2, cov2 = kf.predict(mean, cov)
    mean3, cov3 = kf.update(mean2, cov2, meas)
    mx.eval(mean3, cov3)
    np.testing.assert_allclose(float(mean3[0]), 100.0, atol=5.0)


check("KalmanFilterXYWH initiate→predict→update", test_kf_xywh_full_cycle)


def test_kf_multi_predict():
    kf = KalmanFilterXYAH()
    means, covs = [], []
    for i in range(5):
        m, c = kf.initiate(mx.array([50.0 * i, 50.0 * i, 1.0, 40.0]))
        means.append(m)
        covs.append(c)
    multi_mean = mx.stack(means)
    multi_cov = mx.stack(covs)
    mm, mc = kf.multi_predict(multi_mean, multi_cov)
    mx.eval(mm, mc)
    assert mm.shape == (5, 8)
    assert mc.shape == (5, 8, 8)


check("KalmanFilter multi_predict (batch=5)", test_kf_multi_predict)


def test_kf_gating_distance():
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(mx.array([100.0, 100.0, 1.0, 50.0]))
    mean, cov = kf.predict(mean, cov)
    meas = mx.array([[100.0, 100.0, 1.0, 50.0], [500.0, 500.0, 2.0, 100.0]])
    dist = kf.gating_distance(mean, cov, meas)
    mx.eval(dist)
    assert dist.shape == (2,)
    # Near measurement should have low distance, far one high
    assert float(dist[0]) < float(dist[1])


check("KalmanFilter gating_distance", test_kf_gating_distance)


# ────────────────────────────────────────────────────────────────────
# 4. Matching Module
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 4. Matching Module ═══")

from yolo26mlx.trackers.basetrack import BaseTrack  # noqa: E402
from yolo26mlx.trackers.byte_tracker import STrack  # noqa: E402
from yolo26mlx.trackers.matching import (  # noqa: E402
    embedding_distance,
    fuse_score,
    iou_distance,
    linear_assignment,
)


def test_linear_assignment_basic():
    cost = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    matches, ua, ub = linear_assignment(cost, thresh=0.5)
    assert len(matches) == 2
    assert len(ua) == 0
    assert len(ub) == 0


check("linear_assignment basic", test_linear_assignment_basic)


def test_linear_assignment_empty():
    cost = np.empty((0, 0), dtype=np.float32)
    matches, ua, ub = linear_assignment(cost, thresh=0.5)
    assert len(matches) == 0


check("linear_assignment empty", test_linear_assignment_empty)


def test_iou_distance_overlapping():
    BaseTrack._count = 0
    kf = KalmanFilterXYAH()
    t1 = STrack(np.array([75, 75, 50, 50, 0]), 0.9, 0)
    t1.activate(kf, 1)
    d1 = STrack(np.array([80, 80, 50, 50, 0]), 0.9, 0)
    d1.activate(kf, 1)
    cost = iou_distance([t1], [d1])
    mx.eval(cost)
    val = float(cost[0, 0])
    assert 0 < val < 0.5, f"Expected IoU cost < 0.5, got {val}"


check("iou_distance overlapping boxes", test_iou_distance_overlapping)


def test_iou_distance_non_overlapping():
    BaseTrack._count = 0
    kf = KalmanFilterXYAH()
    t1 = STrack(np.array([50, 50, 30, 30, 0]), 0.9, 0)
    t1.activate(kf, 1)
    d1 = STrack(np.array([400, 400, 30, 30, 0]), 0.9, 0)
    d1.activate(kf, 1)
    cost = iou_distance([t1], [d1])
    mx.eval(cost)
    assert float(cost[0, 0]) > 0.99


check("iou_distance non-overlapping boxes", test_iou_distance_non_overlapping)


def test_embedding_distance():
    from yolo26mlx.trackers.bot_sort import BOTrack

    BaseTrack._count = 0
    kf = KalmanFilterXYWH()
    t1 = BOTrack(
        np.array([100, 100, 50, 50, 0]), 0.9, 0, feat=np.array([1, 0, 0], dtype=np.float32)
    )
    t1.activate(kf, 1)
    d1 = BOTrack(
        np.array([105, 105, 50, 50, 0]), 0.85, 0, feat=np.array([0.95, 0.05, 0], dtype=np.float32)
    )
    d1.activate(kf, 1)
    d2 = BOTrack(
        np.array([200, 200, 50, 50, 0]), 0.85, 0, feat=np.array([0, 0, 1], dtype=np.float32)
    )
    d2.activate(kf, 1)
    cost = embedding_distance([t1], [d1, d2])
    mx.eval(cost)
    # d1 similar to t1 → low cost; d2 dissimilar → high cost
    assert float(cost[0, 0]) < float(cost[0, 1])


check("embedding_distance cosine", test_embedding_distance)


def test_fuse_score():
    BaseTrack._count = 0
    kf = KalmanFilterXYAH()
    t1 = STrack(np.array([100, 100, 50, 50, 0]), 0.9, 0)
    t1.activate(kf, 1)
    d1 = STrack(np.array([105, 105, 50, 50, 0]), 0.9, 0)
    d1.activate(kf, 1)
    d2 = STrack(np.array([110, 110, 50, 50, 0]), 0.3, 0)
    d2.activate(kf, 1)
    iou_cost = iou_distance([t1], [d1, d2])
    fused = fuse_score(iou_cost, [d1, d2])
    mx.eval(fused)
    # Higher confidence det (d1) should get lower fused cost
    assert float(fused[0, 0]) <= float(fused[0, 1])


check("fuse_score", test_fuse_score)


# ────────────────────────────────────────────────────────────────────
# 5. STrack + BYTETracker
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 5. STrack + BYTETracker ═══")

from yolo26mlx.engine.results import Boxes, Results  # noqa: E402
from yolo26mlx.trackers.byte_tracker import BYTETracker  # noqa: E402


def _make_args(**kw):
    defaults = dict(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _make_results(boxes_data, shape=(480, 640)):
    img = np.zeros((*shape, 3), dtype=np.uint8)
    boxes = Boxes(np.asarray(boxes_data, dtype=np.float32), orig_shape=shape)
    return Results(orig_img=img, path="t.jpg", names={0: "person"}, boxes=boxes)


def test_strack_lifecycle():
    BaseTrack._count = 0
    kf = KalmanFilterXYAH()
    t = STrack(np.array([100, 100, 50, 50, 0]), 0.9, 0)
    t.activate(kf, 1)
    assert t.is_activated
    t.predict()
    det = STrack(np.array([102, 102, 50, 50, 0]), 0.85, 0)
    t.update(det, 2)
    assert t.frame_id == 2
    t.mark_lost()
    from yolo26mlx.trackers.basetrack import TrackState

    assert t.state == TrackState.Lost
    t.mark_removed()
    assert t.state == TrackState.Removed


check("STrack full lifecycle", test_strack_lifecycle)


def test_strack_convert_coords():
    BaseTrack._count = 0
    t = STrack(np.array([100, 100, 50, 50, 0]), 0.9, 0)
    xyah = t.convert_coords(t._tlwh)
    mx.eval(xyah)
    assert xyah.shape == (4,)


check("STrack convert_coords (tlwh→xyah)", test_strack_convert_coords)


def test_strack_tlbr_xywh_xyxy():
    BaseTrack._count = 0
    kf = KalmanFilterXYAH()
    t = STrack(np.array([100, 100, 50, 50, 0]), 0.9, 0)
    t.activate(kf, 1)
    tlbr = t.tlbr
    xywh = t.xywh
    xyxy = t.xyxy
    mx.eval(tlbr, xywh, xyxy)
    assert tlbr.shape == (4,)
    assert xywh.shape == (4,)
    assert xyxy.shape == (4,)
    # tlbr and xyxy should be same
    np.testing.assert_allclose(np.array(tlbr), np.array(xyxy), atol=1e-3)


check("STrack tlbr/xywh/xyxy properties", test_strack_tlbr_xywh_xyxy)


def test_bytetracker_20_frames():
    BaseTrack._count = 0
    tracker = BYTETracker(_make_args(), frame_rate=30)
    ids_seen = set()
    for f in range(20):
        off = f * 3
        data = np.array(
            [[50 + off, 50 + off, 100 + off, 100 + off, 0.9, 0], [300, 300, 350, 350, 0.85, 0]],
            dtype=np.float32,
        )
        out = tracker.update(_make_results(data))
        for row in out:
            ids_seen.add(int(row[4]))
    # Should have exactly 2 distinct IDs
    assert len(ids_seen) == 2, f"Expected 2 IDs, got {ids_seen}"


check("BYTETracker 20-frame 2-object tracking", test_bytetracker_20_frames)


def test_bytetracker_empty_frames():
    BaseTrack._count = 0
    tracker = BYTETracker(_make_args(), frame_rate=30)
    for _ in range(5):
        out = tracker.update(_make_results(np.empty((0, 6), dtype=np.float32)))
        assert out.ndim == 2 and out.shape[1] == 8


check("BYTETracker empty detection frames", test_bytetracker_empty_frames)


def test_bytetracker_reset():
    BaseTrack._count = 0
    tracker = BYTETracker(_make_args(), frame_rate=30)
    data = np.array([[50, 50, 100, 100, 0.9, 0]], dtype=np.float32)
    tracker.update(_make_results(data))
    tracker.update(_make_results(data))
    assert tracker.frame_id > 0
    tracker.reset()
    assert tracker.frame_id == 0
    assert len(tracker.tracked_stracks) == 0


check("BYTETracker reset", test_bytetracker_reset)


# ────────────────────────────────────────────────────────────────────
# 6. BOTrack + BOTSORT
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 6. BOTrack + BOTSORT ═══")

from yolo26mlx.trackers.bot_sort import BOTSORT, BOTrack  # noqa: E402


def _botsort_args(**kw):
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
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_botrack_features_ema():
    BaseTrack._count = 0
    feat1 = np.array([1, 0, 0, 0], dtype=np.float32)
    feat2 = np.array([0, 1, 0, 0], dtype=np.float32)
    t = BOTrack(np.array([100, 100, 50, 50, 0]), 0.9, 0, feat=feat1)
    sf1 = np.array(t.smooth_feat)
    assert sf1[0] > 0.9
    t.update_features(feat2)
    sf2 = np.array(t.smooth_feat)
    assert sf2[0] > sf2[1], "EMA should retain first feature dominance"
    # Verify normalized
    n = float(mx.linalg.norm(t.smooth_feat))
    np.testing.assert_allclose(n, 1.0, atol=1e-4)


check("BOTrack feature EMA + normalization", test_botrack_features_ema)


def test_botrack_predict():
    BaseTrack._count = 0
    kf = KalmanFilterXYWH()
    t = BOTrack(np.array([100, 100, 50, 50, 0]), 0.9, 0)
    t.activate(kf, 1)
    t.predict()
    assert t.mean is not None


check("BOTrack predict (XYWH Kalman)", test_botrack_predict)


def test_botrack_multi_predict():
    BaseTrack._count = 0
    kf = KalmanFilterXYWH()
    tracks = []
    for i in range(3):
        t = BOTrack(np.array([50 * i, 50 * i, 30, 30, 0]), 0.9, 0)
        t.activate(kf, 1)
        tracks.append(t)
    BOTrack.multi_predict(tracks)
    for t in tracks:
        assert t.mean is not None


check("BOTrack multi_predict batch", test_botrack_multi_predict)


def test_botrack_tlwh_from_xywh():
    BaseTrack._count = 0
    kf = KalmanFilterXYWH()
    t = BOTrack(np.array([100, 100, 50, 80, 0]), 0.9, 0)
    t.activate(kf, 1)
    tlwh = np.array(t.tlwh)
    # cx=100, cy=100, w=50, h=80 → tl=(75, 60)
    np.testing.assert_allclose(tlwh, [75, 60, 50, 80], atol=1.0)


check("BOTrack tlwh property (XYWH state)", test_botrack_tlwh_from_xywh)


def test_botrack_gmc_translation():
    BaseTrack._count = 0
    kf = KalmanFilterXYWH()
    t = BOTrack(np.array([100, 200, 50, 50, 0]), 0.9, 0)
    t.activate(kf, 1)
    H = np.eye(2, 3, dtype=np.float64)
    H[0, 2] = 15.0
    H[1, 2] = -10.0
    BOTrack.multi_gmc([t], H)
    m = np.array(t.mean)
    np.testing.assert_allclose(m[0], 115.0, atol=1e-3)
    np.testing.assert_allclose(m[1], 190.0, atol=1e-3)


check("BOTrack GMC translation", test_botrack_gmc_translation)


def test_botsort_20_frames():
    BaseTrack._count = 0
    tracker = BOTSORT(_botsort_args(), frame_rate=30)
    ids_seen = set()
    for f in range(20):
        off = f * 3
        data = np.array([[50 + off, 50 + off, 100 + off, 100 + off, 0.9, 0]], dtype=np.float32)
        out = tracker.update(_make_results(data))
        if len(out) > 0:
            ids_seen.update(int(r[4]) for r in out)
    assert len(ids_seen) == 1, f"Single object should have 1 ID, got {ids_seen}"


check("BOTSORT 20-frame single-object", test_botsort_20_frames)


def test_botsort_two_objects():
    BaseTrack._count = 0
    tracker = BOTSORT(_botsort_args(), frame_rate=30)
    for f in range(15):
        off = f * 3
        data = np.array(
            [
                [50 + off, 50 + off, 100 + off, 100 + off, 0.9, 0],
                [300 + off, 300 + off, 350 + off, 350 + off, 0.85, 0],
            ],
            dtype=np.float32,
        )
        out = tracker.update(_make_results(data))
    assert out.shape[0] == 2
    ids = set(int(x) for x in out[:, 4])
    assert len(ids) == 2


check("BOTSORT 15-frame 2-object", test_botsort_two_objects)


def test_botsort_with_reid():
    BaseTrack._count = 0
    tracker = BOTSORT(_botsort_args(with_reid=True), frame_rate=30)
    kf = KalmanFilterXYWH()
    t1 = BOTrack(
        np.array([100, 100, 50, 50, 0]), 0.9, 0, feat=np.array([1, 0, 0, 0], dtype=np.float32)
    )
    t1.activate(kf, 1)
    d1 = BOTrack(
        np.array([105, 105, 50, 50, 0]), 0.85, 0, feat=np.array([0.9, 0.1, 0, 0], dtype=np.float32)
    )
    d1.activate(kf, 1)
    dists = tracker.get_dists([t1], [d1])
    mx.eval(dists)
    val = float(dists[0, 0]) if hasattr(dists, "__float__") else float(np.array(dists)[0, 0])
    assert val < 0.5, f"Fused dist should be low for similar track+det, got {val}"


check("BOTSORT get_dists with ReID", test_botsort_with_reid)


def test_botsort_with_gmc_img():
    BaseTrack._count = 0
    tracker = BOTSORT(_botsort_args(gmc_method="none"), frame_rate=30)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    data = np.array([[50, 50, 100, 100, 0.9, 0]], dtype=np.float32)
    tracker.update(_make_results(data), img=img)
    # Should not crash
    tracker.update(_make_results(data), img=img)
    assert True  # if we get here, GMC integration works


check("BOTSORT update with img (GMC path)", test_botsort_with_gmc_img)


def test_botsort_reset():
    BaseTrack._count = 0
    tracker = BOTSORT(_botsort_args(), frame_rate=30)
    data = np.array([[50, 50, 100, 100, 0.9, 0]], dtype=np.float32)
    tracker.update(_make_results(data))
    tracker.reset()
    assert tracker.frame_id == 0
    assert len(tracker.tracked_stracks) == 0


check("BOTSORT reset", test_botsort_reset)


# ────────────────────────────────────────────────────────────────────
# 7. GMC Module
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 7. GMC Module ═══")

from yolo26mlx.trackers.utils.gmc import GMC  # noqa: E402


def test_gmc_none():
    gmc = GMC(method="none")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    H = gmc.apply(img)
    np.testing.assert_allclose(H, np.eye(2, 3), atol=1e-10)


check("GMC method=none", test_gmc_none)


def test_gmc_sparseOptFlow():
    gmc = GMC(method="sparseOptFlow")
    # First frame
    img1 = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (100, 100), (255, 255, 255), -1)
    H1 = gmc.apply(img1)
    assert H1.shape == (2, 3)
    # Second frame (same) — should get near-identity
    H2 = gmc.apply(img1)
    assert H2.shape == (2, 3)
    # Verify near-identity (no motion)
    np.testing.assert_allclose(H2[:2, :2], np.eye(2), atol=0.3)


check("GMC method=sparseOptFlow", test_gmc_sparseOptFlow)


def test_gmc_orb():
    gmc = GMC(method="orb")
    img1 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    H1 = gmc.apply(img1)
    assert H1.shape == (2, 3)
    H2 = gmc.apply(img1)
    assert H2.shape == (2, 3)


check("GMC method=orb", test_gmc_orb)


def test_gmc_reset():
    gmc = GMC(method="sparseOptFlow")
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    gmc.apply(img)
    gmc.reset()
    H = gmc.apply(img)
    assert H.shape == (2, 3)


check("GMC reset", test_gmc_reset)


# ────────────────────────────────────────────────────────────────────
# 8. TrackerManager
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 8. TrackerManager ═══")

from yolo26mlx.engine.tracker import TrackerManager  # noqa: E402


def test_tracker_manager_bytetrack():
    BaseTrack._count = 0
    tm = TrackerManager("bytetrack.yaml", frame_rate=30)
    assert isinstance(tm.tracker, BYTETracker)
    data = np.array([[50, 50, 100, 100, 0.9, 0]], dtype=np.float32)
    for _ in range(5):
        r = tm.update(_make_results(data))
    assert r.boxes.is_track
    assert len(r.boxes) > 0


check("TrackerManager bytetrack.yaml", test_tracker_manager_bytetrack)


def test_tracker_manager_botsort():
    BaseTrack._count = 0
    tm = TrackerManager("botsort.yaml", frame_rate=30)
    assert isinstance(tm.tracker, BOTSORT)
    data = np.array([[50, 50, 100, 100, 0.9, 0]], dtype=np.float32)
    for _ in range(5):
        r = tm.update(_make_results(data))
    assert r.boxes.is_track
    assert len(r.boxes) > 0


check("TrackerManager botsort.yaml", test_tracker_manager_botsort)


def test_tracker_manager_reset():
    BaseTrack._count = 0
    tm = TrackerManager("bytetrack.yaml", frame_rate=30)
    data = np.array([[50, 50, 100, 100, 0.9, 0]], dtype=np.float32)
    tm.update(_make_results(data))
    tm.reset()
    assert tm.tracker.frame_id == 0


check("TrackerManager reset", test_tracker_manager_reset)


# ────────────────────────────────────────────────────────────────────
# 9. MOT Metrics
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 9. MOT Metrics ═══")

from yolo26mlx.utils.mot_metrics import MOTAccumulator, load_mot_gt  # noqa: E402


def test_mot_perfect():
    acc = MOTAccumulator()
    for _f in range(10):
        gt_ids = np.array([1])
        gt_boxes = np.array([[50, 50, 100, 100]], dtype=np.float32)
        pred_ids = np.array([1])
        pred_boxes = np.array([[50, 50, 100, 100]], dtype=np.float32)
        acc.update(gt_ids, gt_boxes, pred_ids, pred_boxes)
    m = acc.compute()
    assert m["MOTA"] == 100.0
    assert m["FP"] == 0
    assert m["FN"] == 0


check("MOTAccumulator perfect tracking", test_mot_perfect)


def test_mot_all_fn():
    acc = MOTAccumulator()
    for _f in range(10):
        gt_ids = np.array([1])
        gt_boxes = np.array([[50, 50, 100, 100]], dtype=np.float32)
        pred_ids = np.array([], dtype=np.int64)
        pred_boxes = np.empty((0, 4), dtype=np.float32)
        acc.update(gt_ids, gt_boxes, pred_ids, pred_boxes)
    m = acc.compute()
    assert m["MOTA"] == 0.0
    assert m["FN"] == 10


check("MOTAccumulator all-FN", test_mot_all_fn)


def test_mot_id_switches():
    acc = MOTAccumulator()
    # Frame 1: gt1→pred1
    acc.update(
        np.array([1]),
        np.array([[50, 50, 100, 100]], dtype=np.float32),
        np.array([1]),
        np.array([[55, 55, 105, 105]], dtype=np.float32),
    )
    # Frame 2: gt1→pred2 (ID switch)
    acc.update(
        np.array([1]),
        np.array([[60, 60, 110, 110]], dtype=np.float32),
        np.array([2]),
        np.array([[62, 62, 112, 112]], dtype=np.float32),
    )
    m = acc.compute()
    assert m["IDSW"] >= 1


check("MOTAccumulator ID switches", test_mot_id_switches)


def test_load_mot_gt():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("1,1,50,50,100,100,1,1,1.0\n")
        f.write("1,2,200,200,50,50,1,1,1.0\n")
        f.write("2,1,55,55,100,100,1,1,1.0\n")
        _ = f.name
    try:
        gt = load_mot_gt(f.name)
        assert 1 in gt and 2 in gt
        assert len(gt[1]) == 2
        assert len(gt[2]) == 1
    finally:
        os.unlink(f.name)


check("load_mot_gt parser", test_load_mot_gt)


# ────────────────────────────────────────────────────────────────────
# 10. Results / Boxes Tracking Support
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 10. Results + Boxes Tracking ═══")


def test_boxes_with_track_ids():
    data = np.array(
        [
            [50, 50, 100, 100, 0.9, 0],
            [200, 200, 280, 280, 0.8, 1],
        ],
        dtype=np.float32,
    )
    track_ids = np.array([1, 2], dtype=np.int64)
    boxes = Boxes(data, orig_shape=(480, 640), track_ids=track_ids)
    assert boxes.is_track
    assert len(boxes.id) == 2
    assert boxes.id[0] == 1
    assert boxes.id[1] == 2


check("Boxes with track IDs", test_boxes_with_track_ids)


def test_boxes_without_track_ids():
    data = np.array([[50, 50, 100, 100, 0.9, 0]], dtype=np.float32)
    boxes = Boxes(data, orig_shape=(480, 640))
    assert not boxes.is_track


check("Boxes without track IDs", test_boxes_without_track_ids)


def test_results_proxy_properties():
    data = np.array([[50, 50, 100, 100, 0.9, 0]], dtype=np.float32)
    r = _make_results(data)
    assert r.conf is not None
    assert r.cls is not None
    assert r.xywh is not None
    assert len(r.conf) == 1


check("Results proxy properties (conf, cls, xywh)", test_results_proxy_properties)


# ────────────────────────────────────────────────────────────────────
# 11. CLI Track Subcommand
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 11. CLI Integration ═══")


def test_cli_has_track_command():
    from yolo26mlx.cli import build_parser

    parser = build_parser()
    # parse 'track' subcommand
    try:
        args = parser.parse_args(["track", "--source", "video.mp4", "--model", "yolo26n.pt"])
        assert args.command == "track"
        assert args.source == "video.mp4"
    except SystemExit as exc:
        raise AssertionError("CLI parser rejected valid 'track' command") from exc


check("CLI 'track' subcommand parses", test_cli_has_track_command)


# ────────────────────────────────────────────────────────────────────
# 12. YOLO.track() Integration
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 12. YOLO.track() Integration ═══")

from yolo26mlx.engine.model import YOLO  # noqa: E402


def test_yolo_track_single_frame():
    BaseTrack._count = 0
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

    def fake_predict(source, conf=0.25, imgsz=640, **kwargs):
        img = source if isinstance(source, np.ndarray) else np.zeros((480, 640, 3), dtype=np.uint8)
        data = np.array([[50, 50, 100, 100, 0.9, 0]], dtype=np.float32)
        boxes = Boxes(data, orig_shape=img.shape[:2])
        return [Results(orig_img=img, path="f.jpg", names={0: "person"}, boxes=boxes)]

    yolo.predict = fake_predict
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = yolo.track(frame)
    assert len(results) == 1
    assert results[0].boxes.is_track


check("YOLO.track() single numpy frame", test_yolo_track_single_frame)


def test_yolo_track_video():
    BaseTrack._count = 0
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
    frame_idx = [0]

    def fake_predict(source, conf=0.25, imgsz=640, **kwargs):
        frame_idx[0] += 1
        img = source if isinstance(source, np.ndarray) else np.zeros((480, 640, 3), dtype=np.uint8)
        off = frame_idx[0] * 3
        data = np.array([[50 + off, 50 + off, 100 + off, 100 + off, 0.9, 0]], dtype=np.float32)
        boxes = Boxes(data, orig_shape=img.shape[:2])
        return [Results(orig_img=img, path="f.jpg", names={0: "person"}, boxes=boxes)]

    yolo.predict = fake_predict

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(tmp.name, fourcc, 25, (640, 480))
    for _ in range(8):
        w.write(np.zeros((480, 640, 3), dtype=np.uint8))
    w.release()

    try:
        results = yolo.track(tmp.name)
        assert len(results) == 8
        for r in results:
            assert r.boxes.is_track
        # All frames should have same track ID
        ids = set()
        for r in results:
            if len(r.boxes) > 0:
                ids.update(int(i) for i in r.boxes.id)
        assert len(ids) == 1, f"Expected 1 track ID across video, got {ids}"
    finally:
        os.unlink(tmp.name)


check("YOLO.track() 8-frame video", test_yolo_track_video)


# ────────────────────────────────────────────────────────────────────
# 13. Video Utils
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 13. Video Utils ═══")

from yolo26mlx.utils.video import VideoSource, get_track_color  # noqa: E402


def test_video_source():
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(tmp.name, fourcc, 25, (320, 240))
    for _ in range(5):
        w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()
    try:
        vs = VideoSource(tmp.name)
        assert vs.fps == 25
        assert vs.width == 320
        assert vs.height == 240
        assert vs.total_frames == 5
        frames = list(vs)
        assert len(frames) == 5
        vs.release()
    finally:
        os.unlink(tmp.name)


check("VideoSource read 5-frame video", test_video_source)


def test_get_track_color():
    c1 = get_track_color(1)
    _c2 = get_track_color(2)
    c3 = get_track_color(1)
    assert c1 == c3, "Same ID should get same color"
    assert isinstance(c1, tuple) and len(c1) == 3


check("get_track_color deterministic", test_get_track_color)


# ────────────────────────────────────────────────────────────────────
# 14. Tracker __init__.py Registry
# ────────────────────────────────────────────────────────────────────
logger.info("\n═══ 14. Tracker Registry ═══")


def test_tracker_map():
    from yolo26mlx.trackers import TRACKER_MAP

    assert "bytetrack" in TRACKER_MAP
    assert "botsort" in TRACKER_MAP
    assert TRACKER_MAP["bytetrack"] is BYTETracker
    assert TRACKER_MAP["botsort"] is BOTSORT


check("TRACKER_MAP has both trackers", test_tracker_map)


# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════
logger.info("\n" + "═" * 60)
logger.info(f"  RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
logger.info("═" * 60)

if errors:
    logger.error("\n  FAILURES:")
    for name, tb in errors:
        logger.error(f"\n  --- {name} ---")
        logger.error(tb)
    sys.exit(1)
else:
    logger.info("  All tests PASSED ✓")
    sys.exit(0)
