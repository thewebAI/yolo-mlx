"""BoT-SORT tracker — extends ByteTrack with appearance + camera motion.

BOTrack(STrack):  Uses KalmanFilterXYWH, adds feature EMA smoothing.
BOTSORT(BYTETracker): Overrides distance computation with embedding
    distance and fuses with IoU. Adds Global Motion Compensation (GMC).

Reference: Ultralytics trackers/bot_sort.py
"""

from __future__ import annotations

from collections import deque

import mlx.core as mx
import numpy as np

from . import matching
from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .kalman_filter import KalmanFilterXYWH
from .utils.gmc import GMC


class BOTrack(STrack):
    """Extended STrack with appearance features and XYWH Kalman filter.

    Attributes:
        shared_kalman: Class-level KalmanFilterXYWH.
        smooth_feat: Exponentially smoothed feature vector (mx.array).
        curr_feat: Current detection feature vector (mx.array).
        features: Deque of recent features.
        alpha: EMA smoothing factor.
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, xywh, score, cls, feat=None, feat_history: int = 50):
        """Initialize BOTrack.

        Args:
            xywh: Bounding box as [cx, cy, w, h, idx] or [cx, cy, w, h, angle, idx].
            score: Detection confidence.
            cls: Class label.
            feat: Optional appearance feature vector (numpy or mx.array).
            feat_history: Max feature history length.
        """
        super().__init__(xywh, score, cls)
        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9  # EMA factor
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat):
        """Normalize and EMA-smooth the appearance feature.

        Args:
            feat: Raw feature vector (numpy array or mx.array).
        """
        feat = mx.array(feat, dtype=mx.float32) if not isinstance(feat, mx.array) else feat
        # L2 normalize
        norm = mx.linalg.norm(feat, keepdims=True)
        feat = feat / mx.maximum(norm, mx.array([1e-6]))
        mx.eval(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat
            # Re-normalize smoothed feature
            sn = mx.linalg.norm(self.smooth_feat, keepdims=True)
            self.smooth_feat = self.smooth_feat / mx.maximum(sn, mx.array([1e-6]))
            mx.eval(self.smooth_feat)
        self.features.append(feat)

    def predict(self):
        """Predict next state using XYWH Kalman filter."""
        if self.state != TrackState.Tracked:
            mean_state = mx.concatenate([self.mean[:7], mx.array([0.0])])
        else:
            mean_state = self.mean
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self._invalidate_coords()

    @staticmethod
    def _batch_coords_from_means(means):
        """Batch-compute tlwh and xyxy from (N, 8) XYWH state means.

        Args:
            means: (N, 8) state mean matrix in XYWH format.

        Returns:
            (tlwh, xyxy): Tuple of (N, 4) mx.arrays in tlwh and xyxy formats.
        """
        w = means[:, 2:3]
        h = means[:, 3:4]
        tl_x = means[:, 0:1] - w / 2
        tl_y = means[:, 1:2] - h / 2
        tlwh = mx.concatenate([tl_x, tl_y, w, h], axis=1)
        xyxy = mx.concatenate([tl_x, tl_y, tl_x + w, tl_y + h], axis=1)
        return tlwh, xyxy

    @staticmethod
    def multi_predict(stracks):
        """Batch Kalman prediction for BOTrack (XYWH variant).

        Args:
            stracks: List of BOTrack objects to predict.
        """
        if len(stracks) == 0:
            return
        means = []
        covs = []
        for st in stracks:
            mean = st.mean
            if st.state != TrackState.Tracked:
                mean = mx.concatenate([mean[:7], mx.array([0.0])])
            means.append(mean)
            covs.append(st.covariance)

        multi_mean = mx.stack(means)
        multi_covariance = mx.stack(covs)
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )
        all_tlwh, all_xyxy = BOTrack._batch_coords_from_means(multi_mean)
        for i in range(len(stracks)):
            stracks[i].mean = multi_mean[i]
            stracks[i].covariance = multi_covariance[i]
            stracks[i]._cached_tlwh = all_tlwh[i]
            stracks[i]._cached_xyxy = all_xyxy[i]

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate track, optionally updating appearance features.

        Args:
            new_track: BOTrack detection to match against.
            frame_id: Current frame number.
            new_id: If True, assign a new track ID.
        """
        super().re_activate(new_track, frame_id, new_id)
        if hasattr(new_track, "curr_feat") and new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

    def update(self, new_track, frame_id):
        """Update matched track with new detection and features.

        Args:
            new_track: BOTrack detection matched to this track.
            frame_id: Current frame number.
        """
        super().update(new_track, frame_id)
        if hasattr(new_track, "curr_feat") and new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

    def convert_coords(self, tlwh):
        """Convert tlwh to XYWH measurement format (for KalmanFilterXYWH).

        Args:
            tlwh: Bounding box in top-left-width-height format.

        Returns:
            mx.array measurement vector in [cx, cy, w, h] format.
        """
        return self.tlwh_to_xywh(tlwh)

    @property
    def tlwh(self):
        """Bounding box in top-left-width-height format (XYWH state, cached)."""
        if self._cached_tlwh is not None:
            return self._cached_tlwh
        if self.mean is None:
            return self._tlwh
        # mean[:4] is [cx, cy, w, h] for XYWH filter
        w = self.mean[2:3]
        h = self.mean[3:4]
        tl_x = self.mean[0:1] - w / 2
        tl_y = self.mean[1:2] - h / 2
        self._cached_tlwh = mx.concatenate([tl_x, tl_y, w, h])
        return self._cached_tlwh

    @staticmethod
    def multi_gmc(stracks, H: np.ndarray):
        """Apply Global Motion Compensation to tracks.

        Warps the track position (mean[0:2]) by the affine transform.

        Args:
            stracks: List of BOTrack objects.
            H: (2, 3) affine matrix from GMC.
        """
        if len(stracks) == 0:
            return
        # Rotation/scale part and translation
        R = H[:2, :2]
        t = H[:2, 2]

        for st in stracks:
            if st.mean is None:
                continue
            # Warp center position
            pos = np.array(st.mean[:2])
            new_pos = R @ pos + t
            # Update mean
            new_mean = np.array(st.mean)
            new_mean[0] = new_pos[0]
            new_mean[1] = new_pos[1]
            st.mean = mx.array(new_mean, dtype=mx.float32)

            # Warp covariance (rotate the position block)
            cov = np.array(st.covariance)
            R4 = np.eye(8)
            R4[:2, :2] = R
            cov = R4 @ cov @ R4.T
            st.covariance = mx.array(cov, dtype=mx.float32)
            st._invalidate_coords()


class BOTSORT(BYTETracker):
    """BoT-SORT tracker — extends BYTETracker with appearance and GMC.

    Overrides three factory methods:
    - get_kalmanfilter() → KalmanFilterXYWH
    - init_track() → BOTrack with features
    - get_dists() → fused IoU + embedding distance

    Adds Global Motion Compensation (GMC) before association.
    """

    def __init__(self, args, frame_rate=30):
        """Initialize BOTSORT tracker.

        Args:
            args: Namespace with tracker config (same as BYTETracker plus
                  proximity_thresh, appearance_thresh, with_reid, gmc_method).
            frame_rate: Video frame rate (used to scale track_buffer).
        """
        super().__init__(args, frame_rate)
        self.proximity_thresh = getattr(args, "proximity_thresh", 0.5)
        self.appearance_thresh = getattr(args, "appearance_thresh", 0.25)
        self.with_reid = getattr(args, "with_reid", False)
        gmc_method = getattr(args, "gmc_method", "sparseOptFlow")
        self.gmc = GMC(method=gmc_method)

    def _batch_convert_coords(self, tlwhs):
        """Batch convert (N, 4) tlwh to xywh measurement format.

        Args:
            tlwhs: (N, 4) mx.array of bounding boxes in tlwh format.

        Returns:
            (N, 4) mx.array of measurements in [cx, cy, w, h] format.
        """
        cx = tlwhs[:, 0:1] + tlwhs[:, 2:3] / 2
        cy = tlwhs[:, 1:2] + tlwhs[:, 3:4] / 2
        return mx.concatenate([cx, cy, tlwhs[:, 2:3], tlwhs[:, 3:4]], axis=1)

    def get_kalmanfilter(self):
        """Use XYWH Kalman filter (width-based, more stable for pedestrians).

        Returns:
            KalmanFilterXYWH instance.
        """
        return KalmanFilterXYWH()

    def init_track(self, results, feats=None):
        """Create BOTrack instances from detection results.

        Args:
            results: Detection Results with xywh/conf/cls.
            feats: Optional (N, D) feature array for appearance matching.

        Returns:
            List of BOTrack instances for the current detections.
        """
        if len(results) == 0:
            return []
        bboxes = results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        tracks = []
        for i, (xywh, s, c) in enumerate(zip(bboxes, results.conf, results.cls, strict=False)):
            feat = feats[i] if feats is not None and len(feats) > i else None
            tracks.append(BOTrack(xywh, s, c, feat=feat))
        return tracks

    def get_dists(self, tracks, detections):
        """Compute fused distance: IoU + optional embedding distance.

        If ReID is enabled and features are available, fuses IoU distance
        with cosine embedding distance using proximity and appearance thresholds.

        Args:
            tracks: List of BOTrack objects.
            detections: List of BOTrack detections.

        Returns:
            mx.array cost matrix of shape (len(tracks), len(detections)).
        """
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        # Fuse with embedding distance when features are available
        if self.with_reid:
            has_feats = (
                len(tracks) > 0
                and len(detections) > 0
                and hasattr(tracks[0], "smooth_feat")
                and tracks[0].smooth_feat is not None
                and hasattr(detections[0], "curr_feat")
                and detections[0].curr_feat is not None
            )
            if has_feats:
                emb_dists = matching.embedding_distance(tracks, detections)
                mx.eval(emb_dists)
                emb_np = np.array(emb_dists)
                dists_np = np.array(dists)

                # Only use embedding where IoU is within proximity range
                # (avoid matching very far-away objects by appearance alone)
                mask = dists_np > self.proximity_thresh
                emb_mask = emb_np > self.appearance_thresh

                # Merge: use minimum of IoU dist and emb dist where valid
                fused = dists_np.copy()
                valid = ~mask & ~emb_mask
                fused[valid] = np.minimum(dists_np[valid], emb_np[valid])
                fused[mask] = 1.0  # too far by IoU
                fused[emb_mask & ~mask] = dists_np[emb_mask & ~mask]

                dists = mx.array(fused, dtype=mx.float32)

        return dists

    def multi_predict(self, tracks):
        """Run batch Kalman prediction using BOTrack's XYWH variant.

        Args:
            tracks: List of BOTrack objects to predict.
        """
        BOTrack.multi_predict(tracks)

    def update(self, results, img=None, feats=None):
        """Update tracker with new detections.

        Adds GMC step: compensate tracked/lost track positions for camera motion.

        Args:
            results: Results object with conf, cls, xywh properties.
            img: Optional image for Global Motion Compensation.
            feats: Optional feature embeddings for ReID matching.

        Returns:
            numpy array of shape (N, 8): [x1, y1, x2, y2, track_id, score, cls, idx].
        """
        if img is not None:
            H = self.gmc.apply(img)
        else:
            H = None

        # Apply GMC to existing tracks before association
        if H is not None:
            BOTrack.multi_gmc(self.tracked_stracks, H)
            BOTrack.multi_gmc(self.lost_stracks, H)

        # Delegate to BYTETracker.update
        return super().update(results, img=img, feats=feats)

    def reset(self):
        """Reset tracker and GMC state."""
        super().reset()
        self.gmc.reset()
