# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""ByteTrack multi-object tracker — STrack and BYTETracker.

STrack: single-track representation with Kalman filter state.
BYTETracker: main tracker implementing the ByteTrack two-stage association.

Reference: Ultralytics trackers/byte_tracker.py
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import matching
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilterXYAH


class STrack(BaseTrack):
    """Single object track with Kalman filter state estimation.

    Attributes:
        shared_kalman: Class-level KalmanFilterXYAH for batch prediction.
        _tlwh: Stored bounding box in top-left-width-height format (mx.array).
        kalman_filter: Instance Kalman filter (set on activation).
        mean: Kalman state mean (8,) mx.array.
        covariance: Kalman state covariance (8, 8) mx.array.
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh, score, cls):
        """Initialize track from detection.

        Args:
            xywh: Bounding box as [cx, cy, w, h, idx] or [cx, cy, w, h, angle, idx].
            score: Detection confidence.
            cls: Class label.
        """
        super().__init__()
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = self._xywh2tlwh(xywh[:4])
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self._cached_tlwh = None
        self._cached_xyxy = None

        self.score = float(score)
        self.tracklet_len = 0
        self.cls = cls
        self.idx = int(xywh[-1])
        self.angle = float(xywh[4]) if len(xywh) == 6 else None

    def _invalidate_coords(self):
        """Clear cached coordinate properties (call when self.mean changes)."""
        self._cached_tlwh = None
        self._cached_xyxy = None

    @staticmethod
    def _xywh2tlwh(xywh):
        """Convert [cx, cy, w, h] to [tl_x, tl_y, w, h] as mx.array.

        Args:
            xywh: Bounding box in center-width-height format (4 elements).

        Returns:
            mx.array in top-left-width-height format (4,).
        """
        x = mx.array(xywh[:4], dtype=mx.float32)
        tl = x[:2] - x[2:] / 2
        return mx.concatenate([tl, x[2:]])

    def predict(self):
        """Predict next state using Kalman filter."""
        if self.state != TrackState.Tracked:
            mean_state = mx.concatenate([self.mean[:7], mx.array([0.0])])
        else:
            mean_state = self.mean
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self._invalidate_coords()

    @staticmethod
    def multi_predict(stracks):
        """Batch Kalman prediction for multiple tracks.

        Args:
            stracks: List of STrack objects to predict.
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
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )
        # Batch-compute coords: ~5 graph nodes instead of N×8 per-track chains.
        # Still lazy — fused with iou_distance into one GPU pass at eval time.
        all_tlwh, all_xyxy = STrack._batch_coords_from_means(multi_mean)
        for i in range(len(stracks)):
            stracks[i].mean = multi_mean[i]
            stracks[i].covariance = multi_covariance[i]
            stracks[i]._cached_tlwh = all_tlwh[i]
            stracks[i]._cached_xyxy = all_xyxy[i]

    def activate(self, kalman_filter, frame_id):
        """Activate a new tracklet.

        Args:
            kalman_filter: KalmanFilter instance for state estimation.
            frame_id: Current frame number.
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))
        self._invalidate_coords()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a lost track with a new detection.

        Args:
            new_track: STrack detection to match against.
            frame_id: Current frame number.
            new_id: If True, assign a new track ID.
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self._invalidate_coords()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track, frame_id):
        """Update matched track with new detection.

        Args:
            new_track: STrack detection matched to this track.
            frame_id: Current frame number.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self._invalidate_coords()
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh):
        """Convert tlwh to Kalman filter measurement format (xyah).

        Args:
            tlwh: Bounding box in top-left-width-height format.

        Returns:
            mx.array measurement vector in [cx, cy, a, h] format.
        """
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """Bounding box in top-left-width-height format (cached)."""
        if self._cached_tlwh is not None:
            return self._cached_tlwh
        if self.mean is None:
            return self._tlwh
        # mean[:4] is [cx, cy, aspect_ratio, height]
        w = self.mean[2:3] * self.mean[3:4]  # a * h
        h = self.mean[3:4]
        tl_x = self.mean[0:1] - w / 2
        tl_y = self.mean[1:2] - h / 2
        self._cached_tlwh = mx.concatenate([tl_x, tl_y, w, h])
        return self._cached_tlwh

    @property
    def xyxy(self):
        """Bounding box in [x1, y1, x2, y2] format (cached)."""
        if self._cached_xyxy is not None:
            return self._cached_xyxy
        t = self.tlwh
        self._cached_xyxy = mx.concatenate([t[:2], t[:2] + t[2:]])
        return self._cached_xyxy

    @property
    def tlbr(self):
        """Alias for xyxy (top-left, bottom-right)."""
        return self.xyxy

    @property
    def xywh(self):
        """Bounding box in [cx, cy, w, h] format."""
        t = self.tlwh
        return mx.concatenate([t[:2] + t[2:] / 2, t[2:]])

    @property
    def xywha(self):
        """Bounding box in [cx, cy, w, h, angle] format."""
        if self.angle is None:
            return self.xywh
        return mx.concatenate([self.xywh, mx.array([self.angle])])

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert [tl_x, tl_y, w, h] to [cx, cy, aspect_ratio, height].

        Args:
            tlwh: Bounding box in top-left-width-height format.

        Returns:
            mx.array in [cx, cy, aspect_ratio, height] format (4,).
        """
        if not isinstance(tlwh, mx.array):
            tlwh = mx.array(tlwh, dtype=mx.float32)
        cx = tlwh[0:1] + tlwh[2:3] / 2
        cy = tlwh[1:2] + tlwh[3:4] / 2
        a = tlwh[2:3] / tlwh[3:4]
        h = tlwh[3:4]
        return mx.concatenate([cx, cy, a, h])

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert [tl_x, tl_y, w, h] to [cx, cy, w, h].

        Args:
            tlwh: Bounding box in top-left-width-height format.

        Returns:
            mx.array in [cx, cy, w, h] format (4,).
        """
        if not isinstance(tlwh, mx.array):
            tlwh = mx.array(tlwh, dtype=mx.float32)
        cx = tlwh[0:1] + tlwh[2:3] / 2
        cy = tlwh[1:2] + tlwh[3:4] / 2
        return mx.concatenate([cx, cy, tlwh[2:3], tlwh[3:4]])

    @staticmethod
    def _batch_coords_from_means(means):
        """Batch-compute tlwh and xyxy from (N, 8) XYAH state means.

        Replaces N per-track lazy chains (~8 graph nodes each) with a single
        batched computation (~5 nodes), dramatically reducing dispatch overhead
        at the next ``mx.eval`` call.

        Args:
            means: (N, 8) state mean matrix in XYAH format.

        Returns:
            (tlwh, xyxy): Tuple of (N, 4) mx.arrays in tlwh and xyxy formats.
        """
        w = means[:, 2:3] * means[:, 3:4]  # aspect_ratio * height
        h = means[:, 3:4]
        tl_x = means[:, 0:1] - w / 2
        tl_y = means[:, 1:2] - h / 2
        tlwh = mx.concatenate([tl_x, tl_y, w, h], axis=1)
        xyxy = mx.concatenate([tl_x, tl_y, tl_x + w, tl_y + h], axis=1)
        return tlwh, xyxy

    @property
    def result(self):
        """Tracking result: [*coords, track_id, score, cls, idx]."""
        coords = self.xywha if self.angle is not None else self.xyxy
        mx.eval(coords)
        return [*np.array(coords).tolist(), self.track_id, self.score, self.cls, self.idx]

    def __repr__(self):
        """String representation with track ID and frame range."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """ByteTrack multi-object tracker with two-stage association.

    Implements the ByteTrack algorithm: first associates high-confidence
    detections, then low-confidence detections, handling unconfirmed
    tracks and track lifecycle management.

    Attributes:
        tracked_stracks: List of currently tracked tracks.
        lost_stracks: List of lost tracks awaiting re-identification.
        removed_stracks: List of removed tracks.
        frame_id: Current frame counter.
        args: Tracker configuration (from YAML).
        max_time_lost: Max frames before a lost track is removed.
        kalman_filter: Shared Kalman filter instance.
    """

    def __init__(self, args, frame_rate=30):
        """Initialize BYTETracker.

        Args:
            args: Namespace with track_buffer, track_high_thresh, track_low_thresh,
                  new_track_thresh, match_thresh, fuse_score.
            frame_rate: Video frame rate (used to scale track_buffer).
        """
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def update(self, results, img=None, feats=None):
        """Update tracker with new detections.

        Args:
            results: Results object with conf, cls, xywh properties and
                     __getitem__ support (added in Step 8).
            img: Optional image (used by BoT-SORT for GMC).
            feats: Optional feature embeddings (used by BoT-SORT for ReID).

        Returns:
            numpy array of shape (N, 8): [x1, y1, x2, y2, track_id, score, cls, idx]
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Separate detections by confidence
        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh
        inds_second = inds_low & inds_high

        results_second = results[inds_second]
        results_main = results[remain_inds]
        feats_keep = feats_second = None
        if feats is not None and len(feats):
            feats_keep = feats[remain_inds]
            feats_second = feats[inds_second]

        detections = self.init_track(results_main, feats_keep)

        # Separate unconfirmed (single-frame) from confirmed tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # --- First association: high-confidence detections ---
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.args.match_thresh
        )

        a, r = self._batch_update_tracks(strack_pool, detections, matches)
        activated_stracks.extend(a)
        refind_stracks.extend(r)

        # --- Second association: low-confidence detections ---
        detections_second = self.init_track(results_second, feats_second)
        r_tracked_stracks = [
            strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track_second, _ = matching.linear_assignment(dists, thresh=0.5)

        a, r = self._batch_update_tracks(r_tracked_stracks, detections_second, matches)
        activated_stracks.extend(a)
        refind_stracks.extend(r)

        for it in u_track_second:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # --- Handle unconfirmed tracks ---
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        a, r = self._batch_update_tracks(unconfirmed, detections, matches)
        activated_stracks.extend(a)
        refind_stracks.extend(r)

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # --- Initialize new tracks ---
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # --- Update track pools ---
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        return self._collect_results()

    def _batch_update_tracks(self, tracks, detections, match_pairs):
        """Batch Kalman updates for matched track-detection pairs.

        Replaces per-track ``kf.update()`` calls with a single batched
        ``kf.multi_update()``, reducing N CPU/GPU sync points to one.
        Also batch-converts measurement coordinates and pre-computes
        output xyxy to minimize lazy graph node count.

        Args:
            tracks: List of STrack objects (track pool for this association stage).
            detections: List of STrack detections in the current frame.
            match_pairs: Array of (track_idx, detection_idx) pairs from assignment.

        Returns:
            (activated, refound): Lists of tracks by category.
        """
        if len(match_pairs) == 0:
            return [], []

        matched_tracks = [tracks[i] for i, _ in match_pairs]
        matched_dets = [detections[j] for _, j in match_pairs]

        means = mx.stack([t.mean for t in matched_tracks])
        covs = mx.stack([t.covariance for t in matched_tracks])

        det_tlwhs = mx.stack([d.tlwh for d in matched_dets])
        measurements = self._batch_convert_coords(det_tlwhs)

        new_means, new_covs = self.kalman_filter.multi_update(means, covs, measurements)

        batch_coords_fn = type(matched_tracks[0])._batch_coords_from_means
        all_tlwh, all_xyxy = batch_coords_fn(new_means)

        activated = []
        refound = []
        for idx, (itracked, idet) in enumerate(match_pairs):
            track = tracks[itracked]
            det = detections[idet]
            was_tracked = track.state == TrackState.Tracked

            track.mean = new_means[idx]
            track.covariance = new_covs[idx]
            track._cached_tlwh = all_tlwh[idx]
            track._cached_xyxy = all_xyxy[idx]

            if was_tracked:
                track.tracklet_len += 1
            else:
                track.tracklet_len = 0

            track.frame_id = self.frame_id
            track.state = TrackState.Tracked
            track.is_activated = True
            track.score = det.score
            track.cls = det.cls
            track.angle = det.angle
            track.idx = det.idx

            if was_tracked:
                activated.append(track)
            else:
                refound.append(track)

        return activated, refound

    def _collect_results(self):
        """Batch-extract tracking results with a single mx.eval.

        Instead of calling STrack.result per-track (N separate mx.eval + np.array),
        batches all coordinates into one mx.stack for a single GPU→CPU transfer.

        Returns:
            numpy array of shape (N, 8): [x1, y1, x2, y2, track_id, score, cls, idx]
            or (N, 9) for oriented bounding boxes with angle.
        """
        active_tracks = [t for t in self.tracked_stracks if t.is_activated]
        if len(active_tracks) == 0:
            return np.empty((0, 8), dtype=np.float32)

        # Check if any track has angle (rare — only OBB tasks)
        has_angle = any(t.angle is not None for t in active_tracks)
        if has_angle:
            # Mixed or all-angle — fall back to per-track (rare path)
            activated = [t.result for t in active_tracks]
            return np.asarray(activated, dtype=np.float32)

        # Common path: all xyxy (4 coords) — single batched eval
        coords_mx = mx.stack([t.xyxy for t in active_tracks])  # (N, 4)
        mx.eval(coords_mx)  # Single GPU→CPU sync
        coords_np = np.array(coords_mx)  # Single transfer

        # Build metadata columns in NumPy
        meta = np.array(
            [[t.track_id, t.score, t.cls, t.idx] for t in active_tracks],
            dtype=np.float32,
        )
        return np.concatenate([coords_np, meta], axis=1)

    def _batch_convert_coords(self, tlwhs):
        """Batch convert (N, 4) tlwh to xyah measurement format.

        Args:
            tlwhs: (N, 4) mx.array of bounding boxes in tlwh format.

        Returns:
            (N, 4) mx.array of measurements in [cx, cy, a, h] format.
        """
        cx = tlwhs[:, 0:1] + tlwhs[:, 2:3] / 2
        cy = tlwhs[:, 1:2] + tlwhs[:, 3:4] / 2
        a = tlwhs[:, 2:3] / tlwhs[:, 3:4]
        h = tlwhs[:, 3:4]
        return mx.concatenate([cx, cy, a, h], axis=1)

    def get_kalmanfilter(self):
        """Return Kalman filter instance. Override in BoT-SORT for XYWH variant.

        Returns:
            KalmanFilterXYAH instance.
        """
        return KalmanFilterXYAH()

    def init_track(self, results, img=None):
        """Create STrack instances from detection results.

        Override in BoT-SORT to include feature embeddings.

        Args:
            results: Results object with xywh, conf, cls properties.
            img: Optional image (unused in ByteTrack, used by BoT-SORT).

        Returns:
            List of STrack instances for the current detections.
        """
        if len(results) == 0:
            return []
        bboxes = results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        return [
            STrack(xywh, s, c)
            for xywh, s, c in zip(bboxes, results.conf, results.cls, strict=False)
        ]

    def get_dists(self, tracks, detections):
        """Compute distance matrix. Override in BoT-SORT for embedding distance.

        Args:
            tracks: List of STrack objects.
            detections: List of STrack detections.

        Returns:
            mx.array cost matrix of shape (len(tracks), len(detections)).
        """
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks):
        """Run batch Kalman prediction. Override in BoT-SORT for XYWH variant.

        Args:
            tracks: List of STrack objects to predict.
        """
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Reset STrack ID counter."""
        STrack.reset_id()

    def reset(self):
        """Reset tracker state completely."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Merge two track lists without duplicates (by track_id).

        Args:
            tlista: First list of STrack objects.
            tlistb: Second list of STrack objects.

        Returns:
            Merged list with unique track_ids.
        """
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            if not exists.get(t.track_id, 0):
                exists[t.track_id] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """Remove tracks in tlistb from tlista by track_id.

        Args:
            tlista: Base list of STrack objects.
            tlistb: List of STrack objects to remove.

        Returns:
            Filtered list of tracks from tlista not in tlistb.
        """
        ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Remove duplicate tracks between two lists based on IoU.

        For tracks with IoU > 0.15, keep the one with longer history.

        Args:
            stracksa: First list of STrack objects (typically tracked).
            stracksb: Second list of STrack objects (typically lost).

        Returns:
            (resa, resb): De-duplicated versions of both lists.
        """
        if len(stracksa) == 0 or len(stracksb) == 0:
            return stracksa, stracksb
        pdist = matching.iou_distance(stracksa, stracksb)
        mx.eval(pdist)
        pairs = np.where(np.array(pdist) < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs, strict=False):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
