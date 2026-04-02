"""Built-in MOT (Multi-Object Tracking) metrics — no extra dependencies.

Computes standard tracking metrics using the existing pure-MLX ``box_iou``
and the pure-Python Hungarian assignment from ``matching.py``.

Metrics
-------
MOTA  — Multi-Object Tracking Accuracy: ``1 - (FN + FP + IDSW) / GT``
IDF1  — ID F1 Score: ``2 * IDTP / (2 * IDTP + IDFP + IDFN)``
MT    — Mostly Tracked (%): GT tracks matched ≥ 80 % of their lifespan
ML    — Mostly Lost (%): GT tracks matched ≤ 20 % of their lifespan
FP    — False Positives
FN    — False Negatives
IDSW  — ID Switches
Frag  — Fragmentations (track interrupted and resumed)
"""

from __future__ import annotations

from collections import defaultdict

import mlx.core as mx
import numpy as np

from yolo26mlx.trackers.matching import linear_assignment
from yolo26mlx.utils.ops import box_iou

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


class MOTAccumulator:
    """Accumulate per-frame matching results for MOT metric computation.

    Usage::

        acc = MOTAccumulator()
        for frame_id in frames:
            acc.update(gt_ids, gt_boxes, pred_ids, pred_boxes)
        metrics = acc.compute()
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

        # Per-frame accumulators
        self.total_gt = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_idsw = 0
        self.total_frag = 0

        # Previous frame GT→pred ID mapping (for ID switch detection)
        self._prev_match: dict[int, int] = {}

        # Per-GT-track bookkeeping for MT/ML and IDF1
        # gt_id → list of booleans (True = matched in that frame)
        self._gt_track_matched: dict[int, list[bool]] = defaultdict(list)
        # gt_id → list of matched pred_ids (or -1 if unmatched)
        self._gt_track_pred_ids: dict[int, list[int]] = defaultdict(list)

        # For IDF1: per-GT longest set of contiguous matches with same pred_id
        # Accumulated via _gt_track_pred_ids after all frames processed.

        # Fragmentation tracking: gt_id → was_matched_last_frame
        self._gt_was_matched: dict[int, bool] = {}

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        gt_ids: np.ndarray,
        gt_boxes: np.ndarray,
        pred_ids: np.ndarray,
        pred_boxes: np.ndarray,
    ) -> None:
        """Register one frame of ground-truth and predicted tracks.

        Args:
            gt_ids: (M,) int array of ground-truth track IDs.
            gt_boxes: (M, 4) float array in xyxy format.
            pred_ids: (N,) int array of predicted track IDs.
            pred_boxes: (N, 4) float array in xyxy format.
        """
        gt_ids = np.asarray(gt_ids, dtype=np.int64).ravel()
        pred_ids = np.asarray(pred_ids, dtype=np.int64).ravel()
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)

        n_gt = len(gt_ids)
        n_pred = len(pred_ids)
        self.total_gt += n_gt

        # Edge cases
        if n_gt == 0 and n_pred == 0:
            return
        if n_gt == 0:
            self.total_fp += n_pred
            return
        if n_pred == 0:
            self.total_fn += n_gt
            for gid in gt_ids:
                self._gt_track_matched[gid].append(False)
                self._gt_track_pred_ids[gid].append(-1)
                # Fragmentation: was matched last frame, now unmatched
                if self._gt_was_matched.get(gid, False):
                    self.total_frag += 1
                self._gt_was_matched[gid] = False
            return

        # Compute IoU cost matrix via pure-MLX box_iou
        iou_matrix = box_iou(
            mx.array(gt_boxes, dtype=mx.float32),
            mx.array(pred_boxes, dtype=mx.float32),
        )
        mx.eval(iou_matrix)
        iou_np = np.array(iou_matrix, dtype=np.float32)  # (M, N)

        # Convert IoU → cost (1 - IoU) for Hungarian assignment
        cost = 1.0 - iou_np
        matches, unmatched_gt_idx, unmatched_pred_idx = linear_assignment(
            cost,
            thresh=1.0 - self.iou_threshold,
        )

        # Count FP / FN
        fp = len(unmatched_pred_idx)
        fn = len(unmatched_gt_idx)
        self.total_fp += fp
        self.total_fn += fn

        # Build match map: gt_idx → pred_idx for this frame
        match_map: dict[int, int] = {}
        for gi, pi in matches:
            match_map[gi] = pi

        # Process matched GTs
        matched_gt_set = set()
        for gi, pi in matches:
            gid = int(gt_ids[gi])
            pid = int(pred_ids[pi])
            matched_gt_set.add(gid)
            self._gt_track_matched[gid].append(True)
            self._gt_track_pred_ids[gid].append(pid)

            # ID switch: same GT matched to a different pred ID than last frame
            if gid in self._prev_match and self._prev_match[gid] != pid:
                self.total_idsw += 1

            # Fragmentation: GT was unmatched last frame, now matched again
            if gid in self._gt_was_matched and not self._gt_was_matched[gid]:
                if any(self._gt_track_matched[gid][:-1]):
                    # Was matched at some point before → fragmentation
                    self.total_frag += 1

            self._gt_was_matched[gid] = True

        # Process unmatched GTs
        for gi in unmatched_gt_idx:
            gid = int(gt_ids[gi])
            self._gt_track_matched[gid].append(False)
            self._gt_track_pred_ids[gid].append(-1)
            if self._gt_was_matched.get(gid, False):
                self.total_frag += 1
            self._gt_was_matched[gid] = False

        # Update previous match mapping
        self._prev_match = {int(gt_ids[gi]): int(pred_ids[pi]) for gi, pi in matches}

    # ------------------------------------------------------------------
    # Final metric computation
    # ------------------------------------------------------------------

    def compute(self) -> dict[str, float]:
        """Compute aggregate MOT metrics from accumulated frames.

        Returns:
            Dict with keys: MOTA, IDF1, MT, ML, FP, FN, IDSW, Frag,
            num_gt_tracks, num_frames_gt.
        """
        # MOTA
        if self.total_gt > 0:
            mota = 1.0 - (self.total_fn + self.total_fp + self.total_idsw) / self.total_gt
        else:
            mota = 0.0

        # MT / ML
        num_gt_tracks = len(self._gt_track_matched)
        mt_count = 0
        ml_count = 0
        for _gid, matched_list in self._gt_track_matched.items():
            if len(matched_list) == 0:
                ml_count += 1
                continue
            ratio = sum(matched_list) / len(matched_list)
            if ratio >= 0.8:
                mt_count += 1
            elif ratio <= 0.2:
                ml_count += 1

        mt = (mt_count / num_gt_tracks * 100.0) if num_gt_tracks > 0 else 0.0
        ml = (ml_count / num_gt_tracks * 100.0) if num_gt_tracks > 0 else 0.0

        # IDF1 — ID-based F1 score
        idtp, idfp, idfn = self._compute_id_metrics()
        denom = 2 * idtp + idfp + idfn
        idf1 = (2 * idtp / denom * 100.0) if denom > 0 else 0.0

        return {
            "MOTA": round(mota * 100.0, 2),
            "IDF1": round(idf1, 2),
            "MT": round(mt, 2),
            "ML": round(ml, 2),
            "FP": self.total_fp,
            "FN": self.total_fn,
            "IDSW": self.total_idsw,
            "Frag": self.total_frag,
            "num_gt_tracks": num_gt_tracks,
            "num_frames_gt": self.total_gt,
        }

    def _compute_id_metrics(self) -> tuple[int, int, int]:
        """Compute IDTP, IDFP, IDFN for IDF1 calculation.

        For each GT track, find the most frequently assigned prediction ID.
        Frames where the GT is matched to that ID count as IDTP.
        Frames where the GT is matched to a different ID or unmatched count as IDFN.
        Prediction frames not contributing to any GT's best-match count as IDFP.
        """
        # For each GT track, find the dominant (most frequent) pred ID
        gt_dominant_pred: dict[int, int] = {}
        gt_idtp: dict[int, int] = {}

        for gid, pred_id_list in self._gt_track_pred_ids.items():
            # Count occurrences of each pred ID (excluding -1 = unmatched)
            counts: dict[int, int] = defaultdict(int)
            for pid in pred_id_list:
                if pid != -1:
                    counts[pid] += 1
            if counts:
                dominant = max(counts, key=counts.get)
                gt_dominant_pred[gid] = dominant
                gt_idtp[gid] = counts[dominant]
            else:
                gt_idtp[gid] = 0

        idtp = sum(gt_idtp.values())

        # IDFN = total GT appearances - IDTP
        total_gt_appearances = sum(len(v) for v in self._gt_track_matched.values())
        idfn = total_gt_appearances - idtp

        # IDFP: for each pred track, count frames NOT assigned as IDTP to any GT
        # First, collect all (frame_index_within_gt, pred_id) that count as IDTP
        pred_idtp_count: dict[int, int] = defaultdict(int)
        for gid, dominant_pid in gt_dominant_pred.items():
            pred_id_list = self._gt_track_pred_ids[gid]
            for pid in pred_id_list:
                if pid == dominant_pid:
                    pred_idtp_count[pid] += 1

        # Total pred appearances
        total_pred_appearances = self.total_gt - self.total_fn + self.total_fp
        idfp = total_pred_appearances - idtp

        return idtp, max(0, idfp), max(0, idfn)


# ------------------------------------------------------------------
# GT Parsing Helper
# ------------------------------------------------------------------


def load_mot_gt(gt_path: str, eval_class: int = 1) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Load MOT ground truth from a gt.txt file.

    Args:
        gt_path: Path to gt/gt.txt in MOTChallenge format.
        eval_class: Class to keep (1 = pedestrian for MOT17).

    Returns:
        Dict mapping frame_id → list of (track_id, xyxy_box) tuples.
    """
    gt_by_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6])
            cls = int(parts[7]) if len(parts) > 7 else 1

            # Filter: only keep specified class with conf > 0
            if cls != eval_class or conf <= 0:
                continue

            # Convert tlwh → xyxy
            xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
            gt_by_frame[frame_id].append((track_id, xyxy))

    return dict(gt_by_frame)
