"""TrackerManager — thin wrapper bridging Results and the pure-MLX tracker core.

Handles YAML config loading, tracker instantiation, and the NumPy↔MLX
conversion boundary.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from yolo26mlx.engine.results import Boxes, Results
from yolo26mlx.trackers import TRACKER_MAP


class TrackerManager:
    """Manages tracker lifecycle and Results↔tracker format conversion.

    Attributes:
        cfg: SimpleNamespace of tracker config parameters.
        tracker: Instantiated tracker (BYTETracker or BOTSORT).
    """

    def __init__(self, tracker_cfg: str = "bytetrack.yaml", frame_rate: int = 30):
        """Initialize tracker from YAML config.

        Args:
            tracker_cfg: YAML filename (e.g. "bytetrack.yaml") or absolute path.
            frame_rate: Video frame rate for scaling track_buffer.
        """
        self.cfg = self._load_config(tracker_cfg)
        self.tracker = self._create_tracker(frame_rate)

    def _load_config(self, tracker_cfg: str) -> SimpleNamespace:
        """Load tracker config from YAML file.

        Searches in order:
        1. Absolute/relative path as given
        2. Package cfg/trackers/ directory
        """
        cfg_path = Path(tracker_cfg)
        if not cfg_path.is_file():
            # Look in package cfg directory
            pkg_dir = Path(os.path.dirname(__file__)).parent / "cfg" / "trackers"
            # Also handle case where engine/ is inside the package
            alt_pkg_dir = Path(__file__).resolve().parent.parent / "cfg" / "trackers"
            for d in [pkg_dir, alt_pkg_dir]:
                candidate = d / tracker_cfg
                if candidate.is_file():
                    cfg_path = candidate
                    break
            else:
                # Try from yolo26mlx package root
                import yolo26mlx

                root = Path(yolo26mlx.__file__).parent / "cfg" / "trackers"
                cfg_path = root / tracker_cfg
                if not cfg_path.is_file():
                    raise FileNotFoundError(
                        f"Tracker config not found: {tracker_cfg}. "
                        f"Searched: {tracker_cfg}, {root / tracker_cfg}"
                    )

        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        return SimpleNamespace(**data)

    def _create_tracker(self, frame_rate: int):
        """Instantiate tracker based on config tracker_type.

        Args:
            frame_rate: Video frame rate, used to scale the track buffer.

        Returns:
            Tracker instance (BYTETracker or BOTSORT).

        Raises:
            ValueError: If tracker_type from config is not recognized.
        """
        tracker_type = self.cfg.tracker_type
        if tracker_type not in TRACKER_MAP:
            available = list(TRACKER_MAP.keys())
            raise ValueError(f"Unknown tracker type: {tracker_type}. Available: {available}")
        tracker_cls = TRACKER_MAP[tracker_type]
        return tracker_cls(self.cfg, frame_rate=frame_rate)

    def update(self, results: Results) -> Results:
        """Run tracker update with detection results.

        Conversion boundary:
        1. Pass Results directly to tracker (tracker extracts numpy data)
        2. Tracker runs all math in MLX
        3. Convert tracked output back to Results with track IDs

        Args:
            results: Detection Results from model.predict().

        Returns:
            New Results with track IDs populated in boxes.id.
        """
        # Run tracker — internally uses MLX for all math
        # tracker.update() expects a Results-like object with conf, cls, xywh, __getitem__
        tracked_output = self.tracker.update(results)

        # tracked_output shape: (N, 8) — [x1, y1, x2, y2, track_id, score, cls, idx]
        if len(tracked_output) == 0:
            empty_boxes = Boxes(
                np.empty((0, 6), dtype=np.float32),
                orig_shape=results.orig_shape,
                track_ids=np.empty((0,), dtype=np.int64),
            )
            r = Results(
                orig_img=results.orig_img,
                path=results.path,
                names=results.names,
                boxes=empty_boxes,
            )
            return r

        # Extract fields from tracker output
        xyxy = tracked_output[:, :4]
        track_ids = tracked_output[:, 4].astype(np.int64)
        scores = tracked_output[:, 5]
        cls_ids = tracked_output[:, 6]

        # Build boxes data: [x1, y1, x2, y2, conf, cls]
        boxes_data = np.column_stack([xyxy, scores, cls_ids]).astype(np.float32)
        new_boxes = Boxes(
            boxes_data,
            orig_shape=results.orig_shape,
            track_ids=track_ids,
        )

        r = Results(
            orig_img=results.orig_img,
            path=results.path,
            names=results.names,
            boxes=new_boxes,
        )
        return r

    def reset(self):
        """Reset tracker state."""
        self.tracker.reset()
