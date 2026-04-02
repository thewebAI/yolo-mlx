# Copyright (c) 2026 webAI, Inc.
# Adapted from Ultralytics AGPL-3.0 — https://ultralytics.com/license
"""Base classes and structures for object tracking in YOLO26 MLX."""

from collections import OrderedDict
from typing import Any

import numpy as np


class TrackState:
    """Possible states of a tracked object.

    Attributes:
        New (int): Newly detected object.
        Tracked (int): Successfully tracked in subsequent frames.
        Lost (int): No longer tracked.
        Removed (int): Removed from tracking.
    """

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Base class for object tracking, providing foundational attributes and methods.

    Attributes:
        _count (int): Class-level counter for unique track IDs.
        track_id (int): Unique identifier for the track.
        is_activated (bool): Whether the track is currently active.
        state (TrackState): Current state of the track.
        history (OrderedDict): Ordered history of the track's states.
        features (list): Features extracted from the object for tracking.
        curr_feature (Any): Current feature of the tracked object.
        score (float): Confidence score of the tracking.
        start_frame (int): Frame number where tracking started.
        frame_id (int): Most recent frame ID processed by the track.
        time_since_update (int): Frames since last update.
        location (tuple): Object location for multi-camera tracking.
    """

    _count = 0

    def __init__(self):
        """Initialize a new track with a unique ID and foundational tracking attributes."""
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        """Return the ID of the most recent frame where the object was tracked."""
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        """Increment and return the next unique global track ID."""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args: Any) -> None:
        """Activate the track with provided arguments."""
        raise NotImplementedError

    def predict(self) -> None:
        """Predict the next state of the track."""
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the track with new observations."""
        raise NotImplementedError

    def mark_lost(self) -> None:
        """Mark the track as lost."""
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Mark the track as removed."""
        self.state = TrackState.Removed

    @staticmethod
    def reset_id() -> None:
        """Reset the global track ID counter to its initial value."""
        BaseTrack._count = 0
