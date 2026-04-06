# Copyright (c) 2026 webAI, Inc.
"""Video I/O utilities for tracking pipeline.

Provides VideoSource (frame iterator for files/webcams/RTSP),
VideoWriter (annotated output saver), and track color generation.
"""

from __future__ import annotations

import colorsys

import cv2


class VideoSource:
    """Iterate frames from a video file, webcam index, or RTSP URL.

    Supports ``vid_stride`` to skip frames (e.g. process every Nth frame).

    Attributes:
        source: Original source path or webcam index.
        cap: Underlying cv2.VideoCapture instance.
        vid_stride: Frame skip interval.
        frame_count: Running count of frames read.

    Usage::

        src = VideoSource("video.mp4", vid_stride=1)
        for frame in src:
            ...  # frame is BGR np.ndarray (H, W, 3)
        src.release()
    """

    def __init__(self, source: str | int, vid_stride: int = 1):
        """Open a video source.

        Args:
            source: File path, webcam index (0, 1, …), or RTSP/HTTP URL.
            vid_stride: Yield every Nth frame (1 = every frame).
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise OSError(f"Cannot open video source: {source}")
        self.vid_stride = max(1, int(vid_stride))
        self.frame_count = 0

    def __iter__(self):
        """Yield BGR frames, respecting vid_stride.

        Yields:
            numpy.ndarray: BGR frame with shape (H, W, 3).
        """
        self.frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_count += 1
            if self.frame_count % self.vid_stride != 0:
                continue
            yield frame

    @property
    def fps(self) -> float:
        """Frames per second of the source."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def total_frames(self) -> int:
        """Total frame count (0 for live streams)."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self):
        """Release the underlying capture."""
        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        self.release()


class VideoWriter:
    """Write annotated frames to an MP4 video file.

    Attributes:
        writer: Underlying cv2.VideoWriter instance.

    Usage::

        writer = VideoWriter("out.mp4", fps=30.0, width=640, height=480)
        writer.write(frame)
        writer.release()
    """

    def __init__(self, path: str, fps: float, width: int, height: int):
        """Open an output video file.

        Args:
            path: Output file path (e.g. "output.mp4").
            fps: Frames per second.
            width: Frame width.
            height: Frame height.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise OSError(f"Cannot open video writer: {path}")

    def write(self, frame):
        """Write a single BGR frame.

        Args:
            frame: BGR numpy array with shape (H, W, 3).
        """
        self.writer.write(frame)

    def release(self):
        """Finalize and close the video file."""
        if self.writer is not None:
            self.writer.release()

    def __del__(self):
        self.release()


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Generate a deterministic RGB color for a track ID.

    Uses golden-ratio hue spacing for maximum visual separation
    between adjacent track IDs.  Returns PIL-compatible (R, G, B) ints.

    Args:
        track_id: Integer track identifier (>= 0).

    Returns:
        (R, G, B) tuple with values in [0, 255].
    """
    hue = (track_id * 0.618033988749895) % 1.0  # golden ratio spread
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))
