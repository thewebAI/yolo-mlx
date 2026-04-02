#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""YOLO26 MLX Tracking Demo — video file or webcam.

Usage:
    # Video file tracking (display + save)
    python scripts/track_demo.py --model yolo26n.npz --source video.mp4 --show --save

    # Webcam tracking
    python scripts/track_demo.py --model yolo26n.npz --source 0 --show

    # Frame-by-frame custom processing
    python scripts/track_demo.py --model yolo26n.npz --source video.mp4 --mode framewise --show
"""

import argparse
import logging
import sys

import cv2

from yolo26mlx import YOLO

logger = logging.getLogger(__name__)


def run_batch(
    model, source, show=False, save=False, tracker="bytetrack.yaml", conf=0.25, vid_stride=1
):
    """High-level tracking: process entire video in one call."""
    results = model.track(
        source=source,
        tracker=tracker,
        conf=conf,
        show=show,
        save=save,
        vid_stride=vid_stride,
    )
    logger.info(f"Tracked {len(results)} frames.")
    for i, r in enumerate(results):
        n = len(r.boxes) if r.boxes is not None else 0
        ids = list(r.boxes.id) if r.boxes is not None and r.boxes.is_track else []
        logger.info(f"  Frame {i}: {n} tracks — IDs {ids}")


def run_framewise(model, source, show=False, tracker="bytetrack.yaml", conf=0.25):
    """Frame-by-frame tracking with custom processing loop."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, tracker=tracker, conf=conf, persist=True)
        tracked = results[0]
        n = len(tracked.boxes) if tracked.boxes is not None else 0
        ids = list(tracked.boxes.id) if tracked.boxes is not None and tracked.boxes.is_track else []
        logger.info(f"Frame {frame_idx}: {n} tracks — IDs {ids}")

        if show:
            annotated = tracked.plot()
            cv2.imshow("YOLO26 Tracking (framewise)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1

    cap.release()
    if show:
        cv2.destroyAllWindows()
    logger.info(f"Processed {frame_idx} frames.")


def main():
    parser = argparse.ArgumentParser(description="YOLO26 MLX Tracking Demo")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--source", required=True, help="Video path or webcam index (0)")
    parser.add_argument(
        "--mode",
        choices=["batch", "framewise"],
        default="batch",
        help="Tracking mode: batch (default) or framewise",
    )
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config YAML")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display annotated frames")
    parser.add_argument("--save", action="store_true", help="Save annotated video")
    parser.add_argument(
        "--vid-stride", type=int, default=1, dest="vid_stride", help="Process every Nth frame"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Convert webcam index
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)

    if args.mode == "batch":
        run_batch(
            model,
            source,
            show=args.show,
            save=args.save,
            tracker=args.tracker,
            conf=args.conf,
            vid_stride=args.vid_stride,
        )
    else:
        run_framewise(model, source, show=args.show, tracker=args.tracker, conf=args.conf)


if __name__ == "__main__":
    main()
