#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""Create a short sample tracking video from MOT17 image sequence frames.

Extracts the first N frames from a MOT17 sequence and writes them as an MP4
video suitable for quick-start tracking demos.

Usage:
    # Default: 90 frames (~3s) from MOT17-09-SDP → images/pedestrians.mp4
    python scripts/create_sample_video.py

    # Custom sequence and frame count
    python scripts/create_sample_video.py --sequence MOT17-04-SDP --frames 60

    # Custom output path
    python scripts/create_sample_video.py --output my_video.mp4

Requires the MOT17 dataset to be downloaded first:
    bash scripts/download_mot17.sh
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Create sample tracking video from MOT17 frames")
    parser.add_argument("--data", default=None, help="MOT17 dataset directory (default: datasets/MOT17)")
    parser.add_argument("--sequence", default="MOT17-09-SDP", help="MOT17 sequence name")
    parser.add_argument("--frames", type=int, default=90, help="Number of frames to include")
    parser.add_argument("--output", "-o", default=None, help="Output video path (default: images/pedestrians.mp4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Resolve paths relative to project root
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    data_dir = Path(args.data) if args.data else project_dir / "datasets" / "MOT17"
    out_path = Path(args.output) if args.output else project_dir / "images" / "pedestrians.mp4"

    # Find sequence frames
    seq_dir = data_dir / "train" / args.sequence / "img1"
    if not seq_dir.exists():
        logger.error(f"Sequence not found at {seq_dir}")
        logger.error("Download MOT17 first: bash scripts/download_mot17.sh")
        sys.exit(1)

    frames = sorted(seq_dir.glob("*.jpg"))[:args.frames]
    if not frames:
        logger.error(f"No frames found in {seq_dir}")
        sys.exit(1)

    # Read first frame to get dimensions
    sample = cv2.imread(str(frames[0]))
    h, w = sample.shape[:2]

    # Read seqinfo.ini for frame rate
    seqinfo = data_dir / "train" / args.sequence / "seqinfo.ini"
    fps = 30
    if seqinfo.exists():
        for line in seqinfo.read_text().splitlines():
            if line.startswith("frameRate="):
                fps = int(line.split("=")[1])
                break

    # Write video
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        writer.write(img)

    writer.release()
    logger.info(f"Created {out_path} ({len(frames)} frames, {w}x{h}, {fps} FPS, ~{len(frames)/fps:.1f}s)")


if __name__ == "__main__":
    main()
