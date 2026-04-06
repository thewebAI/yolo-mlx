# Copyright (c) 2026 webAI, Inc.
"""YOLO26 MLX Trackers — Multi-Object Tracking implementations."""

TRACKER_MAP = {}

# ByteTrack (created in Step 6)
try:
    from .byte_tracker import BYTETracker

    TRACKER_MAP["bytetrack"] = BYTETracker
except ImportError:
    pass

# BoT-SORT is optional (Phase 6). Import only when available.
try:
    from .bot_sort import BOTSORT

    TRACKER_MAP["botsort"] = BOTSORT
except ImportError:
    pass
