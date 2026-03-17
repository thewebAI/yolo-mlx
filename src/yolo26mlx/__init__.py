# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 MLX - Pure MLX Implementation of YOLO26

A high-performance implementation of YOLO26 using Apple's MLX framework
for efficient inference and training on Apple Silicon.
"""

__version__ = "0.1.0"
__author__ = "YOLO26 MLX Team"

__all__ = ["YOLO", "__version__"]


def __getattr__(name: str):
    """Lazily import heavy modules only when needed."""
    if name == "YOLO":
        from yolo26mlx.engine.model import YOLO

        return YOLO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
