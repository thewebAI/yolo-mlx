# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 MLX - Pure MLX Implementation of YOLO26

A high-performance implementation of YOLO26 using Apple's MLX framework
for efficient inference and training on Apple Silicon.
"""

__version__ = "0.2.0"
__author__ = "YOLO26 MLX Team"

__all__ = ["YOLO", "__version__"]

# Configure package-level logging so logger.info() messages are visible
# when using the Python API directly (CLI configures its own logging).
import logging as _logging

_logger = _logging.getLogger(__name__)
if not _logger.handlers and not _logging.root.handlers:
    _handler = _logging.StreamHandler()
    _handler.setFormatter(_logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(_logging.INFO)


def __getattr__(name: str):
    """Lazily import heavy modules only when needed."""
    if name == "YOLO":
        from yolo26mlx.engine.model import YOLO

        return YOLO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
