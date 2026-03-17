# Copyright (c) 2026 webAI, Inc.
"""YOLO26 MLX converters package.

This module intentionally avoids eagerly importing ``convert`` to prevent
``python -m yolo26mlx.converters.convert`` runtime warnings from ``runpy``.
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "convert_yolo26_weights",
    "load_converted_weights",
    "verify_conversion",
    "convert_conv_weight",
    "convert_conv_transpose_weight",
]

if TYPE_CHECKING:
    from .convert import (
        convert_conv_transpose_weight,
        convert_conv_weight,
        convert_yolo26_weights,
        load_converted_weights,
        verify_conversion,
    )


def __getattr__(name: str) -> Any:
    """Lazily expose converter helpers from ``convert`` submodule."""
    if name in __all__:
        from . import convert as _convert

        return getattr(_convert, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
