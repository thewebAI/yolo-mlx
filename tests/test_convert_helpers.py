# Copyright (c) 2026 webAI, Inc.
"""Tests for weight-conversion helper behavior."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def _install_fake_mlx() -> None:
    """Install a minimal fake mlx module for converter-helper tests."""
    if "mlx.core" in sys.modules:
        return

    core = types.ModuleType("mlx.core")
    core.array = np.asarray

    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = core

    sys.modules["mlx"] = mlx_pkg
    sys.modules["mlx.core"] = core


def _load_convert_module():
    """Import converter module after installing fake mlx."""
    _install_fake_mlx()
    return importlib.import_module("yolo26mlx.converters.convert")


def test_convert_conv_weight_transposes_oihw_to_ohwi() -> None:
    """Conv weights should transpose from OIHW to OHWI."""
    convert = _load_convert_module()
    x = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    y = convert.convert_conv_weight(x)
    assert y.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(y, np.transpose(x, (0, 2, 3, 1)))


def test_convert_conv_transpose_weight_transposes_iohw_to_ohwi() -> None:
    """ConvTranspose weights should transpose from IOHW to OHWI."""
    convert = _load_convert_module()
    x = np.arange(3 * 2 * 4 * 5).reshape(3, 2, 4, 5)
    y = convert.convert_conv_transpose_weight(x)
    assert y.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(y, np.transpose(x, (1, 2, 3, 0)))


def test_conv_weight_pattern_detection() -> None:
    """Known YOLO conv names should be detected."""
    convert = _load_convert_module()
    assert convert.is_conv_weight("model.0.conv.weight", (16, 3, 3, 3))
    assert convert.is_conv_weight("model.22.cv2.0.0.conv.weight", (64, 64, 3, 3))
    assert not convert.is_conv_weight("model.0.bn.bias", (64,))


def test_conv_transpose_pattern_detection() -> None:
    """Known ConvTranspose names should be detected."""
    convert = _load_convert_module()
    assert convert.is_conv_transpose_weight("model.23.upsample.weight", (64, 64, 2, 2))
    assert not convert.is_conv_transpose_weight("model.0.conv.weight", (16, 3, 3, 3))
