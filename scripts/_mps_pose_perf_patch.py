# Copyright (c) 2026 webAI, Inc.
"""
PyTorch MPS compatibility patch for Ultralytics' YOLO26 pose training.

Ultralytics' end-to-end pose loss (``PoseLoss26``) builds its per-keypoint
RLE target weights with::

    self.target_weights = torch.from_numpy(RLE_WEIGHT).to(self.device)

``RLE_WEIGHT`` (``ultralytics.utils.metrics``) is a **float64** numpy array, so
``torch.from_numpy`` yields a float64 tensor. Moving that tensor to an Apple
GPU raises::

    TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS
    framework doesn't support float64. Please use float32 instead.

This makes ``device="mps"`` pose training abort before the first step (CPU is
unaffected because it supports float64). ``OKS_SIGMA`` is already float32 and
needs no change.

The fix casts the module-level ``RLE_WEIGHT`` that ``PoseLoss26`` reads to
float32. The per-keypoint weights are scale factors, so float32 is numerically
inconsequential and keeps the MPS run comparable to the CPU/MLX runs.

Apply once at process start::

    from _mps_pose_perf_patch import apply_mps_pose_patch
    apply_mps_pose_patch()
"""

from __future__ import annotations

import numpy as np
import ultralytics.utils.loss as _ul_loss

_ORIGINAL_RLE_WEIGHT = _ul_loss.RLE_WEIGHT


def apply_mps_pose_patch() -> None:
    """Cast Ultralytics' ``RLE_WEIGHT`` to float32 for MPS pose training.

    MPS cannot host float64 tensors, and the upstream end-to-end pose loss
    moves a float64 ``RLE_WEIGHT`` array to the training device. Casting the
    module-level array to float32 lets ``torch.from_numpy(RLE_WEIGHT).to('mps')``
    succeed. Idempotent — repeated calls have no additional effect.
    """
    if getattr(_ul_loss, "_RLE_WEIGHT_MPS_PATCHED", False):
        return
    if getattr(_ul_loss.RLE_WEIGHT, "dtype", None) == np.float64:
        _ul_loss.RLE_WEIGHT = _ul_loss.RLE_WEIGHT.astype(np.float32)
    _ul_loss._RLE_WEIGHT_MPS_PATCHED = True


def revert_mps_pose_patch() -> None:
    """Restore the original float64 Ultralytics ``RLE_WEIGHT`` array."""
    _ul_loss.RLE_WEIGHT = _ORIGINAL_RLE_WEIGHT
    _ul_loss._RLE_WEIGHT_MPS_PATCHED = False
