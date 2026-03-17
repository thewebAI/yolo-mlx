# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 MLX Neural Network Package

Neural network modules for YOLO26 implementation.
"""

from yolo26mlx.nn.modules import (
    C2PSA,
    C3,
    DFL,
    OBB,
    SPPF,
    A2C2f,
    AAttn,
    ABlock,
    Attention,
    Bottleneck,
    C2f,
    C3k,
    C3k2,
    Concat,
    Conv,
    ConvTranspose2d,
    Detect,
    DWConv,
    Pose,
    PSABlock,
    Segment,
)
from yolo26mlx.nn.tasks import DetectionModel, ModuleList, Sequential, build_model

__all__ = [
    # Convolutions
    "Conv",
    "DWConv",
    "ConvTranspose2d",
    "Concat",
    # Blocks
    "Bottleneck",
    "C2f",
    "C3",
    "C3k",
    "C3k2",
    "SPPF",
    "DFL",
    "C2PSA",
    # Attention
    "Attention",
    "PSABlock",
    "AAttn",
    "ABlock",
    "A2C2f",
    # Heads
    "Detect",
    "Segment",
    "Pose",
    "OBB",
    # Tasks
    "DetectionModel",
    "ModuleList",
    "Sequential",
    "build_model",
]
