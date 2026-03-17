# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 Transformer Modules - Pure MLX Implementation

Transformer-based modules for YOLO26.
"""

import mlx.core as mx
import mlx.nn as nn

from yolo26mlx.nn.modules.conv import Conv


class TransformerLayer(nn.Module):
    """Transformer encoder layer with multi-head self-attention and feed-forward network."""

    def __init__(self, c: int, num_heads: int):
        """Initialize transformer layer.

        Args:
            c: Number of channels
            num_heads: Number of attention heads
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiHeadAttention(c, num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward through self-attention and feed-forward network with residual connections.

        Args:
            x: Input tensor (B, N, C) where N is the sequence length.

        Returns:
            Output tensor (B, N, C) after self-attention + FFN, each with a residual connection.
        """
        # MLX MultiHeadAttention returns a single array (not a tuple like PyTorch)
        x = self.ma(self.q(x), self.k(x), self.v(x)) + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    """Transformer block with optional conv projection and multiple transformer layers."""

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        """Initialize transformer block.

        Args:
            c1: Input channels
            c2: Output channels
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def __call__(self, x: mx.array) -> mx.array:
        """Forward through optional conv projection, then transformer layers.

        Args:
            x: Input tensor (B, H, W, C1) in NHWC format.

        Returns:
            Output tensor (B, H, W, C2) after optional channel projection and transformer encoding.
        """
        if self.conv is not None:
            x = self.conv(x)
        b, h, w, c = x.shape
        x = mx.reshape(x, (b, h * w, c))
        x = self.linear(x)
        x = self.tr(x)
        x = mx.reshape(x, (b, h, w, c))
        return x
