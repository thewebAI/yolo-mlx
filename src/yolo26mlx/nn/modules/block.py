# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 Block Modules - Pure MLX Implementation

Core building blocks including Bottleneck, C2f, C3k2, SPPF, DFL.
Reference: ultralytics/ultralytics/nn/modules/block.py

MLX specifics:
- NHWC format: concatenation on axis=-1 (channels)
- PyTorch chunk(2, 1) -> MLX split along axis=-1
"""

import mlx.core as mx
import mlx.nn as nn

from .conv import Conv


class Bottleneck(nn.Module):
    """Standard bottleneck block.

    Reference: ultralytics Bottleneck class
    Two convolutions with optional residual connection.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        """Initialize Bottleneck.

        Args:
            c1: Input channels
            c2: Output channels
            shortcut: Use residual connection if c1 == c2
            g: Groups for second convolution
            k: Kernel sizes for (cv1, cv2)
            e: Expansion ratio for hidden channels
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: cv1 -> cv2, with optional residual.

        Args:
            x: Input tensor (B, H, W, C) in NHWC format.

        Returns:
            Output tensor (B, H, W, C2) after two convolutions, added to input if shortcut is enabled.
        """
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """Faster CSP Bottleneck with 2 convolutions.

    Reference: ultralytics C2f class
    Split input, apply bottlenecks to one branch, concatenate all.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f block.

        Args:
            c1: Input channels
            c2: Output channels
            n: Number of Bottleneck blocks
            shortcut: Use shortcut in Bottleneck
            g: Groups for Bottleneck
            e: Expansion ratio
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # concat all branches
        self.m = [Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: split -> bottlenecks -> concat -> cv2.

        Args:
            x: Input tensor (B, H, W, C1) in NHWC format.

        Returns:
            Output tensor (B, H, W, C2) after channel split, sequential bottleneck processing, and concatenation.
        """
        # MLX NHWC: split on channel axis (-1)
        y = self.cv1(x)
        # Split into two equal parts along channel axis
        y0, y1 = mx.split(y, 2, axis=-1)
        outputs = [y0, y1]

        # Apply each bottleneck to the last output
        current = y1
        for m in self.m:
            current = m(current)
            outputs.append(current)

        return self.cv2(mx.concatenate(outputs, axis=-1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions.

    Reference: ultralytics C3 class
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C3 block.

        Args:
            c1: Input channels
            c2: Output channels
            n: Number of Bottleneck blocks
            shortcut: Use shortcut in Bottleneck
            g: Groups
            e: Expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = [Bottleneck(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: parallel paths -> concat -> cv3.

        Args:
            x: Input tensor (B, H, W, C1) in NHWC format.

        Returns:
            Output tensor (B, H, W, C2) after parallel cv1+bottleneck and cv2 paths are concatenated and projected.
        """
        # Apply cv1 -> bottlenecks, cv2 in parallel, concat
        branch1 = self.cv1(x)
        for m in self.m:
            branch1 = m(branch1)
        branch2 = self.cv2(x)
        return self.cv3(mx.concatenate([branch1, branch2], axis=-1))


class C3k(C3):
    """C3 with customizable kernel size.

    Reference: ultralytics C3k class
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ):
        """Initialize C3k with custom kernel size k.

        Args:
            c1: Input channels.
            c2: Output channels.
            n: Number of Bottleneck blocks.
            shortcut: Use residual connection in Bottleneck.
            g: Groups for convolution.
            e: Expansion ratio for hidden channels.
            k: Kernel size for Bottleneck convolutions (used as (k, k)).
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # Override with custom kernel size
        self.m = [Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)]


class BottleneckPSA(nn.Module):
    """Sequential block with Bottleneck followed by PSABlock.

    Used in C3k2 with attn=True for YOLO26.
    """

    def __init__(self, c: int, shortcut: bool = True, g: int = 1):
        """Initialize BottleneckPSA.

        Args:
            c: Channels (input == output)
            shortcut: Use shortcut in bottleneck
            g: Groups
        """
        super().__init__()
        from .attention import PSABlock

        # Matches PyTorch: nn.Sequential(Bottleneck(...), PSABlock(...))
        self.layers = [
            Bottleneck(c, c, shortcut, g),
            PSABlock(c, attn_ratio=0.5, num_heads=max(c // 64, 1)),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward through bottleneck then PSA.

        Args:
            x: Input tensor (B, H, W, C) in NHWC format.

        Returns:
            Output tensor (B, H, W, C) after bottleneck convolutions and position-sensitive attention.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class C3k2(C2f):
    """C3k2 block - C2f variant with optional C3k or attention blocks.

    Reference: ultralytics C3k2 class
    YOLO26 extension: optional attention blocks via attn parameter.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 block.

        Args:
            c1: Input channels
            c2: Output channels
            n: Number of blocks
            c3k: Use C3k blocks instead of Bottleneck
            e: Expansion ratio
            attn: Use attention blocks (Bottleneck + PSABlock)
            g: Groups
            shortcut: Use shortcut connections
        """
        super().__init__(c1, c2, n, shortcut, g, e)

        # Override blocks based on mode
        if attn:
            # YOLO26 attention mode: Sequential(Bottleneck, PSABlock)
            self.m = [BottleneckPSA(self.c, shortcut, g) for _ in range(n)]
        elif c3k:
            self.m = [C3k(self.c, self.c, 2, shortcut, g) for _ in range(n)]
        else:
            self.m = [Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)]


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast.

    Reference: ultralytics SPPF class
    Multiple sequential maxpools for multi-scale features.

    YOLO26 enhancements:
    - n parameter for variable pooling iterations
    - shortcut option for residual connection
    - act=False in cv1
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 5,
        n: int = 3,
        shortcut: bool = False,
    ):
        """Initialize SPPF block.

        Args:
            c1: Input channels
            c2: Output channels
            k: MaxPool kernel size
            n: Number of pooling iterations
            shortcut: Use residual connection
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)  # act=False in YOLO26
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        # MLX MaxPool2d: kernel_size, stride, padding
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: sequential pooling -> concat -> cv2.

        Args:
            x: Input tensor (B, H, W, C1) in NHWC format.

        Returns:
            Output tensor (B, H, W, C2) after multi-scale max-pooling concatenation, with optional residual.
        """
        identity = x
        x = self.cv1(x)

        # Sequential pooling
        outputs = [x]
        for _ in range(self.n):
            x = self.m(x)
            outputs.append(x)

        y = self.cv2(mx.concatenate(outputs, axis=-1))

        return identity + y if self.add else y


class DFL(nn.Module):
    """Distribution Focal Loss layer.

    Reference: ultralytics DFL class
    Integral module for distributional box regression.

    Note: YOLO26 uses reg_max=1 by default (no DFL distribution),
    but this is kept for compatibility with other reg_max values.
    """

    def __init__(self, c1: int = 16):
        """Initialize DFL layer.

        Args:
            c1: Number of distribution bins (reg_max)
        """
        super().__init__()
        self.c1 = c1
        # 1x1 conv to integrate distribution
        self.conv = nn.Conv2d(c1, 1, kernel_size=1, bias=False)
        # Initialize weight as [0, 1, 2, ..., c1-1] for expectation calculation
        # MLX Conv2d weight shape: (out_channels, kH, kW, in_channels)
        weight = mx.arange(c1, dtype=mx.float32)
        self.conv.weight = mx.reshape(weight, (1, 1, 1, c1))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply DFL to input tensor.

        Args:
            x: Input tensor (b, h, w, 4*reg_max) or (b, 4*reg_max, anchors)

        Returns:
            Box coordinates (b, 4, anchors) or (b, h, w, 4)
        """
        if x.ndim == 3:
            # (b, channels, anchors) format
            b, c, a = x.shape
            # Reshape to (b, 4, reg_max, a), apply softmax, integrate
            x = mx.reshape(x, (b, 4, self.c1, a))
            x = mx.transpose(x, (0, 1, 3, 2))  # (b, 4, a, reg_max)
            x = mx.softmax(x, axis=-1)
            # Weighted sum: expectation of distribution
            weights = mx.arange(self.c1, dtype=mx.float32)
            x = mx.sum(x * weights, axis=-1)  # (b, 4, a)
            return x
        else:
            # (b, h, w, channels) NHWC format
            b, h, w, c = x.shape
            x = mx.reshape(x, (b, h, w, 4, self.c1))
            x = mx.softmax(x, axis=-1)
            weights = mx.arange(self.c1, dtype=mx.float32)
            x = mx.sum(x * weights, axis=-1)  # (b, h, w, 4)
            return x


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction.

    Reference: ultralytics C2PSA class

    This module implements a convolutional block with attention mechanisms to enhance
    feature extraction and processing capabilities. It includes a series of PSABlock
    modules for self-attention and feed-forward operations.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1: Input channels
            c2: Output channels (must equal c1)
            n: Number of PSABlock modules
            e: Expansion ratio
        """
        super().__init__()
        assert c1 == c2, "C2PSA requires c1 == c2"
        self.c = int(c1 * e)
        self.n = n
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # Import PSABlock here to avoid circular imports
        from .attention import PSABlock

        # Number of attention heads based on channel dimension (matches ultralytics: c // 64)
        num_heads = max(self.c // 64, 1)
        # Use dict for proper MLX parameter tracking
        self.m = {
            f"psa{i}": PSABlock(self.c, attn_ratio=0.5, num_heads=num_heads) for i in range(n)
        }

    def __call__(self, x: mx.array) -> mx.array:
        """Process input through C2PSA module.

        Args:
            x: Input tensor (B, H, W, C) NHWC format

        Returns:
            Output tensor (B, H, W, C)
        """
        # Split along channel axis (NHWC: axis=-1)
        y = self.cv1(x)
        a, b = mx.split(y, 2, axis=-1)

        # Apply PSABlock modules to branch b
        for i in range(self.n):
            b = self.m[f"psa{i}"](b)

        # Concatenate and project
        return self.cv2(mx.concatenate([a, b], axis=-1))
