# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 Convolution Modules - Pure MLX Implementation

Convolution building blocks for YOLO26.
Reference: ultralytics/ultralytics/nn/modules/conv.py

MLX specifics:
- Uses NHWC format (channels-last) vs PyTorch NCHW
- Conv2d supports: in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
- BatchNorm takes num_features as first arg
"""

import mlx.core as mx
import mlx.nn as nn


def autopad(k: int | tuple[int, int], p: int | None = None, d: int = 1) -> int | tuple[int, int]:
    """Compute padding for 'same' convolution.

    Reference: ultralytics conv.py autopad function

    Args:
        k: Kernel size (int or tuple)
        p: Padding (None for auto-compute)
        d: Dilation

    Returns:
        Padding value(s) for 'same' spatial dimensions
    """
    if d > 1:
        # Compute actual kernel size with dilation
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # Auto-pad for 'same' output
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation.

    Reference: ultralytics Conv class
    - Conv2d (no bias) -> BatchNorm2d -> SiLU (default)

    MLX Conv2d format: NHWC (batch, height, width, channels)
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ):
        """Initialize Conv layer.

        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding (None for auto-compute 'same' padding)
            g: Groups for grouped convolution
            d: Dilation
            act: Activation (True=SiLU, False=Identity, or nn.Module instance)
        """
        super().__init__()

        # MLX Conv2d supports: in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        self.conv = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            dilation=d,
            groups=g,
            bias=False,  # BatchNorm handles bias
        )
        # MLX BatchNorm: eps=1e-3 and momentum=0.03 to match PyTorch's initialize_weights()
        # (PyTorch overrides all BN layers after construction in torch_utils.py L463-471)
        self.bn = nn.BatchNorm(num_features=c2, eps=1e-3, momentum=0.03, affine=True)

        # Activation setup - create new instance for each Conv to avoid shared state
        if act is True:
            self.act = nn.SiLU()  # Create fresh instance
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: Conv -> BatchNorm -> Activation.

        Args:
            x: Input tensor (B, H, W, C) in NHWC format.

        Returns:
            Output tensor (B, H', W', C2) after convolution, batch norm, and activation.
        """
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

    def forward_fuse(self, x: mx.array) -> mx.array:
        """Forward pass with fused conv+bn (for inference optimization).

        Args:
            x: Input tensor (B, H, W, C) in NHWC format.

        Returns:
            Output tensor (B, H', W', C2) after fused convolution and activation (no separate BN).
        """
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DWConv(Conv):
    """Depthwise convolution.

    Reference: ultralytics DWConv class
    Groups = input channels for depthwise operation.
    """

    def __init__(
        self, c1: int, c2: int, k: int = 1, s: int = 1, d: int = 1, act: bool | nn.Module = True
    ):
        """Initialize depthwise convolution with groups=c1.

        Args:
            c1: Input channels (also used as groups)
            c2: Output channels
            k: Kernel size
            s: Stride
            d: Dilation
            act: Activation
        """
        # Depthwise: groups = input channels
        super().__init__(c1, c2, k, s, g=c1, d=d, act=act)


class ConvTranspose2d(nn.Module):
    """Transposed convolution with BatchNorm and activation.

    Reference: Used in ultralytics Proto module for upsampling.
    MLX ConvTranspose2d format: NHWC
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 2,
        s: int = 2,
        p: int = 0,
        act: bool | nn.Module = True,
    ):
        """Initialize transposed convolution.

        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride (upsampling factor)
            p: Padding
            act: Activation
        """
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=True,  # Bias typically included in transposed conv
        )
        # eps=1e-3 and momentum=0.03 to match PyTorch's initialize_weights()
        self.bn = nn.BatchNorm(num_features=c2, eps=1e-3, momentum=0.03, affine=True)

        if act is True:
            self.act = nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: ConvTranspose -> BatchNorm -> Activation.

        Args:
            x: Input tensor (B, H, W, C) in NHWC format.

        Returns:
            Upsampled output tensor (B, H*s, W*s, C2) after transposed convolution, batch norm, and activation.
        """
        x = self.conv_transpose(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Reference: ultralytics Concat class

    Note: MLX uses NHWC format, so dimension=1 (PyTorch channel dim) maps to dimension=-1 (MLX channel dim)
    """

    def __init__(self, dimension: int = 1):
        """Initialize Concat module.

        Args:
            dimension: Dimension along which to concatenate.
                       In YAML configs, dimension=1 means channel dimension (PyTorch NCHW).
                       We convert this to axis=-1 for MLX NHWC format.
        """
        super().__init__()
        # Convert PyTorch dimension (NCHW) to MLX dimension (NHWC)
        # PyTorch dim 1 (channels) -> MLX dim -1 (channels)
        self.d = -1 if dimension == 1 else dimension

    def __call__(self, x: list) -> mx.array:
        """Concatenate input tensors along specified dimension.

        Args:
            x: List of input tensors

        Returns:
            Concatenated tensor
        """
        return mx.concatenate(x, axis=self.d)
