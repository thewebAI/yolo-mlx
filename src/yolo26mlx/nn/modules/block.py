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

import math

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


class Proto(nn.Module):
    """Mask prototype generation module.

    Reference: ultralytics Proto class in nn/modules/block.py

    MLX specifics:
    - Uses ConvTranspose2d for learned 2x upsampling (matches Ultralytics)
    - NHWC format throughout
    """

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize Proto module.

        Args:
            c1: Input channels
            c_: Intermediate channels
            c2: Output channels (number of prototypes)
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, kernel_size=2, stride=2, padding=0, bias=True)
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def __call__(self, x: mx.array) -> mx.array:
        """Generate mask prototypes from input feature map.

        Args:
            x: Feature map (B, H, W, C) in NHWC format.

        Returns:
            Prototypes (B, H*2, W*2, c2) after learned upsample + convolutions.
        """
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Proto26(Proto):
    """YOLO26 multi-scale mask prototype module.

    Reference: ultralytics Proto26 class in nn/modules/block.py

    Fuses P3 + P4 + P5 features before generating prototypes, producing
    higher-quality masks than single-scale Proto. Includes an auxiliary
    semantic segmentation head used only during training.

    MLX specifics:
    - NHWC format: spatial dims at indices 1, 2
    - Uses mx.repeat for nearest-neighbor upsampling to match P3 spatial size
    - feat_refine stored as dict for MLX parameter tracking
    """

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize Proto26 with multi-scale feature fusion.

        Args:
            ch: Tuple of channel sizes from backbone feature maps (P3, P4, P5)
            c_: Intermediate channels for proto generation
            c2: Output channels (number of prototypes)
            nc: Number of classes for auxiliary semantic segmentation head
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = {f"layer{i}": Conv(x, ch[0], k=1) for i, x in enumerate(ch[1:])}
        self.feat_fuse = Conv(ch[0], c_, k=3)
        from .head import Sequential

        self.semseg = Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def __call__(self, x: list[mx.array]) -> mx.array | tuple:
        """Generate prototypes from multi-scale features.

        Args:
            x: List of feature maps [P3, P4, P5] in NHWC format.

        Returns:
            During inference: prototypes (B, H_proto, W_proto, c2).
            During training: tuple of (prototypes, semseg_logits).
        """
        feat = x[0]
        for i in range(len(self.feat_refine)):
            up_feat = self.feat_refine[f"layer{i}"](x[i + 1])
            sh = feat.shape[1] // up_feat.shape[1]
            sw = feat.shape[2] // up_feat.shape[2]
            if sh > 1 or sw > 1:
                up_feat = mx.repeat(mx.repeat(up_feat, sh, axis=1), sw, axis=2)
            feat = feat + up_feat
        p = super().__call__(self.feat_fuse(feat))
        if self.training and self.semseg is not None:
            return (p, self.semseg(feat))
        return p

    def fuse(self):
        """Strip the semantic segmentation head for inference."""
        self.semseg = None


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


class RealNVP(nn.Module):
    """RealNVP flow-based density model for keypoint residual scoring.

    Reference: ultralytics RealNVP class in nn/modules/block.py
    (https://arxiv.org/abs/1605.08803,
    https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py)

    Used by the YOLO26 Pose26 head during training to evaluate the
    log-likelihood of predicted keypoint residuals (residual log-likelihood
    estimation). It is not exercised during inference.

    MLX specifics:
    - loc/cov/mask kept as plain mx.array buffers (loaded from weights, frozen).
    - The s/t coupling networks are stored as dicts for MLX parameter tracking.
    - The base prior is a 2-D standard normal (loc=0, cov=I), so its log density
      is evaluated in closed form rather than via a distribution object.
    """

    def __init__(self):
        """Initialize coupling networks and prior buffers."""
        super().__init__()
        from .head import Sequential

        self.loc = mx.zeros((2,))
        self.cov = mx.eye(2)
        self.mask = mx.array([[0.0, 1.0], [1.0, 0.0]] * 3)  # (6, 2) alternating masks
        n = self.mask.shape[0]

        self.s = {
            f"layer{i}": Sequential(
                nn.Linear(2, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 2),
                nn.Tanh(),
            )
            for i in range(n)
        }
        self.t = {
            f"layer{i}": Sequential(
                nn.Linear(2, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 2),
            )
            for i in range(n)
        }

    def _backward_p(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Map data-space samples to latent space and accumulate log|det J|.

        Args:
            x: Keypoint residual samples (N, 2) in data space.

        Returns:
            Tuple of latent samples (N, 2) and log-determinant of the Jacobian (N,).
        """
        n = self.mask.shape[0]
        log_det_jacob = mx.zeros((x.shape[0],))
        z = x
        for i in reversed(range(n)):
            mask_i = self.mask[i]
            z_ = mask_i * z
            s = self.s[f"layer{i}"](z_) * (1 - mask_i)
            t = self.t[f"layer{i}"](z_) * (1 - mask_i)
            z = (1 - mask_i) * (z - t) * mx.exp(-s) + z_
            log_det_jacob = log_det_jacob - mx.sum(s, axis=1)
        return z, log_det_jacob

    def log_prob(self, x: mx.array) -> mx.array:
        """Compute the log probability of keypoint residual samples.

        Args:
            x: Keypoint residual samples (N, 2) in data space.

        Returns:
            Log probability (N,) under the flow-transformed standard-normal prior.
        """
        z, log_det = self._backward_p(x)
        # 2-D standard-normal prior (loc=0, cov=I): log p(z) in closed form.
        prior_ll = -0.5 * mx.sum(z**2, axis=1) - math.log(2.0 * math.pi)
        return prior_ll + log_det
