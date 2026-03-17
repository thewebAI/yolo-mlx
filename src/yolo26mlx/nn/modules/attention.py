# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 Attention Modules - Pure MLX Implementation

Attention modules for YOLO26 including Area-Attention.
Reference: ultralytics/ultralytics/nn/modules/block.py (Attention, PSABlock classes)

MLX specifics:
- NHWC format: Input shape (B, H, W, C) vs PyTorch NCHW
- Matrix operations adjusted for channels-last format
"""

import mlx.core as mx
import mlx.nn as nn

from .conv import Conv


class Attention(nn.Module):
    """Multi-head self-attention module.

    Reference: ultralytics Attention class
    Self-attention with positional encoding via depthwise convolution.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize Attention module.

        Args:
            dim: Input dimension (channels)
            num_heads: Number of attention heads
            attn_ratio: Ratio of key dimension to head dimension
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5

        # Q, K (both key_dim), V (head_dim) per head
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2  # total: V(dim) + Q(nh_kd) + K(nh_kd)

        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # Depthwise for positional encoding

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with self-attention.

        Args:
            x: Input tensor (B, H, W, C) NHWC format

        Returns:
            Output tensor (B, H, W, C)
        """
        B, H, W, C = x.shape
        N = H * W

        qkv = self.qkv(x)  # (B, H, W, h)

        # Reshape to (B, N, heads, key_dim*2 + head_dim)
        qkv = mx.reshape(qkv, (B, N, self.num_heads, self.key_dim * 2 + self.head_dim))

        # Split Q, K, V
        q = qkv[:, :, :, : self.key_dim]  # (B, N, heads, key_dim)
        k = qkv[:, :, :, self.key_dim : self.key_dim * 2]  # (B, N, heads, key_dim)
        v = qkv[:, :, :, self.key_dim * 2 :]  # (B, N, heads, head_dim)

        # Transpose to (B, heads, N, dim) for attention
        q = mx.transpose(q, (0, 2, 1, 3))  # (B, heads, N, key_dim)
        k = mx.transpose(k, (0, 2, 1, 3))  # (B, heads, N, key_dim)
        v = mx.transpose(v, (0, 2, 1, 3))  # (B, heads, N, head_dim)

        # Attention: q @ k.T -> (B, heads, N, key_dim) @ (B, heads, key_dim, N) = (B, heads, N, N)
        attn = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = mx.softmax(attn, axis=-1)

        # Apply attention to values: attn @ v -> (B, heads, N, N) @ (B, heads, N, head_dim) = (B, heads, N, head_dim)
        out = attn @ v  # (B, heads, N, head_dim)

        # Reshape back to (B, H, W, C)
        out = mx.transpose(out, (0, 2, 1, 3))  # (B, N, heads, head_dim)
        out = mx.reshape(out, (B, H, W, C))

        # Positional encoding
        v_spatial = mx.transpose(v, (0, 2, 1, 3))  # (B, N, heads, head_dim)
        v_spatial = mx.reshape(v_spatial, (B, H, W, C))
        out = out + self.pe(v_spatial)

        return self.proj(out)


class PSABlock(nn.Module):
    """Position-Sensitive Attention Block.

    Reference: ultralytics PSABlock class
    Attention + FFN with residual connections.

    Uses list-based ffn to match PyTorch's nn.Sequential naming (ffn.0, ffn.1).
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True):
        """Initialize PSABlock.

        Args:
            c: Number of channels
            attn_ratio: Attention dimension ratio
            num_heads: Number of attention heads
            shortcut: Use residual connections
        """
        super().__init__()
        self.attn = Attention(c, num_heads, attn_ratio)
        # Use list for proper parameter naming: ffn.0, ffn.1 (matches PyTorch nn.Sequential)
        self.ffn = [
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False),
        ]
        self.add = shortcut

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: x + attn(x), then x + ffn(x).

        Args:
            x: Input tensor (B, H, W, C) in NHWC format.

        Returns:
            Output tensor (B, H, W, C) after self-attention and feed-forward network with residual connections.
        """
        if self.add:
            x = x + self.attn(x)
            ffn_out = self.ffn[0](x)
            ffn_out = self.ffn[1](ffn_out)
            x = x + ffn_out
        else:
            x = self.attn(x)
            x = self.ffn[0](x)
            x = self.ffn[1](x)
        return x


class AAttn(nn.Module):
    """Area-Attention module for YOLO26.

    Reference: ultralytics AAttn class
    YOLO26-specific: Splits spatial dimensions into areas for efficient attention.
    Uses 7x7 kernel with groups=dim for positional encoding (depthwise).
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize AAttn module.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            area: Number of areas to split spatial dimensions into
        """
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        # 7x7 kernel, groups=dim (depthwise) for positional encoding - matches ultralytics
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with area-based attention.

        Args:
            x: Input tensor (B, H, W, C) in NHWC format.

        Returns:
            Output tensor (B, H, W, C) after area-partitioned multi-head attention with positional encoding.
        """
        B, H, W, C = x.shape
        N = H * W

        # qkv: (B, H, W, 3*C) -> (B, N, 3*C)
        qkv = self.qkv(x)
        qkv = mx.reshape(qkv, (B, N, C * 3))

        # Area-based: reshape to process areas independently
        if self.area > 1:
            qkv = mx.reshape(qkv, (B * self.area, N // self.area, C * 3))
            B_eff = B * self.area
            N_eff = N // self.area
        else:
            B_eff = B
            N_eff = N

        # Reshape for attention: (B_eff, N_eff, num_heads, 3*head_dim)
        qkv = mx.reshape(qkv, (B_eff, N_eff, self.num_heads, self.head_dim * 3))
        # Permute to (B_eff, num_heads, 3*head_dim, N_eff)
        qkv = mx.transpose(qkv, (0, 2, 3, 1))

        # Split q, k, v
        q = qkv[:, :, : self.head_dim, :]  # (B_eff, heads, head_dim, N_eff)
        k = qkv[:, :, self.head_dim : 2 * self.head_dim, :]  # (B_eff, heads, head_dim, N_eff)
        v = qkv[:, :, 2 * self.head_dim :, :]  # (B_eff, heads, head_dim, N_eff)

        # Attention: (q.T @ k) = (N_eff, head_dim) @ (head_dim, N_eff) = (N_eff, N_eff)
        # q: (B_eff, heads, head_dim, N_eff) -> q.T: (B_eff, heads, N_eff, head_dim)
        attn = (mx.transpose(q, (0, 1, 3, 2)) @ k) * self.scale  # (B_eff, heads, N_eff, N_eff)
        attn = mx.softmax(attn, axis=-1)

        # Apply attention: v @ attn.T = (head_dim, N_eff) @ (N_eff, N_eff) = (head_dim, N_eff)
        # But attn.T is (N_eff, N_eff), so v @ attn.T -> (B_eff, heads, head_dim, N_eff)
        out = v @ mx.transpose(attn, (0, 1, 3, 2))  # (B_eff, heads, head_dim, N_eff)

        # Permute: (B_eff, heads, head_dim, N_eff) -> (B_eff, N_eff, heads, head_dim)
        out = mx.transpose(out, (0, 3, 1, 2))
        v_out = mx.transpose(v, (0, 3, 1, 2))

        # Reshape back if area-based
        if self.area > 1:
            out = mx.reshape(out, (B, N, C))
            v_out = mx.reshape(v_out, (B, N, C))
        else:
            out = mx.reshape(out, (B_eff, N_eff, C))
            v_out = mx.reshape(v_out, (B_eff, N_eff, C))

        # Reshape to spatial: (B, H, W, C) for NHWC
        out = mx.reshape(out, (B, H, W, C))
        v_out = mx.reshape(v_out, (B, H, W, C))

        # Positional encoding and projection
        out = out + self.pe(v_out)
        return self.proj(out)


class ABlock(nn.Module):
    """Area-Attention Block for YOLO26.

    Reference: ultralytics ABlock class
    YOLO26-specific: Combines AAttn with MLP (using Conv layers).
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize ABlock.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            area: Number of areas for attention
        """
        super().__init__()
        self.attn = AAttn(dim, num_heads, area)
        mlp_hidden = int(dim * mlp_ratio)
        # Use dict for proper MLX parameter tracking
        self.mlp = {
            "cv1": Conv(dim, mlp_hidden, 1),
            "cv2": Conv(mlp_hidden, dim, 1, act=False),
        }

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: x + attn(x), x + mlp(x).

        Args:
            x: Input tensor (B, H, W, C) in NHWC format.

        Returns:
            Output tensor (B, H, W, C) after area-attention and MLP with residual connections.
        """
        x = x + self.attn(x)
        mlp_out = self.mlp["cv1"](x)
        mlp_out = self.mlp["cv2"](mlp_out)
        return x + mlp_out


class A2C2f(nn.Module):
    """Area-Attention C2f module for YOLO26.

    Reference: ultralytics A2C2f class
    YOLO26-specific: C2f variant with Area-Attention blocks.
    Includes optional learnable residual scaling (gamma).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize A2C2f module.

        Args:
            c1: Input channels
            c2: Output channels
            n: Number of blocks
            a2: Use area-attention blocks
            area: Number of areas for attention
            residual: Use learnable residual connection
            mlp_ratio: MLP expansion ratio
            e: Channel expansion ratio
            g: Groups for C3k blocks
            shortcut: Use shortcut in C3k
        """
        super().__init__()
        self.c = int(c2 * e)
        self.n = n

        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((1 + n) * self.c, c2, 1)

        # Learnable residual scaling
        self.residual = a2 and residual and c1 == c2
        if self.residual:
            self.gamma = mx.ones((c2,)) * 0.01
        else:
            self.gamma = None

        # Build blocks - use dict for proper MLX parameter tracking
        # In ultralytics: nn.Sequential(*(ABlock(...) for _ in range(2))) per n
        if a2:
            # Use ABlock for area-attention: 2 ABlocks per module, n modules
            num_heads = self.c // 32  # matches ultralytics
            self.m = {
                f"block{i}": {
                    "ab0": ABlock(self.c, num_heads, mlp_ratio, area),
                    "ab1": ABlock(self.c, num_heads, mlp_ratio, area),
                }
                for i in range(n)
            }
            self.use_a2 = True
        else:
            # Fall back to C3k from block module
            from .block import C3k

            self.m = {f"block{i}": C3k(self.c, self.c, 2, shortcut, g) for i in range(n)}
            self.use_a2 = False

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: cv1 -> blocks -> concat -> cv2.

        Args:
            x: Input tensor (B, H, W, C1) in NHWC format.

        Returns:
            Output tensor (B, H, W, C2) after projection, area-attention blocks, concatenation, and optional residual scaling.
        """
        identity = x
        y = [self.cv1(x)]

        for i in range(self.n):
            block = self.m[f"block{i}"]
            if self.use_a2:
                # Two ABlocks in sequence
                current = block["ab0"](y[-1])
                current = block["ab1"](current)
            else:
                # Single C3k block
                current = block(y[-1])
            y.append(current)

        out = self.cv2(mx.concatenate(y, axis=-1))

        if self.residual and self.gamma is not None:
            # NHWC format: reshape gamma to (1, 1, 1, c2)
            gamma = mx.reshape(self.gamma, (1, 1, 1, -1))
            return identity + gamma * out
        return out
