# Copyright (c) 2026 webAI, Inc.
"""
PyTorch to MLX Weight Converter for YOLO26

This module handles conversion of YOLO26 PyTorch weights to MLX format.

Key transformations:
1. Conv2d weights: PyTorch OIHW -> MLX OHWI
   PyTorch: (out_channels, in_channels, H, W)
   MLX:     (out_channels, H, W, in_channels)
   Transpose: (0, 2, 3, 1)

2. ConvTranspose2d weights: PyTorch IOHW -> MLX OHWI
   PyTorch: (in_channels, out_channels, H, W)
   MLX:     (out_channels, H, W, in_channels)
   Transpose: (1, 2, 3, 0)

3. BatchNorm/LayerNorm: No change needed (1D parameters)

4. Linear weights: No change (MLX Linear expects same layout)

5. Parameter naming: Follows MLX Module.parameters() hierarchy

Verified against MLX v0.30.3 documentation:
- Conv2d weight shape: (out_channels, kernel_h, kernel_w, in_channels)
- ConvTranspose2d weight shape: (out_channels, kernel_h, kernel_w, in_channels)

Usage:
    # Convert PyTorch .pt file to MLX .safetensors
    python -m yolo26mlx.converters.convert model.pt -o model.safetensors

    # Or programmatically
    from yolo26mlx.converters.convert import convert_yolo26_weights
    weights = convert_yolo26_weights("yolo26n.pt", "yolo26n.safetensors")
"""

import logging
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def convert_conv_weight(pt_weight: np.ndarray) -> mx.array:
    """Convert PyTorch Conv2d weight to MLX format.

    PyTorch Conv2d: (out_channels, in_channels, H, W) - OIHW
    MLX Conv2d:     (out_channels, H, W, in_channels) - OHWI

    Transpose axes: (0, 2, 3, 1)
    - Axis 0 (out_channels) stays at position 0
    - Axis 2 (H) moves to position 1
    - Axis 3 (W) moves to position 2
    - Axis 1 (in_channels) moves to position 3

    Args:
        pt_weight: NumPy array from PyTorch tensor with shape (O, I, H, W)

    Returns:
        MLX array with shape (O, H, W, I)
    """
    if len(pt_weight.shape) != 4:
        raise ValueError(f"Expected 4D conv weight, got shape {pt_weight.shape}")
    return mx.array(np.transpose(pt_weight, (0, 2, 3, 1)))


def convert_conv_transpose_weight(pt_weight: np.ndarray) -> mx.array:
    """Convert PyTorch ConvTranspose2d weight to MLX format.

    PyTorch ConvTranspose2d: (in_channels, out_channels, H, W) - IOHW
    MLX ConvTranspose2d:     (out_channels, H, W, in_channels) - OHWI

    Note: PyTorch ConvTranspose has (in, out) while regular Conv has (out, in).
    Both MLX Conv2d and ConvTranspose2d use OHWI format.

    Transpose axes: (1, 2, 3, 0)
    - Axis 1 (out_channels) moves to position 0
    - Axis 2 (H) moves to position 1
    - Axis 3 (W) moves to position 2
    - Axis 0 (in_channels) moves to position 3

    Args:
        pt_weight: NumPy array from PyTorch tensor with shape (I, O, H, W)

    Returns:
        MLX array with shape (O, H, W, I)
    """
    if len(pt_weight.shape) != 4:
        raise ValueError(f"Expected 4D conv transpose weight, got shape {pt_weight.shape}")
    return mx.array(np.transpose(pt_weight, (1, 2, 3, 0)))


def convert_bn_weight(pt_weight: np.ndarray) -> mx.array:
    """Convert BatchNorm parameters to MLX format.

    Args:
        pt_weight: NumPy array of BatchNorm parameters (1D).

    Returns:
        MLX array (no transformation needed).
    """
    return mx.array(pt_weight)


def convert_linear_weight(pt_weight: np.ndarray) -> mx.array:
    """Convert Linear layer weight to MLX format.

    Args:
        pt_weight: NumPy array of Linear layer weights.

    Returns:
        MLX array (no transformation needed).
    """
    return mx.array(pt_weight)


# ============================================================================
# YOLO26-specific weight detection patterns
# ============================================================================

# Patterns that indicate Conv2d weights (need OIHW -> OHWI conversion)
CONV_WEIGHT_PATTERNS = [
    # Standard Conv wrapper patterns
    r"\.conv\.weight$",
    r"\.cv[1-4]\.conv\.weight$",
    # Attention module patterns (AAttn)
    r"\.qkv\.conv\.weight$",
    r"\.proj\.conv\.weight$",
    r"\.pe\.conv\.weight$",
    # Detection head patterns (cv2, cv3 with nested indices)
    r"\.cv[234]\.\d+\.\d+\.conv\.weight$",  # e.g., model.22.cv2.0.0.conv.weight
    r"\.cv[234]\.\d+\.\d+\.weight$",  # e.g., model.22.cv2.0.2.weight (final conv)
    # End-to-end detection head patterns
    r"\.one2one_cv[23]\.\d+\.\d+\.conv\.weight$",
    r"\.one2one_cv[23]\.\d+\.\d+\.weight$",
    # DFL module (1x1 conv)
    r"\.dfl\.conv\.weight$",
    # Direct nn.Conv2d in ModuleList
    r"\.m\.\d+\.weight$",
    # MLP patterns in ABlock (mlp.0.conv.weight, mlp.1.conv.weight)
    r"\.mlp\.\d+\.conv\.weight$",
    # Proto/Proto26 patterns
    r"\.cv[1-3]\.conv\.weight$",
]

# Patterns that indicate ConvTranspose2d weights (need IOHW -> OHWI conversion)
CONV_TRANSPOSE_PATTERNS = [
    r"\.upsample\.weight$",  # Proto module
    r"convtranspose.*\.weight$",  # Generic ConvTranspose
    r"conv_transpose.*\.weight$",
]

# Compiled regex for efficiency
_CONV_PATTERN = re.compile("|".join(CONV_WEIGHT_PATTERNS), re.IGNORECASE)
_CONV_TRANSPOSE_PATTERN = re.compile("|".join(CONV_TRANSPOSE_PATTERNS), re.IGNORECASE)


def is_conv_weight(name: str, shape: tuple) -> bool:
    """Check if parameter is a Conv2d weight that needs OIHW->OHWI conversion.

    Args:
        name: Parameter name from state_dict
        shape: Parameter shape tuple

    Returns:
        True if this is a conv weight that needs transposition
    """
    # Must be 4D tensor
    if len(shape) != 4:
        return False

    # Check against known patterns
    if _CONV_PATTERN.search(name):
        return True

    return False


def is_conv_transpose_weight(name: str, shape: tuple) -> bool:
    """Check if parameter is a ConvTranspose2d weight.

    Args:
        name: Parameter name from state_dict
        shape: Parameter shape tuple

    Returns:
        True if this is a conv transpose weight
    """
    if len(shape) != 4:
        return False

    return bool(_CONV_TRANSPOSE_PATTERN.search(name))


def convert_name_pytorch_to_mlx(pt_name: str) -> str:
    """Convert PyTorch parameter name to MLX naming convention.

    MLX Module.parameters() returns nested dict like:
    {
        'model': {
            '0': {
                'conv': {'weight': array},
                'bn': {'weight': array, 'bias': array, ...}
            },
            ...
        }
    }

    For load_weights(), we flatten to dot-separated names.

    Args:
        pt_name: PyTorch state_dict key (e.g., "model.0.conv.weight")

    Returns:
        MLX parameter name (typically same format)
    """
    # Most names can be used directly
    # Special handling only if needed

    # Handle num_batches_tracked (not needed in MLX BatchNorm)
    if "num_batches_tracked" in pt_name:
        return None  # Skip this parameter

    return pt_name


def convert_yolo26_weights(
    pt_path: str, output_path: str | None = None, verbose: bool = True
) -> list[tuple[str, mx.array]]:
    """Convert YOLO26 PyTorch weights to MLX format.

    Args:
        pt_path: Path to PyTorch .pt file
        output_path: Optional path to save MLX weights (.safetensors or .npz)
        verbose: Print conversion progress

    Returns:
        List of (name, array) tuples for MLX load_weights()
    """
    # Note: This requires torch for conversion only
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for weight conversion. " "Install with: pip install torch"
        ) from None

    if verbose:
        logger.info(f"Loading PyTorch weights from {pt_path}")

    # Load PyTorch weights
    pt_checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Extract state dict from various checkpoint formats
    if isinstance(pt_checkpoint, dict):
        if "model" in pt_checkpoint:
            model = pt_checkpoint["model"]
            if hasattr(model, "state_dict"):
                pt_weights = model.float().state_dict()
            else:
                pt_weights = model
        elif "state_dict" in pt_checkpoint:
            pt_weights = pt_checkpoint["state_dict"]
        else:
            pt_weights = pt_checkpoint
    elif hasattr(pt_checkpoint, "state_dict"):
        pt_weights = pt_checkpoint.float().state_dict()
    else:
        pt_weights = pt_checkpoint

    if verbose:
        logger.info(f"Found {len(pt_weights)} parameters in checkpoint")

    mlx_weights = []
    skipped = []
    converted = {"conv": 0, "conv_transpose": 0, "bn": 0, "linear": 0, "other": 0}

    for pt_name, pt_tensor in pt_weights.items():
        # Convert tensor to numpy
        if hasattr(pt_tensor, "numpy"):
            np_tensor = pt_tensor.cpu().numpy()
        else:
            np_tensor = np.array(pt_tensor)

        # Convert name
        mlx_name = convert_name_pytorch_to_mlx(pt_name)
        if mlx_name is None:
            skipped.append(pt_name)
            continue

        # Determine conversion based on tensor shape and name
        shape = np_tensor.shape

        if is_conv_transpose_weight(pt_name, shape):
            # ConvTranspose2d weight
            mlx_tensor = convert_conv_transpose_weight(np_tensor)
            converted["conv_transpose"] += 1
        elif is_conv_weight(pt_name, shape):
            # Regular Conv2d weight
            mlx_tensor = convert_conv_weight(np_tensor)
            converted["conv"] += 1
        elif "bn" in pt_name.lower() or "norm" in pt_name.lower():
            # BatchNorm / LayerNorm
            mlx_tensor = convert_bn_weight(np_tensor)
            converted["bn"] += 1
        elif "linear" in pt_name.lower() or "fc" in pt_name.lower():
            # Linear / Fully Connected
            mlx_tensor = convert_linear_weight(np_tensor)
            converted["linear"] += 1
        elif len(shape) == 4:
            # Unknown 4D tensor - likely a conv weight we didn't match
            # Log warning for debugging but still convert
            if verbose:
                logger.warning(
                    f"  Warning: Unknown 4D tensor '{pt_name}' with shape {shape}, assuming conv weight"
                )
            mlx_tensor = convert_conv_weight(np_tensor)
            converted["conv"] += 1
        else:
            # Other (1D bias, gamma, running_mean, running_var, etc.)
            mlx_tensor = mx.array(np_tensor)
            converted["other"] += 1

        mlx_weights.append((mlx_name, mlx_tensor))

    if verbose:
        logger.info("\nConversion summary:")
        logger.info(f"  Conv weights: {converted['conv']}")
        logger.info(f"  ConvTranspose weights: {converted['conv_transpose']}")
        logger.info(f"  BatchNorm params: {converted['bn']}")
        logger.info(f"  Linear weights: {converted['linear']}")
        logger.info(f"  Other params: {converted['other']}")
        logger.info(f"  Skipped: {len(skipped)}")
        if skipped:
            logger.info(f"  Skipped params: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".safetensors":
            # Save as safetensors
            try:
                from safetensors.numpy import save_file

                # Convert to dict with numpy arrays
                np_dict = {name: np.array(arr) for name, arr in mlx_weights}
                save_file(np_dict, str(output_path))
                if verbose:
                    logger.info(f"\nSaved MLX weights to {output_path} (safetensors)")
            except ImportError:
                logger.warning("Warning: safetensors not installed, falling back to .npz")
                output_path = output_path.with_suffix(".npz")
                # Fall through to npz saving

        if output_path.suffix == ".npz":
            # Save as npz
            # mx.savez expects **kwargs, so we need to convert
            weight_dict = {name: arr for name, arr in mlx_weights}
            mx.savez(str(output_path), **weight_dict)
            if verbose:
                logger.info(f"\nSaved MLX weights to {output_path} (npz)")

    return mlx_weights


def load_converted_weights(
    model, weights_path: str, strict: bool = False, verbose: bool = True
) -> None:
    """Load converted weights into an MLX model.

    Args:
        model: MLX model instance (nn.Module)
        weights_path: Path to .safetensors or .npz file
        strict: Raise error if weights don't match model exactly
        verbose: Print loading info
    """
    if verbose:
        logger.info(f"Loading weights from {weights_path}")

    # MLX Module.load_weights handles both .safetensors and .npz
    model.load_weights(weights_path, strict=strict)

    if verbose:
        logger.info("Weights loaded successfully")


def verify_conversion(pt_path: str, mlx_weights: list[tuple[str, mx.array]]) -> bool:
    """Verify conversion by checking shapes match expected MLX format.

    Args:
        pt_path: Original PyTorch weights path
        mlx_weights: Converted MLX weights

    Returns:
        True if verification passes
    """
    import torch

    pt_checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "model" in pt_checkpoint:
        model = pt_checkpoint["model"]
        if hasattr(model, "state_dict"):
            pt_weights = model.float().state_dict()
        else:
            pt_weights = model
    else:
        pt_weights = pt_checkpoint

    mlx_dict = dict(mlx_weights)

    issues = []
    for pt_name, pt_tensor in pt_weights.items():
        if "num_batches_tracked" in pt_name:
            continue

        mlx_name = convert_name_pytorch_to_mlx(pt_name)
        if mlx_name not in mlx_dict:
            issues.append(f"Missing: {pt_name}")
            continue

        pt_shape = tuple(pt_tensor.shape)
        mlx_shape = mlx_dict[mlx_name].shape

        # Check expected shape based on weight type
        if len(pt_shape) == 4 and is_conv_transpose_weight(pt_name, pt_shape):
            # ConvTranspose: PyTorch (I, O, H, W) -> MLX (O, H, W, I)
            expected_shape = (pt_shape[1], pt_shape[2], pt_shape[3], pt_shape[0])
            if mlx_shape != expected_shape:
                issues.append(
                    f"Shape mismatch for ConvTranspose '{pt_name}': "
                    f"expected {expected_shape}, got {mlx_shape}"
                )
        elif len(pt_shape) == 4 and is_conv_weight(pt_name, pt_shape):
            # Conv: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
            expected_shape = (pt_shape[0], pt_shape[2], pt_shape[3], pt_shape[1])
            if mlx_shape != expected_shape:
                issues.append(
                    f"Shape mismatch for Conv '{pt_name}': "
                    f"expected {expected_shape}, got {mlx_shape}"
                )
        elif len(pt_shape) == 4:
            # Unknown 4D tensor - treated as conv
            expected_shape = (pt_shape[0], pt_shape[2], pt_shape[3], pt_shape[1])
            if mlx_shape != expected_shape:
                issues.append(
                    f"Shape mismatch for unknown 4D '{pt_name}': "
                    f"expected {expected_shape}, got {mlx_shape}"
                )
        elif mlx_shape != pt_shape:
            # Non-4D should match exactly
            issues.append(f"Shape mismatch for {pt_name}: " f"expected {pt_shape}, got {mlx_shape}")

    if issues:
        logger.info("Verification issues:")
        for issue in issues[:10]:
            logger.info(f"  {issue}")
        if len(issues) > 10:
            logger.info(f"  ... and {len(issues) - 10} more")
        return False

    logger.info("Verification passed!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert YOLO26 PyTorch weights to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert to safetensors (recommended)
    python -m yolo26mlx.converters.convert yolo26n.pt -o yolo26n.safetensors
    
    # Convert to npz
    python -m yolo26mlx.converters.convert yolo26n.pt -o yolo26n.npz
    
    # Verify conversion
    python -m yolo26mlx.converters.convert yolo26n.pt -o yolo26n.safetensors --verify
""",
    )
    parser.add_argument("input", help="Path to PyTorch .pt file")
    parser.add_argument("-o", "--output", help="Output path for MLX weights (.safetensors or .npz)")
    parser.add_argument(
        "-v", "--verify", action="store_true", help="Verify conversion by checking shapes"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    verbose = not args.quiet

    # Convert weights
    mlx_weights = convert_yolo26_weights(args.input, args.output, verbose=verbose)

    # Verify if requested
    if args.verify:
        verify_conversion(args.input, mlx_weights)
