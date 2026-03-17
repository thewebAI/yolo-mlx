# Copyright (c) 2026 webAI, Inc.
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 Model Parsing - Pure MLX Implementation

Model architecture parsing from YAML configuration.
Based on ultralytics parse_model but for MLX with NHWC format.
"""

import ast
import contextlib
import logging
import math
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import yaml

# Import modules
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

logger = logging.getLogger(__name__)


def make_divisible(x: float, divisor: int = 8) -> int:
    """Returns nearest x divisible by divisor.

    Args:
        x: Number to make divisible
        divisor: The divisor

    Returns:
        Nearest integer divisible by divisor
    """
    return math.ceil(x / divisor) * divisor


class ModuleList(nn.Module):
    """A module list for holding submodules.

    MLX Note: Uses 'layers' attribute (not '_modules') because MLX skips
    underscore-prefixed attributes when collecting parameters.
    """

    def __init__(self, modules: list[nn.Module] | None = None):
        """Initialize ModuleList with an optional list of modules.

        Args:
            modules: Optional list of nn.Module instances to store, or None for an empty list.
        """
        super().__init__()
        self.layers = modules or []

    def __getitem__(self, idx):
        """Support both integer indexing and string key access for MLX.

        Args:
            idx: Integer index into the module list, or string key for MLX parameter access.

        Returns:
            The module at the given index, or the MLX parameter subtree for string keys.
        """
        if isinstance(idx, str):
            # MLX parameter access - delegate to parent
            return super().__getitem__(idx)
        # Integer indexing for user code
        return self.layers[idx]

    def __len__(self) -> int:
        """Return the number of modules in the list."""
        return len(self.layers)

    def __iter__(self):
        """Iterate over the stored modules."""
        return iter(self.layers)

    def append(self, module: nn.Module):
        """Append a module to the end of the list.

        Args:
            module: The nn.Module instance to append.
        """
        self.layers.append(module)

    def __call__(self, x: mx.array, idx: int) -> mx.array:
        """Forward input through the module at the given index.

        Args:
            x: Input tensor to pass through the selected module.
            idx: Index of the module to execute.

        Returns:
            Output tensor from the module at position idx.
        """
        return self.layers[idx](x)


class Sequential(nn.Module):
    """Sequential container with proper MLX handling.

    MLX Note: Uses 'layers' attribute (not '_modules') because MLX skips
    underscore-prefixed attributes when collecting parameters.
    """

    def __init__(self, *modules: nn.Module):
        """Initialize Sequential with the given layers.

        Args:
            *modules: Variable number of nn.Module layers to execute sequentially.
        """
        super().__init__()
        self.layers = list(modules)

    def __getitem__(self, idx):
        """Support integer indexing, slicing, and string key access for MLX.

        Args:
            idx: Integer index, slice, or string key for MLX parameter access.

        Returns:
            The module at the given index, a new Sequential for slices, or the MLX parameter subtree for string keys.
        """
        if isinstance(idx, str):
            # MLX parameter access - delegate to parent
            return super().__getitem__(idx)
        if isinstance(idx, slice):
            return Sequential(*self.layers[idx])
        return self.layers[idx]

    def __len__(self) -> int:
        """Return the number of layers in the sequence."""
        return len(self.layers)

    def __iter__(self):
        """Iterate over the stored layers."""
        return iter(self.layers)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward input through all layers sequentially.

        Args:
            x: Input tensor passed through each layer in order.

        Returns:
            Output tensor after applying all layers sequentially.
        """
        for module in self.layers:
            x = module(x)
        return x


class DetectionModel(nn.Module):
    """YOLO26 Detection Model.

    This is the main model class that parses YAML config and builds the model.
    Uses NHWC format throughout for MLX optimization.
    """

    _compiled_predict = None  # Compiled prediction function for mx.compile

    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        scale: str | None = None,
    ):
        """Initialize YOLO26 detection model.

        Args:
            cfg: Path to YAML config or config dictionary
            ch: Number of input channels (3 for RGB)
            nc: Number of classes (overrides config if provided)
            verbose: Print model information
            scale: Model scale (n, s, m, l, x) - overrides scale in config
        """
        super().__init__()

        # Load configuration
        if isinstance(cfg, str):
            self.yaml_file = cfg
            cfg = load_model_config(cfg)
        else:
            self.yaml_file = None

        # Store config
        self.cfg = cfg

        # Override scale if provided
        if scale is not None:
            cfg["scale"] = scale

        # Override nc if provided
        if nc is not None:
            cfg["nc"] = nc

        # Parse model
        self.model, self.save = parse_model(cfg, [ch], verbose=verbose)

        # Model attributes from config
        self.nc = cfg.get("nc", 80)
        self.reg_max = cfg.get("reg_max", 1)
        self.end2end = cfg.get("end2end", True)
        self.names = {i: f"class{i}" for i in range(self.nc)}

        # Get stride from last layer (Detect)
        if len(self.model) > 0:
            m = self.model[-1]
            if hasattr(m, "stride"):
                self.stride = m.stride
            else:
                self.stride = mx.array([8.0, 16.0, 32.0])

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor in NHWC format (N, H, W, C)

        Returns:
            Model output
        """
        # Use compiled prediction if available (27-50% faster)
        if self._compiled_predict is not None:
            return self._compiled_predict(x)
        return self._forward_once(x)

    def compile_for_inference(self) -> None:
        """Compile the model's forward pass with mx.compile for faster inference."""
        if self._compiled_predict is not None:
            return  # Already compiled

        def compiled_forward(x):
            """JIT-compiled wrapper around _forward_once for accelerated inference."""
            return self._forward_once(x)

        self._compiled_predict = mx.compile(compiled_forward)

    def disable_compiled_inference(self) -> None:
        """Clear the compiled forward function, reverting to the standard forward pass."""
        self._compiled_predict = None

    def _forward_once(self, x: mx.array) -> mx.array:
        """Forward pass through model.

        Args:
            x: Input tensor in NHWC format

        Returns:
            Output from last layer
        """
        y = []  # outputs

        for m in self.model:
            if hasattr(m, "f"):
                # Multi-input module (Concat, Detect, etc.)
                if m.f != -1:  # if not from previous layer
                    if isinstance(m.f, int):
                        x = y[m.f]
                    else:
                        x = [y[j] if j != -1 else x for j in m.f]

            x = m(x)

            # Save output if needed by later layers
            if hasattr(m, "i"):
                y.append(x if m.i in self.save else None)
            else:
                y.append(x)

        return x


def parse_model(cfg: dict, ch: list[int], verbose: bool = True) -> tuple[ModuleList, list[int]]:
    """Parse YOLO26 model from configuration dictionary.

    Following ultralytics parse_model structure but adapted for MLX NHWC.

    Args:
        cfg: Model configuration dictionary
        ch: List of input channels [ch0] where ch0 is typically 3
        verbose: Print model info

    Returns:
        Tuple of (model layers, save indices)
    """
    # Module mapping
    module_map = {
        "Conv": Conv,
        "DWConv": DWConv,
        "ConvTranspose2d": ConvTranspose2d,
        "Concat": Concat,
        "Bottleneck": Bottleneck,
        "C2f": C2f,
        "C3": C3,
        "C3k": C3k,
        "C3k2": C3k2,
        "SPPF": SPPF,
        "DFL": DFL,
        "C2PSA": C2PSA,
        "Attention": Attention,
        "PSABlock": PSABlock,
        "AAttn": AAttn,
        "ABlock": ABlock,
        "A2C2f": A2C2f,
        "Detect": Detect,
        "Segment": Segment,
        "Pose": Pose,
        "OBB": OBB,
    }

    # Get model parameters
    nc = cfg.get("nc", 80)
    reg_max = cfg.get("reg_max", 1)
    end2end = cfg.get("end2end", True)

    # Model scale parameters (width, depth, max_channels)
    scales = cfg.get("scales", {})
    scale = cfg.get("scale", "n")  # default to nano

    if scale in scales:
        depth, width, max_channels = scales[scale]
    else:
        depth, width, max_channels = 1.0, 1.0, 1024

    # Modules that take c1, c2 as first args
    base_modules = {
        Conv,
        DWConv,
        ConvTranspose2d,
        Bottleneck,
        C2f,
        C3,
        C3k,
        C3k2,
        SPPF,
        C2PSA,
        Attention,
        PSABlock,
        AAttn,
        ABlock,
        A2C2f,
    }

    # Modules that use repeat parameter
    repeat_modules = {C2f, C3, C3k, C3k2, C2PSA, A2C2f}

    layers = []
    save = []  # save indices for concat/detect

    if verbose:
        logger.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    # Process backbone + head
    for i, (f, n, m, args) in enumerate(cfg.get("backbone", []) + cfg.get("head", [])):
        # Get module class
        if m.startswith("nn."):
            # MLX nn module
            m_name = m[3:]
            if hasattr(nn, m_name):
                m = getattr(nn, m_name)
            else:
                raise ValueError(f"Unknown nn module: {m_name}")
        elif m in module_map:
            m = module_map[m]
        else:
            raise ValueError(f"Unknown module: {m}")

        # Process args - evaluate string expressions
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError, SyntaxError):
                    # Try to evaluate as literal or local variable
                    if a == "nc":
                        args[j] = nc
                    elif a == "reg_max":
                        args[j] = reg_max
                    else:
                        args[j] = ast.literal_eval(a)

        # Apply depth scaling
        n_ = max(round(n * depth), 1) if n > 1 else n

        # Build module based on type
        if m in base_modules:
            # Get input/output channels
            c1 = ch[f] if isinstance(f, int) else ch[f[0]]
            c2 = args[0]

            # Scale output channels (except for nc-sized outputs)
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            # Build args
            args = [c1, c2, *args[1:]]

            # Insert repeat count for repeat modules
            if m in repeat_modules:
                args.insert(2, n_)
                n_ = 1

            # Special handling for C3k2 - enable c3k for larger models
            if m is C3k2 and scale in "mlx":
                if len(args) >= 4:
                    args[3] = True  # c3k=True for M/L/X scales

        elif m is nn.Upsample:
            # MLX Upsample: scale_factor, mode
            c2 = ch[f]
            # args[0] is typically None (size), args[1] is scale_factor
            if len(args) >= 2:
                scale_factor = args[1] if args[1] is not None else 2.0
            else:
                scale_factor = 2.0
            args = [float(scale_factor), "nearest"]

        elif m is Concat:
            c2 = sum(ch[x] for x in f)
            args = [1]  # dimension=1 for NCHW style, will be converted to -1 for NHWC

        elif m is Detect:
            # Detect: nc, reg_max, end2end, ch_list
            c2 = None
            ch_list = [ch[x] for x in f]
            args = [nc, reg_max, end2end, ch_list]

        elif m is Segment:
            # Segment: nc, nm, npr, reg_max, end2end, ch_list
            c2 = None
            ch_list = [ch[x] for x in f]
            # args should have [nm, npr, ...]
            nm = args[0] if len(args) > 0 else 32
            npr = args[1] if len(args) > 1 else 256
            npr = make_divisible(min(npr, max_channels) * width, 8)
            args = [nc, nm, npr, reg_max, end2end, ch_list]

        elif m is Pose:
            # Pose: nc, kpt_shape, reg_max, end2end, ch_list
            c2 = None
            ch_list = [ch[x] for x in f]
            kpt_shape = args[0] if len(args) > 0 else (17, 3)
            args = [nc, kpt_shape, reg_max, end2end, ch_list]

        elif m is OBB:
            # OBB: nc, ne, reg_max, end2end, ch_list
            c2 = None
            ch_list = [ch[x] for x in f]
            ne = args[0] if len(args) > 0 else 1
            args = [nc, ne, reg_max, end2end, ch_list]

        else:
            c2 = ch[f] if isinstance(f, int) else ch[f[0]]

        # Create module (repeat n_ times if needed)
        if n_ > 1:
            m_ = Sequential(*[m(*args) for _ in range(n_)])
        else:
            m_ = m(*args)

        # Get module type name
        t = str(m)[8:-2] if "<class" in str(m) else str(m)
        t = t.replace("__main__.", "").split(".")[-1]

        # Count parameters (handle nested dicts from MLX)
        def count_params(params):
            """Recursively count total number of scalar parameters in a nested dict of arrays."""
            total = 0
            for v in params.values():
                if isinstance(v, dict):
                    total += count_params(v)
                elif hasattr(v, "size"):
                    total += v.size
            return total

        np_count = count_params(m_.parameters()) if hasattr(m_, "parameters") else 0

        # Attach metadata
        m_.i = i  # layer index
        m_.f = f  # from index
        m_.type = t
        m_.np = np_count

        if verbose:
            logger.info(f"{i:>3}{str(f):>20}{n_:>3}{np_count:10.0f}  {t:<45}{str(args):<30}")

        # Track which layers need to be saved
        save.extend(x % (i + 1) for x in ([f] if isinstance(f, int) else f) if x != -1)

        layers.append(m_)

        # Update channel list
        if i == 0:
            ch = []
        if c2 is not None:
            ch.append(c2)
        else:
            ch.append(0)  # For Detect/Segment heads

    return ModuleList(layers), sorted(set(save))


def load_model_config(cfg_path: str) -> dict:
    """Load model configuration from YAML file.

    Args:
        cfg_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    cfg_path = Path(cfg_path)

    # Check common config locations.
    search_paths = [
        cfg_path,
        Path(__file__).parent.parent / "cfg" / "models" / "26" / cfg_path.name,
        Path(__file__).parent.parent / "cfg" / "models" / cfg_path.name,
    ]

    # Fallback to ultralytics packaged configs, e.g. cfg/models/26/yolo26.yaml.
    with contextlib.suppress(ImportError):
        import ultralytics

        ul_root = Path(ultralytics.__file__).resolve().parent
        search_paths.extend(
            [
                ul_root / "cfg" / "models" / "26" / cfg_path.name,
                ul_root / "cfg" / "models" / cfg_path.name,
            ]
        )

    for path in search_paths:
        if path.exists():
            with open(path) as f:
                cfg = yaml.safe_load(f)

            # Extract scale from filename (e.g., yolo26n.yaml -> n)
            match = re.search(r"yolo26([nsmlx])", path.stem)
            if match and "scale" not in cfg:
                cfg["scale"] = match.group(1)

            return cfg

    raise FileNotFoundError(f"Config file not found: {cfg_path}")


def build_model(
    cfg: str | dict,
    ch: int = 3,
    nc: int | None = None,
    verbose: bool = True,
    scale: str | None = None,
) -> DetectionModel:
    """Build YOLO26 model from configuration.

    Args:
        cfg: Path to config file or config dict
        ch: Input channels (default 3 for RGB)
        nc: Number of classes (overrides config)
        verbose: Print model info
        scale: Model scale (n, s, m, l, x) - overrides scale in config

    Returns:
        YOLO26 DetectionModel
    """
    return DetectionModel(cfg=cfg, ch=ch, nc=nc, verbose=verbose, scale=scale)
