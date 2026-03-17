# Copyright (c) 2026 webAI, Inc.
"""Package-level CLI for yolo26mlx."""

import argparse
import logging
from pathlib import Path


def _cmd_converters_convert(args: argparse.Namespace) -> int:
    """Handle `converters convert` subcommand."""
    from yolo26mlx.converters.convert import convert_yolo26_weights, verify_conversion

    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    weights = convert_yolo26_weights(
        pt_path=args.input,
        output_path=args.output,
        verbose=not args.quiet,
    )
    if args.verify:
        ok = verify_conversion(args.input, weights)
        return 0 if ok else 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="yolo26",
        description="YOLO26 MLX command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    converters = subparsers.add_parser(
        "converters",
        help="Weight conversion utilities.",
    )
    converters_subparsers = converters.add_subparsers(dest="converters_command", required=True)

    convert_cmd = converters_subparsers.add_parser(
        "convert",
        help="Convert PyTorch .pt weights to MLX format.",
    )
    convert_cmd.add_argument("input", help="Path to input PyTorch .pt file.")
    convert_cmd.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output MLX weight file (.safetensors or .npz).",
    )
    convert_cmd.add_argument(
        "--verify",
        action="store_true",
        help="Verify converted shapes against source checkpoint.",
    )
    convert_cmd.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress informational logs.",
    )
    convert_cmd.set_defaults(func=_cmd_converters_convert)

    return parser


def main() -> int:
    """Run CLI."""
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
