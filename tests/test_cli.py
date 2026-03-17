# Copyright (c) 2026 webAI, Inc.
"""Basic tests for package CLI and import behavior."""

from yolo26mlx import __version__
from yolo26mlx.cli import build_parser


def test_package_version_available() -> None:
    """Version should be importable without loading heavy runtime deps."""
    assert isinstance(__version__, str)
    assert __version__


def test_cli_parser_has_converters_command() -> None:
    """Top-level parser should expose converters subcommand."""
    parser = build_parser()
    args = parser.parse_args(["converters", "convert", "weights.pt"])
    assert args.command == "converters"
    assert args.converters_command == "convert"
    assert args.input == "weights.pt"


def test_convert_command_flags() -> None:
    """Convert subcommand should parse common flags correctly."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "converters",
            "convert",
            "weights.pt",
            "-o",
            "weights.npz",
            "--verify",
            "--quiet",
        ]
    )
    assert args.output == "weights.npz"
    assert args.verify is True
    assert args.quiet is True
