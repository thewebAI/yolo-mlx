# Copyright (c) 2026 webAI, Inc.
"""Package-level CLI for yolo26mlx.

Subcommands
-----------
predict              Run object detection on images.
train                Train a YOLO26 model on a dataset.
val                  Validate a YOLO26 model on a dataset.
track                Run multi-object tracking on video or webcam.
info                 Display model architecture information.
converters convert   Convert PyTorch .pt weights to MLX format.
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Subcommand handlers ──────────────────────────────────────────────


def _cmd_predict(args: argparse.Namespace) -> int:
    """Handle ``predict`` subcommand."""
    from yolo26mlx import YOLO

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
    )

    if not args.quiet:
        for r in results:
            logger.info(r)

    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Handle ``train`` subcommand."""
    from yolo26mlx import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        save_period=args.save_period,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
    )
    return 0


def _cmd_val(args: argparse.Namespace) -> int:
    """Handle ``val`` subcommand."""
    from yolo26mlx import YOLO

    model = YOLO(args.model)
    model.val(
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    return 0


def _cmd_track(args: argparse.Namespace) -> int:
    """Handle ``track`` subcommand."""
    from yolo26mlx import YOLO

    model = YOLO(args.model)

    source = args.source
    if source.isdigit():
        source = int(source)

    model.track(
        source=source,
        tracker=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        show=args.show,
        save=args.save,
        vid_stride=args.vid_stride,
    )
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """Handle ``info`` subcommand."""
    from yolo26mlx import YOLO

    model = YOLO(args.model, verbose=False)
    model.info(verbose=True)
    return 0


def _cmd_converters_convert(args: argparse.Namespace) -> int:
    """Handle ``converters convert`` subcommand."""
    from yolo26mlx.converters.convert import convert_yolo26_weights, verify_conversion

    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    weights = convert_yolo26_weights(
        pt_path=args.input,
        output_path=args.output,
        verbose=not args.quiet,
    )
    if args.verify:
        ok = verify_conversion(args.input, weights)
        return 0 if ok else 1
    return 0


# ── Parser construction ──────────────────────────────────────────────


def _add_common_model_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--model`` argument shared by most subcommands."""
    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLO26 model weights (.npz, .safetensors, .pt) or config (.yaml).",
    )


def _add_quiet_flag(parser: argparse.ArgumentParser) -> None:
    """Add the ``-q/--quiet`` flag shared by most subcommands."""
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress informational logs.")


def build_parser() -> argparse.ArgumentParser:
    """Build top-level CLI parser with all subcommands.

    Returns:
        argparse.ArgumentParser: The fully configured argument parser.
    """
    from yolo26mlx import __version__

    parser = argparse.ArgumentParser(
        prog="yolo26",
        description="YOLO26 MLX — pure-MLX object detection and tracking.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── predict ──
    predict_cmd = subparsers.add_parser("predict", help="Run object detection on images.")
    _add_common_model_arg(predict_cmd)
    predict_cmd.add_argument(
        "--source",
        required=True,
        help="Image path, directory, or glob pattern.",
    )
    predict_cmd.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold. Default: 0.25."
    )
    predict_cmd.add_argument(
        "--imgsz", type=int, default=640, help="Input image size. Default: 640."
    )
    predict_cmd.add_argument("--save", action="store_true", help="Save annotated images to disk.")
    _add_quiet_flag(predict_cmd)
    predict_cmd.set_defaults(func=_cmd_predict)

    # ── train ──
    train_cmd = subparsers.add_parser("train", help="Train a YOLO26 model.")
    _add_common_model_arg(train_cmd)
    train_cmd.add_argument(
        "--data", required=True, help="Path to data config YAML or dataset name (e.g. coco128)."
    )
    train_cmd.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs. Default: 100."
    )
    train_cmd.add_argument("--imgsz", type=int, default=640, help="Input image size. Default: 640.")
    train_cmd.add_argument("--batch", type=int, default=16, help="Batch size. Default: 16.")
    train_cmd.add_argument(
        "--patience", type=int, default=50, help="Early stopping patience. Default: 50."
    )
    train_cmd.add_argument(
        "--save-period",
        type=int,
        default=-1,
        dest="save_period",
        help="Save checkpoint every N epochs (-1 to disable). Default: -1.",
    )
    train_cmd.add_argument(
        "--project", default="runs/train", help="Project directory. Default: runs/train."
    )
    train_cmd.add_argument("--name", default="exp", help="Experiment name. Default: exp.")
    train_cmd.add_argument(
        "--exist-ok",
        action="store_true",
        dest="exist_ok",
        help="Overwrite existing experiment directory.",
    )
    train_cmd.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint."
    )
    _add_quiet_flag(train_cmd)
    train_cmd.set_defaults(func=_cmd_train)

    # ── val ──
    val_cmd = subparsers.add_parser("val", help="Validate a YOLO26 model on a dataset.")
    _add_common_model_arg(val_cmd)
    val_cmd.add_argument("--data", default=None, help="Path to validation data config YAML.")
    val_cmd.add_argument("--batch", type=int, default=16, help="Batch size. Default: 16.")
    val_cmd.add_argument("--imgsz", type=int, default=640, help="Input image size. Default: 640.")
    val_cmd.add_argument(
        "--conf", type=float, default=0.001, help="Confidence threshold. Default: 0.001."
    )
    val_cmd.add_argument("--iou", type=float, default=0.6, help="IoU threshold. Default: 0.6.")
    _add_quiet_flag(val_cmd)
    val_cmd.set_defaults(func=_cmd_val)

    # ── track ──
    track_cmd = subparsers.add_parser(
        "track", help="Run multi-object tracking on video, image, or webcam."
    )
    _add_common_model_arg(track_cmd)
    track_cmd.add_argument(
        "--source", required=True, help="Video path, image path, or webcam index (0)."
    )
    track_cmd.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Tracker config YAML (bytetrack.yaml or botsort.yaml). Default: bytetrack.yaml.",
    )
    track_cmd.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold. Default: 0.25."
    )
    track_cmd.add_argument("--imgsz", type=int, default=640, help="Input image size. Default: 640.")
    track_cmd.add_argument("--show", action="store_true", help="Display results with cv2.imshow.")
    track_cmd.add_argument("--save", action="store_true", help="Save annotated video to disk.")
    track_cmd.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        dest="vid_stride",
        help="Process every Nth frame. Default: 1.",
    )
    _add_quiet_flag(track_cmd)
    track_cmd.set_defaults(func=_cmd_track)

    # ── info ──
    info_cmd = subparsers.add_parser("info", help="Display model architecture information.")
    _add_common_model_arg(info_cmd)
    info_cmd.set_defaults(func=_cmd_info)

    # ── converters ──
    converters = subparsers.add_parser("converters", help="Weight conversion utilities.")
    converters_sub = converters.add_subparsers(dest="converters_command", required=True)

    convert_cmd = converters_sub.add_parser(
        "convert", help="Convert PyTorch .pt weights to MLX format."
    )
    convert_cmd.add_argument("input", help="Path to input PyTorch .pt file.")
    convert_cmd.add_argument(
        "-o", "--output", default=None, help="Output MLX weight file (.safetensors or .npz)."
    )
    convert_cmd.add_argument(
        "--verify",
        action="store_true",
        help="Verify converted shapes against source checkpoint.",
    )
    _add_quiet_flag(convert_cmd)
    convert_cmd.set_defaults(func=_cmd_converters_convert)

    return parser


# ── Entrypoint ────────────────────────────────────────────────────────


def main() -> int:
    """Run CLI, returning an integer exit code."""
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.WARNING if getattr(args, "quiet", False) else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("\nInterrupted.")
        return 130
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.error(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
