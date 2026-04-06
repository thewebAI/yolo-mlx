#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 MLX - MOT17 Tracking Evaluation Script

Evaluates YOLO26 MLX models + ByteTrack/BoT-SORT on the MOT17 training set,
computing standard MOT metrics (MOTA, IDF1, MT, ML, FP, FN, IDSW, Frag).

Usage:
    # Evaluate a single model with ByteTrack
    python scripts/evaluate_mot17.py --model yolo26n --data datasets/MOT17 --tracker bytetrack

    # Evaluate all models
    python scripts/evaluate_mot17.py --model all --data datasets/MOT17

    # Quick test on one sequence
    python scripts/evaluate_mot17.py --model yolo26n --data datasets/MOT17 --sequences MOT17-02-SDP

    # Compare ByteTrack vs BoT-SORT
    python scripts/evaluate_mot17.py --model yolo26n --data datasets/MOT17 --tracker botsort

Results are saved to results/tracking/ by default.
"""

import argparse
import configparser
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
import yaml
from _runtime_dirs import ensure_runtime_dirs

# Resolve project root relative to this script (scripts/../)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent

# Add src to path
sys.path.insert(0, str(_PROJECT_DIR / "src"))

from yolo26mlx import YOLO  # noqa: E402
from yolo26mlx.engine.tracker import TrackerManager  # noqa: E402
from yolo26mlx.nn.tasks import build_model  # noqa: E402
from yolo26mlx.utils.mot_metrics import MOTAccumulator, load_mot_gt  # noqa: E402

# Try to import TrackEval (optional)
try:
    import trackeval  # noqa: F401

    HAS_TRACKEVAL = True
except ImportError:
    HAS_TRACKEVAL = False

logger = logging.getLogger(__name__)

# The 7 unique evaluation sequences (one detector variant per sequence)
DEFAULT_SEQUENCES = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]

MODEL_VARIANTS = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]

# COCO class index for "person"
PERSON_CLASS = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO26 MLX - MOT17 Tracking Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n",
        help="Model variant (yolo26n/s/m/l/x) or 'all'",
    )
    parser.add_argument("--data", type=str, default="datasets/MOT17", help="Path to MOT17 dataset")
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort"],
        help="Tracker type",
    )
    parser.add_argument("--imgsz", type=int, default=1440, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument(
        "--sequences",
        type=str,
        default="all",
        help="Comma-separated sequence names, or 'all'",
    )
    parser.add_argument(
        "--eval-class", type=int, default=1, help="MOT class to evaluate (1=pedestrian)"
    )
    parser.add_argument(
        "--save-txt", action="store_true", default=True, help="Save MOTChallenge result files"
    )
    parser.add_argument("--output", type=str, default="results/tracking", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def parse_seqinfo(seq_dir: Path) -> dict:
    """Parse seqinfo.ini to get sequence metadata."""
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.exists():
        # Fallback: infer from images
        img_dir = seq_dir / "img1"
        n_frames = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        return {
            "name": seq_dir.name,
            "imDir": "img1",
            "frameRate": 30,
            "seqLength": n_frames,
            "imWidth": 1920,
            "imHeight": 1080,
        }
    config = configparser.ConfigParser()
    config.read(str(ini_path))
    sec = config["Sequence"]
    return {
        "name": sec.get("name", seq_dir.name),
        "imDir": sec.get("imDir", "img1"),
        "frameRate": int(sec.get("frameRate", 30)),
        "seqLength": int(sec.get("seqLength", 0)),
        "imWidth": int(sec.get("imWidth", 1920)),
        "imHeight": int(sec.get("imHeight", 1080)),
    }


def load_model(model_name: str) -> YOLO:
    """Load a YOLO26 model with weights (mirrors evaluate_coco_val.py logic)."""
    import re

    scale = model_name.replace("yolo26", "")

    import yolo26mlx

    pkg_dir = Path(yolo26mlx.__file__).parent
    yaml_path = pkg_dir / "cfg" / "models" / "26" / "yolo26.yaml"

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg["scale"] = scale

    model = YOLO(str(yaml_path))
    model.model = build_model(cfg, verbose=False)
    model._setup_metadata()

    # Weight search paths
    model_path = _PROJECT_DIR / "models" / f"{model_name}.safetensors"
    npz_path = _PROJECT_DIR / "models" / f"{model_name}.npz"
    pt_path = _PROJECT_DIR / "models" / f"{model_name}.pt"

    def map_pytorch_to_mlx_name(pt_name: str) -> str:
        name = pt_name
        name = re.sub(r"^model\.(\d+)\.", r"model.layers.\1.", name)
        name = re.sub(r"layers\.10\.m\.(\d+)\.", r"layers.10.m.psa\1.", name)
        name = re.sub(r"layers\.22\.m\.0\.(\d+)\.", r"layers.22.m.0.layers.\1.", name)
        name = re.sub(
            r"layers\.23\.cv2\.(\d+)\.(\d+)\.",
            r"layers.23.cv2.layer\1.layers.\2.",
            name,
        )

        def map_cv3_nested(m):
            s, b, l_ = m.group(1), int(m.group(2)), int(m.group(3))
            return f"layers.23.cv3.layer{s}.layers.{b * 2 + l_}."

        def map_cv3_final(m):
            s, b = m.group(1), int(m.group(2))
            return f"layers.23.cv3.layer{s}.layers.{b * 2}."

        name = re.sub(r"layers\.23\.cv3\.(\d+)\.(\d+)\.(\d+)\.", map_cv3_nested, name)
        name = re.sub(r"layers\.23\.cv3\.(\d+)\.(\d+)\.", map_cv3_final, name)

        name = re.sub(
            r"layers\.23\.one2one_cv2\.(\d+)\.(\d+)\.",
            r"layers.23.one2one_cv2.layer\1.layers.\2.",
            name,
        )

        def map_o2o_cv3_nested(m):
            s, b, l_ = m.group(1), int(m.group(2)), int(m.group(3))
            return f"layers.23.one2one_cv3.layer{s}.layers.{b * 2 + l_}."

        def map_o2o_cv3_final(m):
            s, b = m.group(1), int(m.group(2))
            return f"layers.23.one2one_cv3.layer{s}.layers.{b * 2}."

        name = re.sub(r"layers\.23\.one2one_cv3\.(\d+)\.(\d+)\.(\d+)\.", map_o2o_cv3_nested, name)
        name = re.sub(r"layers\.23\.one2one_cv3\.(\d+)\.(\d+)\.", map_o2o_cv3_final, name)
        return name

    def _get_param_names(params, prefix=""):
        names = set()
        if isinstance(params, dict):
            for k, v in params.items():
                names.update(_get_param_names(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(params, list):
            for i, v in enumerate(params):
                names.update(_get_param_names(v, f"{prefix}.{i}" if prefix else str(i)))
        elif hasattr(params, "shape"):
            names.add(prefix)
        return names

    if model_path.exists():
        logger.info(f"  Loading weights from safetensors: {model_path}")
        model.model.load_weights(str(model_path))
    elif npz_path.exists():
        logger.info(f"  Loading weights from npz: {npz_path}")
        weights = dict(mx.load(str(npz_path)))
        mapped = [(map_pytorch_to_mlx_name(k), v) for k, v in weights.items()]
        mlx_names = _get_param_names(model.model.parameters())
        matching = [(k, v) for k, v in mapped if k in mlx_names]
        logger.info(f"  Matching weights: {len(matching)}/{len(mapped)}")
        model.model.load_weights(matching, strict=False)
    elif pt_path.exists():
        logger.info(f"  Loading weights from PyTorch: {pt_path}")
        import torch

        pt_data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        if hasattr(pt_data, "state_dict"):
            sd = pt_data.state_dict()
        elif isinstance(pt_data, dict) and "model" in pt_data:
            sd = (
                pt_data["model"].state_dict()
                if hasattr(pt_data["model"], "state_dict")
                else pt_data["model"]
            )
        elif isinstance(pt_data, dict):
            sd = pt_data
        else:
            sd = pt_data.model.state_dict() if hasattr(pt_data.model, "state_dict") else {}

        def _convert_tensor(t):
            if hasattr(t, "numpy"):
                arr = t.float().numpy()
                if len(arr.shape) == 4:
                    arr = arr.transpose(0, 2, 3, 1)  # OIHW -> OHWI
                return mx.array(arr)
            return t

        converted = {k: _convert_tensor(v) for k, v in sd.items()}
        mapped = [(map_pytorch_to_mlx_name(k), v) for k, v in converted.items()]
        mlx_names = _get_param_names(model.model.parameters())
        matching = [(k, v) for k, v in mapped if k in mlx_names]
        logger.info(f"  Matching weights: {len(matching)}/{len(mapped)}")
        model.model.load_weights(matching, strict=False)
        # Cache as npz
        save_path = _PROJECT_DIR / "models" / f"{model_name}.npz"
        mx.savez(str(save_path), **dict(matching))
        logger.info(f"  Cached converted weights to {save_path}")
    else:
        logger.warning(f"  WARNING: No weights found for {model_name}!")

    if hasattr(model.model, "eval"):
        model.model.eval()

    return model


def evaluate_sequence(
    model: YOLO,
    tracker_name: str,
    seq_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    eval_class: int,
    save_txt: bool,
    output_dir: Path,
    model_name: str = "",
) -> dict:
    """Evaluate tracking on a single MOT17 sequence.

    Returns:
        Dict with metrics and timing for this sequence.
    """
    seq_info = parse_seqinfo(seq_dir)
    seq_name = seq_info["name"]
    frame_rate = seq_info["frameRate"]
    img_dir = seq_dir / seq_info["imDir"]

    logger.info(f"\n  Sequence: {seq_name}")
    logger.info(
        f"    Frames: {seq_info['seqLength']}, FPS: {frame_rate}, "
        f"Resolution: {seq_info['imWidth']}x{seq_info['imHeight']}"
    )

    # Load ground truth
    gt_path = seq_dir / "gt" / "gt.txt"
    gt_by_frame = load_mot_gt(str(gt_path), eval_class=eval_class) if gt_path.exists() else {}

    # Initialize tracker
    tracker_cfg = f"{tracker_name}.yaml"
    tracker_mgr = TrackerManager(tracker_cfg, frame_rate=frame_rate)

    # Collect frame paths (sorted by name)
    frame_paths = sorted(img_dir.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(img_dir.glob("*.png"))
    if not frame_paths:
        logger.warning(f"    No frames found in {img_dir}")
        return {}

    total_frames = len(frame_paths)
    logger.info(f"    Found {total_frames} frames")

    # Tracking results in MOTChallenge format
    mot_results = []  # list of (frame, tid, x, y, w, h, conf, -1, -1, -1)
    acc = MOTAccumulator(iou_threshold=0.5)

    det_time = 0.0
    track_time = 0.0
    total_time = 0.0
    io_time = 0.0

    # Warmup: run one detection to trigger JIT compilation before timing
    warmup_frame = cv2.imread(str(frame_paths[0]))
    if warmup_frame is not None:
        warmup_rgb = cv2.cvtColor(warmup_frame, cv2.COLOR_BGR2RGB)
        _ = model.predict(warmup_rgb, conf=conf, imgsz=imgsz)

    def _read_frame(path):
        """Read and convert a frame (runs in background thread)."""
        f = cv2.imread(str(path))
        if f is None:
            return None
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(max_workers=1) as prefetch:
        # Submit first frame read
        future = prefetch.submit(_read_frame, frame_paths[0])

        for frame_idx in range(total_frames):
            frame_num = frame_idx + 1  # 1-based

            t0 = time.perf_counter()

            # Get current frame from prefetch (blocks until ready)
            frame_rgb = future.result()

            # Submit next frame read immediately (overlaps I/O with compute)
            if frame_idx + 1 < total_frames:
                future = prefetch.submit(_read_frame, frame_paths[frame_idx + 1])

            if frame_rgb is None:
                continue
            t_io_end = time.perf_counter()

            # Run detection (person class filter applied after)
            t_det_start = time.perf_counter()
            det_results = model.predict(frame_rgb, conf=conf, imgsz=imgsz)
            t_det_end = time.perf_counter()

            # Filter to person class only
            result = det_results[0]
            if len(result.boxes) > 0:
                person_mask = result.boxes.cls.astype(int) == PERSON_CLASS
                if person_mask.any():
                    result = result[person_mask]
                else:
                    # Create empty result
                    from yolo26mlx.engine.results import Boxes, Results

                    empty_boxes = Boxes(
                        np.empty((0, 6), dtype=np.float32),
                        orig_shape=result.orig_shape,
                    )
                    result = Results(
                        orig_img=result.orig_img,
                        path=result.path,
                        names=result.names,
                        boxes=empty_boxes,
                    )

            # Run tracker
            t_track_start = time.perf_counter()
            tracked = tracker_mgr.update(result)
            t_track_end = time.perf_counter()

            t1 = time.perf_counter()

            io_time += t_io_end - t0
            det_time += t_det_end - t_det_start
            track_time += t_track_end - t_track_start
            total_time += t1 - t0

            # Collect tracked outputs
            if tracked.boxes is not None and len(tracked.boxes) > 0 and tracked.boxes.is_track:
                track_ids = tracked.boxes.id
                xyxy = tracked.boxes.xyxy
                confs = tracked.boxes.conf

                for i in range(len(track_ids)):
                    tid = int(track_ids[i])
                    x1, y1, x2, y2 = xyxy[i]
                    w = x2 - x1
                    h = y2 - y1
                    c = float(confs[i])
                    mot_results.append(
                        (frame_num, tid, float(x1), float(y1), float(w), float(h), c, -1, -1, -1)
                    )

            # Evaluate against ground truth
            gt_entries = gt_by_frame.get(frame_num, [])
            if gt_entries:
                gt_ids_frame = np.array([e[0] for e in gt_entries])
                gt_boxes_frame = np.array([e[1] for e in gt_entries])
            else:
                gt_ids_frame = np.array([])
                gt_boxes_frame = np.empty((0, 4))

            if tracked.boxes is not None and len(tracked.boxes) > 0 and tracked.boxes.is_track:
                pred_ids_frame = tracked.boxes.id
                pred_boxes_frame = tracked.boxes.xyxy
            else:
                pred_ids_frame = np.array([])
                pred_boxes_frame = np.empty((0, 4))

            acc.update(gt_ids_frame, gt_boxes_frame, pred_ids_frame, pred_boxes_frame)

        # Progress
        if (frame_idx + 1) % 100 == 0 or frame_idx == total_frames - 1:
            logger.info(f"    Frame {frame_idx + 1}/{total_frames}")

    # Compute metrics
    metrics = acc.compute()

    # Timing
    avg_det_ms = det_time / max(total_frames, 1) * 1000
    avg_track_ms = track_time / max(total_frames, 1) * 1000
    avg_io_ms = io_time / max(total_frames, 1) * 1000
    avg_total_ms = total_time / max(total_frames, 1) * 1000
    compute_ms = avg_total_ms - avg_io_ms
    fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0
    compute_fps = 1000.0 / compute_ms if compute_ms > 0 else 0.0

    logger.info(
        f"    MOTA: {metrics['MOTA']:.1f}  IDF1: {metrics['IDF1']:.1f}  "
        f"FP: {metrics['FP']}  FN: {metrics['FN']}  IDSW: {metrics['IDSW']}"
    )
    logger.info(
        f"    Speed: {avg_det_ms:.1f}ms det + {avg_track_ms:.1f}ms track + {avg_io_ms:.1f}ms io "
        f"= {avg_total_ms:.1f}ms/frame ({fps:.1f} FPS, {compute_fps:.1f} compute FPS)"
    )

    # Save MOTChallenge format result file
    if save_txt and mot_results:
        txt_dir = output_dir / tracker_name / model_name
        txt_dir.mkdir(parents=True, exist_ok=True)
        txt_path = txt_dir / f"{seq_name}.txt"
        with open(txt_path, "w") as f:
            for row in mot_results:
                f.write(",".join(str(v) for v in row) + "\n")
        logger.info(f"    Saved {len(mot_results)} predictions to {txt_path}")

    return {
        "sequence": seq_name,
        "metrics": metrics,
        "speed": {
            "detection_ms": round(avg_det_ms, 2),
            "tracking_ms": round(avg_track_ms, 2),
            "io_ms": round(avg_io_ms, 2),
            "total_ms": round(avg_total_ms, 2),
            "fps": round(fps, 1),
            "compute_fps": round(compute_fps, 1),
        },
        "num_frames": total_frames,
        "num_predictions": len(mot_results),
    }


def evaluate_model_mot17(model_name: str, args) -> dict:
    """Evaluate a single model on MOT17 and return aggregate results."""
    logger.info("\n" + "=" * 70)
    logger.info(f"MOT17 Evaluation: {model_name} + {args.tracker}")
    logger.info("=" * 70)

    # Load model
    logger.info(f"\nLoading model {model_name}...")
    model = load_model(model_name)

    # Determine sequences
    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    if not train_dir.exists():
        logger.error(f"MOT17 training data not found at {train_dir}")
        logger.error("Run: bash scripts/download_mot17.sh")
        return {}

    if args.sequences == "all":
        sequences = DEFAULT_SEQUENCES
    else:
        sequences = [s.strip() for s in args.sequences.split(",")]

    # Verify sequences exist
    valid_seqs = []
    for seq in sequences:
        seq_dir = train_dir / seq
        if seq_dir.exists():
            valid_seqs.append(seq)
        else:
            logger.warning(f"  Sequence not found: {seq_dir}")
    sequences = valid_seqs

    if not sequences:
        logger.error("No valid sequences found!")
        return {}

    logger.info(f"\nEvaluating {len(sequences)} sequences...")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each sequence
    per_sequence = {}
    total_det_time = 0.0
    total_track_time = 0.0
    total_io_time = 0.0
    total_wall_time = 0.0
    total_frames = 0

    for seq_name in sequences:
        seq_dir = train_dir / seq_name
        result = evaluate_sequence(
            model=model,
            tracker_name=args.tracker,
            seq_dir=seq_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            eval_class=args.eval_class,
            save_txt=args.save_txt,
            output_dir=output_dir,
            model_name=model_name,
        )
        if result:
            per_sequence[seq_name] = result
            total_det_time += result["speed"]["detection_ms"] * result["num_frames"]
            total_track_time += result["speed"]["tracking_ms"] * result["num_frames"]
            total_io_time += result["speed"]["io_ms"] * result["num_frames"]
            total_wall_time += result["speed"]["total_ms"] * result["num_frames"]
            total_frames += result["num_frames"]

    # Aggregate metrics: re-run accumulator over all saved result files
    # OR sum up per-sequence raw counts. We use the simpler approach:
    # accumulate FP/FN/IDSW across sequences.
    agg_fp = sum(r["metrics"]["FP"] for r in per_sequence.values())
    agg_fn = sum(r["metrics"]["FN"] for r in per_sequence.values())
    agg_idsw = sum(r["metrics"]["IDSW"] for r in per_sequence.values())
    agg_frag = sum(r["metrics"]["Frag"] for r in per_sequence.values())
    agg_gt = sum(r["metrics"]["num_frames_gt"] for r in per_sequence.values())

    agg_mota = (1.0 - (agg_fn + agg_fp + agg_idsw) / max(agg_gt, 1)) * 100.0

    # Aggregate speed — use wall-clock total_ms for FPS (consistent with PyTorch script)
    avg_det_ms = total_det_time / max(total_frames, 1)
    avg_track_ms = total_track_time / max(total_frames, 1)
    avg_io_ms = total_io_time / max(total_frames, 1)
    avg_total_ms = total_wall_time / max(total_frames, 1)
    compute_ms = avg_total_ms - avg_io_ms
    agg_fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0
    agg_compute_fps = 1000.0 / compute_ms if compute_ms > 0 else 0.0

    # Aggregate IDF1 — average per-sequence IDF1 (weighted by frame count would be better,
    # but per-sequence average is simpler and common in literature)
    idf1_vals = [r["metrics"]["IDF1"] for r in per_sequence.values()]
    agg_idf1 = sum(idf1_vals) / len(idf1_vals) if idf1_vals else 0.0

    # MT/ML — average across sequences
    mt_vals = [r["metrics"]["MT"] for r in per_sequence.values()]
    ml_vals = [r["metrics"]["ML"] for r in per_sequence.values()]
    agg_mt = sum(mt_vals) / len(mt_vals) if mt_vals else 0.0
    agg_ml = sum(ml_vals) / len(ml_vals) if ml_vals else 0.0

    aggregate_metrics = {
        "MOTA": round(agg_mota, 2),
        "IDF1": round(agg_idf1, 2),
        "MT": round(agg_mt, 2),
        "ML": round(agg_ml, 2),
        "FP": agg_fp,
        "FN": agg_fn,
        "IDSW": agg_idsw,
        "Frag": agg_frag,
    }

    # Print summary table
    _print_summary_table(aggregate_metrics, per_sequence, args.tracker)

    # Build results dict
    results = {
        "model": model_name,
        "tracker": args.tracker,
        "framework": "mlx",
        "dataset": "MOT17-train",
        "num_sequences": len(sequences),
        "imgsz": args.imgsz,
        "conf_thresh": args.conf,
        "trackeval": False,
        "metrics": aggregate_metrics,
        "per_sequence": {
            seq: {
                "MOTA": r["metrics"]["MOTA"],
                "IDF1": r["metrics"]["IDF1"],
                "MT": r["metrics"]["MT"],
                "ML": r["metrics"]["ML"],
                "FP": r["metrics"]["FP"],
                "FN": r["metrics"]["FN"],
                "IDSW": r["metrics"]["IDSW"],
                "Frag": r["metrics"]["Frag"],
            }
            for seq, r in per_sequence.items()
        },
        "speed": {
            "detection_ms": round(avg_det_ms, 2),
            "tracking_ms": round(avg_track_ms, 2),
            "io_ms": round(avg_io_ms, 2),
            "total_ms": round(avg_total_ms, 2),
            "fps": round(agg_fps, 1),
            "compute_fps": round(agg_compute_fps, 1),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # Save JSON
    json_path = output_dir / f"{model_name}_{args.tracker}_mot17_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {json_path}")

    return results


def _print_summary_table(aggregate: dict, per_sequence: dict, tracker_name: str):
    """Print a formatted summary table of MOT metrics."""
    logger.info("\n" + "=" * 90)
    logger.info(f"MOT17 Results Summary ({tracker_name})")
    logger.info("=" * 90)

    header = f"{'Sequence':<18} {'MOTA':>7} {'IDF1':>7} {'MT':>6} {'ML':>6} {'FP':>7} {'FN':>8} {'IDSW':>6} {'Frag':>6}"
    logger.info(header)
    logger.info("-" * 90)

    for seq_name, result in sorted(per_sequence.items()):
        m = result["metrics"]
        row = (
            f"{seq_name:<18} "
            f"{m['MOTA']:>7.1f} "
            f"{m['IDF1']:>7.1f} "
            f"{m['MT']:>6.1f} "
            f"{m['ML']:>6.1f} "
            f"{m['FP']:>7d} "
            f"{m['FN']:>8d} "
            f"{m['IDSW']:>6d} "
            f"{m['Frag']:>6d}"
        )
        logger.info(row)

    logger.info("-" * 90)
    a = aggregate
    row = (
        f"{'AGGREGATE':<18} "
        f"{a['MOTA']:>7.1f} "
        f"{a['IDF1']:>7.1f} "
        f"{a['MT']:>6.1f} "
        f"{a['ML']:>6.1f} "
        f"{a['FP']:>7d} "
        f"{a['FN']:>8d} "
        f"{a['IDSW']:>6d} "
        f"{a['Frag']:>6d}"
    )
    logger.info(row)
    logger.info("=" * 90)


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )

    ensure_runtime_dirs(_PROJECT_DIR)

    logger.info("=" * 70)
    logger.info("YOLO26 MLX — MOT17 Tracking Evaluation")
    logger.info("=" * 70)
    logger.info(f"  Tracker: {args.tracker}")
    logger.info(f"  Image size: {args.imgsz}")
    logger.info(f"  Confidence: {args.conf}")
    logger.info(f"  IoU: {args.iou}")
    logger.info(f"  Dataset: {args.data}")
    logger.info(
        f"  TrackEval: {'available' if HAS_TRACKEVAL else 'not installed (using built-in metrics)'}"
    )

    # Determine models to evaluate
    if args.model == "all":
        model_names = MODEL_VARIANTS
    else:
        model_names = [args.model]

    all_results = []

    for model_name in model_names:
        result = evaluate_model_mot17(model_name, args)
        if result:
            all_results.append(result)

    # Save combined results if multiple models
    if len(all_results) > 1:
        output_dir = Path(args.output)
        combined_path = output_dir / f"yolo26_all_{args.tracker}_mot17_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nCombined results saved to {combined_path}")

    # Final comparison table
    if len(all_results) > 1:
        logger.info("\n" + "=" * 80)
        logger.info(f"Model Comparison — MOT17 + {args.tracker}")
        logger.info("=" * 80)
        header = f"{'Model':<12} {'MOTA':>7} {'IDF1':>7} {'FP':>7} {'FN':>8} {'IDSW':>6} {'FPS':>7} {'Comp.FPS':>9}"
        logger.info(header)
        logger.info("-" * 80)
        for r in all_results:
            m = r["metrics"]
            s = r["speed"]
            row = (
                f"{r['model']:<12} "
                f"{m['MOTA']:>7.1f} "
                f"{m['IDF1']:>7.1f} "
                f"{m['FP']:>7d} "
                f"{m['FN']:>8d} "
                f"{m['IDSW']:>6d} "
                f"{s['fps']:>7.1f}"
                f"{s.get('compute_fps', s['fps']):>9.1f}"
            )
            logger.info(row)
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
