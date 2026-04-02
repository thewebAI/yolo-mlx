#!/usr/bin/env python3
# Copyright (c) 2026 webAI, Inc.
"""
YOLO26 PyTorch - MOT17 Tracking Evaluation Script

Evaluates YOLO26 PyTorch models (via Ultralytics) + ByteTrack on MOT17 training set,
computing standard MOT metrics (MOTA, IDF1, MT, ML, FP, FN, IDSW, Frag).

Uses the same MOTAccumulator and ground truth loader as the MLX version for
apples-to-apples comparison.

Usage:
    # Single model on MPS
    python scripts/evaluate_mot17_pytorch.py --model yolo26n --device mps

    # Single model on CPU
    python scripts/evaluate_mot17_pytorch.py --model yolo26n --device cpu

    # All models
    python scripts/evaluate_mot17_pytorch.py --model all --device mps

    # Quick test on one sequence
    python scripts/evaluate_mot17_pytorch.py --model yolo26n --device mps --sequences MOT17-09-SDP
"""

import argparse
import configparser
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR / "src"))

from yolo26mlx.utils.mot_metrics import MOTAccumulator, load_mot_gt  # noqa: E402

logger = logging.getLogger(__name__)

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
PERSON_CLASS = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO26 PyTorch - MOT17 Tracking Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="yolo26n", help="Model variant (yolo26n/s/m/l/x) or 'all'"
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
        "--sequences", type=str, default="all", help="Comma-separated sequence names, or 'all'"
    )
    parser.add_argument(
        "--eval-class", type=int, default=1, help="MOT class to evaluate (1=pedestrian)"
    )
    parser.add_argument(
        "--save-txt", action="store_true", default=True, help="Save MOTChallenge result files"
    )
    parser.add_argument("--output", type=str, default="results/tracking", help="Output directory")
    parser.add_argument(
        "--device", type=str, default="mps", choices=["mps", "cpu"], help="PyTorch device"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def parse_seqinfo(seq_dir: Path) -> dict:
    """Parse seqinfo.ini to get sequence metadata."""
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.exists():
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


def evaluate_sequence(
    model,
    seq_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    eval_class: int,
    save_txt: bool,
    output_dir: Path,
    model_name: str,
    tracker_name: str,
    device: str,
) -> dict:
    """Evaluate tracking on a single MOT17 sequence using Ultralytics PyTorch."""
    seq_info = parse_seqinfo(seq_dir)
    seq_name = seq_info["name"]
    frame_rate = seq_info["frameRate"]
    img_dir = seq_dir / seq_info["imDir"]

    logger.info(f"\n  Sequence: {seq_name}")
    logger.info(
        f"    Frames: {seq_info['seqLength']}, FPS: {frame_rate}, "
        f"Resolution: {seq_info['imWidth']}x{seq_info['imHeight']}"
    )

    gt_path = seq_dir / "gt" / "gt.txt"
    gt_by_frame = load_mot_gt(str(gt_path), eval_class=eval_class) if gt_path.exists() else {}

    frame_paths = sorted(img_dir.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(img_dir.glob("*.png"))
    if not frame_paths:
        logger.warning(f"    No frames found in {img_dir}")
        return {}

    total_frames = len(frame_paths)
    logger.info(f"    Found {total_frames} frames")

    mot_results = []
    acc = MOTAccumulator(iou_threshold=0.5)

    det_time = 0.0
    total_time = 0.0

    for frame_idx, frame_path in enumerate(frame_paths):
        frame_num = frame_idx + 1

        t0 = time.perf_counter()

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        # Ultralytics model.track() handles BGR input
        # persist=True keeps tracker state between frames within a sequence
        t_det_start = time.perf_counter()
        results = model.track(
            frame,
            persist=True,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            tracker=f"{tracker_name}.yaml",
            classes=[PERSON_CLASS],
            device=device,
            verbose=False,
        )
        t_det_end = time.perf_counter()

        t1 = time.perf_counter()

        # Ultralytics combines det+track in one call, so split is approximate
        det_time += t_det_end - t_det_start
        total_time += t1 - t0

        result = results[0]
        boxes = result.boxes

        # Extract tracked predictions
        pred_ids_frame = np.array([])
        pred_boxes_frame = np.empty((0, 4))

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            pred_ids_frame = track_ids
            pred_boxes_frame = xyxy

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

        acc.update(gt_ids_frame, gt_boxes_frame, pred_ids_frame, pred_boxes_frame)

        if (frame_idx + 1) % 100 == 0 or frame_idx == total_frames - 1:
            logger.info(f"    Frame {frame_idx + 1}/{total_frames}")

    metrics = acc.compute()

    avg_det_ms = det_time / max(total_frames, 1) * 1000
    avg_total_ms = total_time / max(total_frames, 1) * 1000
    fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0

    logger.info(
        f"    MOTA: {metrics['MOTA']:.1f}  IDF1: {metrics['IDF1']:.1f}  "
        f"FP: {metrics['FP']}  FN: {metrics['FN']}  IDSW: {metrics['IDSW']}"
    )
    logger.info(
        f"    Speed: {avg_det_ms:.1f}ms det+track = {avg_total_ms:.1f}ms/frame ({fps:.1f} FPS)"
    )

    if save_txt and mot_results:
        txt_dir = output_dir / f"{tracker_name}_pytorch_{device}" / model_name
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
            "total_ms": round(avg_total_ms, 2),
            "fps": round(fps, 1),
        },
        "num_frames": total_frames,
        "num_predictions": len(mot_results),
    }


def evaluate_model(model_name: str, args) -> dict:
    """Evaluate a single PyTorch model on MOT17."""
    from ultralytics import YOLO

    logger.info("\n" + "=" * 70)
    logger.info(f"MOT17 Evaluation: {model_name} + {args.tracker} (PyTorch {args.device})")
    logger.info("=" * 70)

    pt_path = _PROJECT_DIR / "models" / f"{model_name}.pt"
    if not pt_path.exists():
        logger.error(f"Model not found: {pt_path}")
        return {}

    logger.info(f"\nLoading model {model_name} on {args.device}...")
    model = YOLO(str(pt_path))

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    if not train_dir.exists():
        logger.error(f"MOT17 training data not found at {train_dir}")
        return {}

    sequences = (
        DEFAULT_SEQUENCES
        if args.sequences == "all"
        else [s.strip() for s in args.sequences.split(",")]
    )
    sequences = [s for s in sequences if (train_dir / s).exists()]
    if not sequences:
        logger.error("No valid sequences found!")
        return {}

    logger.info(f"\nEvaluating {len(sequences)} sequences...")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sequence = {}
    total_time_weighted = 0.0
    total_frames = 0

    for seq_name in sequences:
        seq_dir = train_dir / seq_name
        # Create fresh model instance for each sequence to guarantee clean tracker state
        model = YOLO(str(pt_path))
        result = evaluate_sequence(
            model=model,
            seq_dir=seq_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            eval_class=args.eval_class,
            save_txt=args.save_txt,
            output_dir=output_dir,
            model_name=model_name,
            tracker_name=args.tracker,
            device=args.device,
        )
        if result:
            per_sequence[seq_name] = result
            total_time_weighted += result["speed"]["total_ms"] * result["num_frames"]
            total_frames += result["num_frames"]

    # Aggregate
    agg_fp = sum(r["metrics"]["FP"] for r in per_sequence.values())
    agg_fn = sum(r["metrics"]["FN"] for r in per_sequence.values())
    agg_idsw = sum(r["metrics"]["IDSW"] for r in per_sequence.values())
    agg_frag = sum(r["metrics"]["Frag"] for r in per_sequence.values())
    agg_gt = sum(r["metrics"]["num_frames_gt"] for r in per_sequence.values())

    agg_mota = (1.0 - (agg_fn + agg_fp + agg_idsw) / max(agg_gt, 1)) * 100.0

    avg_total_ms = total_time_weighted / max(total_frames, 1)
    agg_fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0

    idf1_vals = [r["metrics"]["IDF1"] for r in per_sequence.values()]
    agg_idf1 = sum(idf1_vals) / len(idf1_vals) if idf1_vals else 0.0

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

    # Print summary
    logger.info("\n" + "=" * 90)
    logger.info(f"MOT17 Results Summary ({args.tracker}, PyTorch {args.device})")
    logger.info("=" * 90)
    header = f"{'Sequence':<18} {'MOTA':>7} {'IDF1':>7} {'FP':>7} {'FN':>8} {'IDSW':>6}"
    logger.info(header)
    logger.info("-" * 90)
    for seq, r in per_sequence.items():
        m = r["metrics"]
        logger.info(
            f"{seq:<18} {m['MOTA']:7.1f} {m['IDF1']:7.1f} {m['FP']:7d} {m['FN']:8d} {m['IDSW']:6d}"
        )
    logger.info("-" * 90)
    logger.info(
        f"{'AGGREGATE':<18} {aggregate_metrics['MOTA']:7.1f} {aggregate_metrics['IDF1']:7.1f} "
        f"{aggregate_metrics['FP']:7d} {aggregate_metrics['FN']:8d} {aggregate_metrics['IDSW']:6d}"
    )
    logger.info(f"\nSpeed: {avg_total_ms:.1f}ms/frame ({agg_fps:.1f} FPS)")

    results = {
        "model": model_name,
        "tracker": args.tracker,
        "framework": f"pytorch_{args.device}",
        "dataset": "MOT17-train",
        "num_sequences": len(sequences),
        "imgsz": args.imgsz,
        "conf_thresh": args.conf,
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
            "total_ms": round(avg_total_ms, 2),
            "fps": round(agg_fps, 1),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    json_path = output_dir / f"{model_name}_{args.tracker}_mot17_pytorch_{args.device}_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {json_path}")

    return results


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if args.model == "all":
        models = MODEL_VARIANTS
    else:
        models = [args.model]

    all_results = []
    for model_name in models:
        result = evaluate_model(model_name, args)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        combined_path = (
            Path(args.output)
            / f"yolo26_all_{args.tracker}_mot17_pytorch_{args.device}_results.json"
        )
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nCombined results saved to {combined_path}")

    if all_results:
        logger.info("\n" + "=" * 90)
        logger.info(f"ALL MODELS SUMMARY (PyTorch {args.device})")
        logger.info("=" * 90)
        logger.info(
            f"{'Model':<12} {'MOTA':>7} {'IDF1':>7} {'FP':>8} {'FN':>8} {'IDSW':>6} {'FPS':>6}"
        )
        logger.info("-" * 60)
        for r in all_results:
            m = r["metrics"]
            s = r["speed"]
            logger.info(
                f"{r['model']:<12} {m['MOTA']:7.1f} {m['IDF1']:7.1f} {m['FP']:8d} {m['FN']:8d} {m['IDSW']:6d} {s['fps']:6.1f}"
            )


if __name__ == "__main__":
    main()
