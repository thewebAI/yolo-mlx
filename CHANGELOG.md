# Changelog

## 0.4.0 — 2026-06-10

- **Pose**: Native MLX keypoint estimation (`Pose`/`Pose26` head) — inference, training, and COCO Keypoints val2017 OKS keypoint mAP eval
- **Pose**: Keypoint mAP within 0.2–0.5 pp of Ultralytics across all 5 sizes (person-only protocol: n 56.8, s 62.7, m 68.6, l 69.9, x 71.4)
- **Pose**: Faithful cv2 `LetterBox` (INTER_LINEAR + round dims/pad) for eval/predict parity; evaluator `--person-only` / `--rect` / `--weights` options
- **Benchmarks**: Added pose inference, COCO val mAP, and training benchmark scripts and charts; chart generators skip empty/absent-backend series
- **Docs**: Added `GUIDE_POSE.md` and pose Quick Start sections

## 0.3.1 — 2026-05-05

- **Trainer**: Auto-downloaded datasets (`coco128`, `coco128-seg`) now land at `./datasets/<name>/` (CWD-relative) instead of inside the venv's `lib/python3.10/datasets/` — matches the README for both editable and PyPI installs

## 0.3.0 — 2026-05-05

- **Segmentation**: Native MLX instance segmentation (`Segment26` head + `Proto26`) — inference, training, and COCO val2017 mAP eval
- **Segmentation**: Mask mAP within 0.3–0.4 pp of Ultralytics across all 5 sizes; MLX seg training fastest on every size (1.25×–3.31× vs MPS)
- **Trainer**: Added MLX `AdamW` optimizer with `optimizer="auto"` (AdamW for short fine-tunes, MuSGD otherwise)
- **Trainer**: Tree-form `mx.eval(tree)` syncs fix upstream `[eval] Attempting to eval an array without a primitive` crash on mlx 0.31.x
- **Packaging**: Pinned `mlx>=0.30.3,<0.31`; anchored `.gitignore` patterns to repo root
- **Benchmarks**: Added segmentation inference, COCO val mAP, and training benchmark scripts and charts
- **Docs**: Added `GUIDE_SEGMENTATION.md` and segmentation Quick Start sections

## 0.2.0 — 2026-04-01

- **Tracking**: Batched Kalman filter updates (single `mx.linalg.inv` call per association stage)
- **Tracking**: Batch-precomputed coordinates reduce MLX graph dispatch overhead
- **Tracking**: MLX tracking now matches or exceeds PyTorch MPS speed
- **Packaging**: Moved `scipy` from core dependencies to `[tracking]` extra
- **Packaging**: Added clear error message when tracking dependencies are missing
- **Benchmarks**: Added tracking benchmark scripts and charts

## 0.1.0 — 2026-03-18

- Initial release: detection, training, COCO validation
