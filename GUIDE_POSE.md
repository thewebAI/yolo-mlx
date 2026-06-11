# YOLO26 MLX — Pose Estimation Guide

---

## Setup (Step by Step)

### Step 1: Create & Activate Virtual Environment

```bash
cd yolo-mlx-pose

# Create a new virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

### Step 2: Install the Package & Dependencies

```bash
# Install yolo-mlx and its core dependencies (mlx, numpy, pillow, pyyaml, tqdm)
pip install -e .

# Install pose dependencies (pycocotools for COCO keypoint mAP, matplotlib for charts, opencv-python for skeleton overlays)
pip install -e ".[pose]"

# Install weight conversion dependencies (needed once to convert .pt → .npz)
pip install -e ".[convert]"

# (Optional) Install PyTorch MPS/CPU comparison benchmarks
pip install torchvision                           # optional, for MPS/CPU benchmarks
```

Runtime directories (`datasets/`, `images/`, `models/`, `results/`) are
auto-created by the scripts when needed.

**Core dependencies installed by `pip install -e .`:**

| Package | Version | Purpose |
|---------|---------|---------|
| mlx | **`>=0.30.3,<0.31`** | Apple Silicon ML framework |
| numpy | >= 2.0.0 | Array operations |
| pillow | >= 10.0.0 | Image loading |
| pyyaml | >= 6.0 | Config parsing |
| tqdm | >= 4.65.0 | Progress bars |

> **MLX version (required):** keep `mlx` in the range `>=0.30.3,<0.31`. The `<0.31`
> cap is the hard constraint — MLX `0.31.x` reproducibly hangs GPU training with
> `kIOGPUCommandBufferCallbackError` (most acutely for `yolo26x-pose`). The `>=0.30.3`
> floor is the minimum that ships the required ops. Any `0.30.x` patch works; benchmark
> numbers here were measured under the `0.30.x` line. Verify with
> `python -c "import mlx.core as mx; print(mx.__version__)"` → a `0.30.x` version.

**Pose dependencies installed by `pip install -e ".[pose]"`:**

| Package | Version | Purpose |
|---------|---------|--------|
| pycocotools | >= 2.0 | Official COCO keypoint mAP evaluation (`COCOeval(iouType='keypoints')`) |
| matplotlib | >= 3.7.0 | Chart generation |
| opencv-python | >= 4.8.0 | Skeleton/keypoint visualization |

**Conversion dependencies installed by `pip install -e ".[convert]"`:**

| Package | Version | Purpose |
|---------|---------|--------|
| torch | >= 2.0.0 | Loading .pt checkpoint files |
| ultralytics | >= 8.0.0 | Deserializing Ultralytics model objects in .pt files |
| safetensors | >= 0.4.0 | Optional safetensors output format |

**Optional benchmark dependencies (PyTorch MPS/CPU comparison only):**

| Package | Version | Purpose |
|---------|---------|--------|
| torchvision | latest | Required by Ultralytics' PyTorch trainer for the MPS/CPU comparison benchmarks |

### Step 3: Download PyTorch Models

Download the official YOLO26-pose pretrained weights (`.pt` files) from Ultralytics:

```bash
# Use the download script (downloads detection + segmentation + pose models)
bash scripts/download_yolo26_models.sh

# Or download a single pose model
bash scripts/download_yolo26_models.sh --model n-pose

# Or download pose models manually
cd models
curl -L -o yolo26n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt
curl -L -o yolo26s-pose.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt
curl -L -o yolo26m-pose.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt
curl -L -o yolo26l-pose.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-pose.pt
curl -L -o yolo26x-pose.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt
cd ..
```

After this step, `models/` should contain:
```
yolo26n-pose.pt  yolo26s-pose.pt  yolo26m-pose.pt  yolo26l-pose.pt  yolo26x-pose.pt
```

### Step 4: Convert Weights to MLX Format (.npz)

Convert the PyTorch `.pt` files to MLX `.npz` format using the built-in converter:

```bash
yolo-mlx converters --help
yolo-mlx converters convert models/yolo26n-pose.pt -o models/yolo26n-pose.npz --verify
```

The `--verify` flag checks that converted weight shapes match the source checkpoint. Repeat for `yolo26s-pose/m-pose/l-pose/x-pose`. After this step, `models/` should contain both formats:
```
yolo26n-pose.pt  yolo26n-pose.npz
yolo26s-pose.pt  yolo26s-pose.npz
yolo26m-pose.pt  yolo26m-pose.npz
yolo26l-pose.pt  yolo26l-pose.npz
yolo26x-pose.pt  yolo26x-pose.npz
```

### Step 5: Download Test Image (for Inference / Benchmark)

```bash
mkdir -p images
curl -L -o images/bus.jpg https://ultralytics.com/images/bus.jpg
```

### Step 6: Download COCO val2017 Keypoints (for Validation)

Download the COCO 2017 validation set (5,000 images, ~1 GB images + 241 MB annotations). The annotations archive includes `person_keypoints_val2017.json`, which the keypoint evaluator requires:

```bash
# Use the download script (writes to datasets/coco by default)
bash scripts/download_coco_val2017.sh datasets/coco
```

Final structure (the keypoint evaluator looks for `annotations/person_keypoints_val2017.json` + `images/val2017/`):
```
datasets/coco/
├── annotations/person_keypoints_val2017.json
├── annotations/instances_val2017.json
└── images/val2017/          # 5,000 images
```

### COCO8-Pose Dataset (Auto-Downloaded)

COCO8-Pose (8 images from COCO with keypoint labels) is used as the training dataset for the MLX training benchmark.

**No manual download needed.** The training benchmark downloads COCO8-Pose on first run into `datasets/coco8-pose/` and reuses it afterwards.

---

## Part 1: Inference (Python API)

```python
from yolo26mlx import YOLO

# Load a converted MLX pose model (task auto-detected from the "-pose" filename)
model = YOLO("models/yolo26n-pose.npz", task="pose")

# Run inference on an image
results = model.predict("images/bus.jpg")
r = results[0]

# Detection boxes (person class)
print(r.boxes.xyxy)          # (N, 4) person boxes in original-image pixels
print(r.boxes.conf)          # (N,) detection confidence

# Keypoints
kpts = r.keypoints
print(kpts.xy)               # (N, 17, 2) keypoint (x, y) in original-image pixels
print(kpts.conf)             # (N, 17)    per-keypoint visibility confidence in [0, 1]
print(kpts.data)             # (N, 17, 3) raw (x, y, confidence)

# Render the skeleton (dots + COCO 17-point limbs) and save
r.plot()                     # returns an annotated image array
r.save("bus_pose.jpg")       # writes the annotated image to disk
```

The COCO keypoint order is: nose, eyes (L/R), ears (L/R), shoulders, elbows, wrists, hips, knees, ankles. Keypoints and limbs below a visibility threshold (0.5) are skipped when plotting.

---

## Part 2: Inference (CLI)

```bash
# Predict and save an annotated skeleton image
yolo-mlx predict --model models/yolo26n-pose.npz --source images/bus.jpg --task pose

# The task is also auto-detected from the "-pose" filename, so --task is optional:
yolo-mlx predict --model models/yolo26n-pose.npz --source images/bus.jpg
```

The CLI aliases `yolo-mlx`, `yolo26`, `yolomlx`, and `yolo26mlx` are interchangeable.

---

## Part 3: Training

Train a pose model on COCO8-Pose with the pure-MLX trainer. The trainer auto-downloads the dataset on first run.

```bash
# CLI (downloads coco8-pose on first run)
yolo-mlx train --model yolo26n-pose.yaml --data coco8-pose --task pose
```

```python
# Python API
from yolo26mlx import YOLO
from yolo26mlx.engine.trainer import Trainer

# Load weights to fine-tune, or pass "yolo26n-pose.yaml" to train from scratch
model = YOLO("models/yolo26n-pose.npz", task="pose")

trainer = Trainer(model=model.model, task="pose")
trainer(data="coco8-pose.yaml", epochs=10, imgsz=640, batch=4, optimizer="auto")
```

### Loss Components

Pose training uses a **6-component** loss vector, reported per step (order matches the faithful Pose26 recipe):

| Loss | Description |
|------|-------------|
| `box` | CIoU bounding box regression loss |
| `pose` | OKS-based keypoint location loss (gain 12.0) |
| `kobj` | Keypoint visibility BCE-with-logits loss (gain 1.0) |
| `cls` | Binary cross-entropy classification loss |
| `dfl` | Distribution focal loss for box refinement |
| `rle` | Residual log-likelihood (RealNVP flow over `(pred_xy − gt_xy) / sigmoid(pred_sigma)`, gain 1.0) |

Only keypoints with ground-truth visibility `≠ 0` contribute to the `pose`/`kobj`/`rle` terms (`kpt_mask = gt_vis != 0`; flags 1 and 2 both count as present).

### YOLO-Pose Label Format

Each line in a label `.txt` file describes one instance:

```
class_id cx cy w h px1 py1 v1 px2 py2 v2 ... px17 py17 v17
```

`cx cy w h` is the normalized bounding box; each `px py v` triple is a normalized keypoint with visibility flag `v` (0 = absent, 1 = occluded, 2 = visible). For COCO this is `5 + 17×3 = 56` values per line.

Horizontal-flip augmentation flips keypoint x **and** swaps left/right joints via `flip_idx`
(`[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]`); flipping is disabled when no `flip_idx` is defined.

---

## Part 4: COCO val2017 Evaluation (Keypoint mAP)

Evaluates keypoint accuracy on the full COCO Keypoints val2017 set (5,000 images) using the official `pycocotools` protocol (`COCOeval(iouType='keypoints')`), and prints the official Ultralytics reference column alongside the MLX numbers. If `pycocotools` is unavailable, it falls back to the in-repo `PoseMetrics`.

**Script:** `scripts/evaluate_coco_pose_val.py`

### Run Full Validation (5,000 images)

```bash
# Single model
python scripts/evaluate_coco_pose_val.py --model yolo26n-pose

# All 5 models
python scripts/evaluate_coco_pose_val.py --model all
```

### Quick Test (subset)

```bash
# First 50 images only (quick sanity check; subsets inflate mAP vs the full set)
python scripts/evaluate_coco_pose_val.py --model yolo26n-pose --subset 50
```

### Common Options

```bash
# Explicit dataset root (default: search datasets/coco, coco-pose, coco8-pose)
python scripts/evaluate_coco_pose_val.py --model yolo26n-pose --data datasets/coco

# Custom confidence threshold / image size / batch size
python scripts/evaluate_coco_pose_val.py --model yolo26n-pose --conf 0.001 --imgsz 640 --batch 16

# Official-comparable scoring: score only the person-containing val split
# (the image set Ultralytics uses for its published pose mAP). Recommended when
# comparing against the official reference numbers.
python scripts/evaluate_coco_pose_val.py --model all --person-only

# Rectangular (stride-aligned) letterbox matching Ultralytics rect=True val
# (less padding; forces batch size 1). Provided for protocol comparison.
python scripts/evaluate_coco_pose_val.py --model yolo26n-pose --rect
```

> **Eval image set matters.** By default the script scores all 5,000 val2017 images. Ultralytics scores pose on the **person-containing subset** (`coco-pose/val2017.txt`); at `conf=0.001` the person-free images only add false-positive person detections and lower AP ≈0.4–0.6 pp. Use `--person-only` for an apples-to-apples comparison with the official numbers (this is how the comparison table below is produced).

### What It Reports

| Metric | Description |
|--------|-------------|
| mAP<sup>pose</sup>@0.5:0.95 | Primary keypoint metric (OKS, averaged over 0.50–0.95) |
| mAP<sup>pose</sup>@0.5 | Keypoint AP at OKS=0.50 |
| mAP<sup>box</sup>@0.5:0.95 / @0.5 | Person-box detection AP for reference |
| FPS / ms per image | Inference throughput during evaluation |

### Defaults

| Setting | Value |
|---------|-------|
| Image size | 640 |
| Confidence threshold | 0.001 |
| Batch size | 16 |

### Comparison Table (MLX vs Official)

COCO Keypoints val2017, imgsz=640, conf=0.001, official `pycocotools` keypoint evaluation, scored with `--person-only` (the person-containing val split Ultralytics uses for its published pose numbers):

| Model | MLX mAP<sup>pose</sup> 50-95 | Official | Gap | MLX mAP<sup>pose</sup> 50 | Official 50 |
|-------|------------------------------:|---------:|------:|---------------------------:|------------:|
| yolo26n-pose | **56.8** | 57.2 | −0.4 | **82.4** | 83.3 |
| yolo26s-pose | **62.7** | 63.0 | −0.3 | **85.8** | 86.6 |
| yolo26m-pose | **68.6** | 68.8 | −0.2 | **89.3** | 89.6 |
| yolo26l-pose | **69.9** | 70.4 | −0.5 | **90.2** | 90.5 |
| yolo26x-pose | **71.4** | 71.6 | −0.2 | **90.9** | 91.6 |

Official numbers are from the [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/). MLX keypoint mAP@0.5:0.95 is within **0.2–0.5 pp** of Official across all five sizes. MLX predictions are mapped back to original-image pixels and scored with the same `COCOeval(iouType='keypoints')` evaluator and the same person-containing image set Ultralytics uses for its published numbers. The MLX model is bit-faithful to PyTorch — fed an identical input tensor, its decoded keypoints/scores match the Ultralytics `.pt` model to float round-off (≈0.0001 on confidence, median 0.000 px on visible keypoints) — so the small residual is the well-documented `pycocotools`-vs-Ultralytics-validator metric difference, not the architecture or weights (conversion is verified shape- and value-exact). Scoring the full 5,000-image val set instead (`--person-only` omitted) lowers each number ≈0.4–0.6 pp because person-free images add only false-positive detections at `conf=0.001`.

The accuracy-vs-official chart is produced by:

```bash
python scripts/benchmark_yolo26_pose_accuracy_vs_official.py
# -> results/charts/yolo26_pose_accuracy_vs_official.png
```

---

## Part 5: Benchmarks

### Script Inventory

| Script | Backend | Default Output |
|--------|---------|----------------|
| `benchmark_yolo26_pose_inference.py` | MLX / MPS / CPU | `results/yolo26_pose_inference_three_way.json` |
| `benchmark_yolo26_pose_training_mlx.py` | MLX GPU | `results/yolo26_pose_mlx_training_final.json` |
| `benchmark_yolo26_pose_training_mps.py` | PyTorch MPS | `results/yolo26_pose_mps_training_final.json` |
| `benchmark_yolo26_pose_training_cpu.py` | PyTorch CPU | `results/yolo26_pose_cpu_training_final.json` |
| `benchmark_yolo26_pose_collect_results.py` | — | `results/yolo26_pose_benchmark_combined.json` |
| `benchmark_yolo26_pose_generate_charts.py` | — | `results/charts/yolo26_pose_*.png` |
| `benchmark_yolo26_pose_accuracy_vs_official.py` | — | `results/charts/yolo26_pose_accuracy_vs_official.png` |
| `evaluate_coco_pose_val.py` | MLX | `results/yolo26_pose_coco_val_results.json` |

All scripts accept `--output` to override the default output path.

### Inference Benchmark

Measures end-to-end pose inference latency on a single image across up to 3 backends: MLX GPU, PyTorch MPS, PyTorch CPU.

```bash
# All models, all backends
python scripts/benchmark_yolo26_pose_inference.py

# Specific models, more timed runs
python scripts/benchmark_yolo26_pose_inference.py --models n s --warmup 5 --runs 20

# MLX only (skip PyTorch comparisons)
python scripts/benchmark_yolo26_pose_inference.py --skip-mps --skip-cpu
```

| Metric | Description |
|--------|-------------|
| End-to-end latency (ms) | Full `model.predict(image)` including pre/post-processing + keypoint decode |
| Forward-pass-only (ms) | Model forward only |
| FPS | Throughput (1000 / mean_ms) |
| Peak memory (MB) | MLX Metal memory usage |
| Speedup ratios | MLX vs MPS, MLX vs CPU |

**Defaults:** 3 warmup runs, 10 timed runs, 640×640, models n/s/m/l/x.

#### Benchmark Results

End-to-end inference FPS (imgsz=640, 20 timed / 5 warmup runs, Apple M3 Pro):

| Model | MLX FPS | MPS FPS | CPU FPS | MLX vs MPS | MLX vs CPU |
|-------|--------:|--------:|--------:|-----------:|-----------:|
| yolo26n-pose | **103.7** | 87.4 | 24.9 | **1.19×** | **4.16×** |
| yolo26s-pose | **70.9** | 65.0 | 15.0 | **1.09×** | **4.72×** |
| yolo26m-pose | **36.9** | 35.7 | 7.7 | **1.03×** | **4.81×** |
| yolo26l-pose | **30.9** | 30.4 | 6.1 | **1.02×** | **5.09×** |
| yolo26x-pose | **17.5** | 16.4 | 3.6 | **1.07×** | **4.90×** |

MLX is the fastest backend on every model size — `1.02×–1.19×` over PyTorch MPS and `4.16×–5.09×` over PyTorch CPU end-to-end. The smaller models (n, s) gain the most from MLX's Metal-optimized graph and `mx.compile` JIT.

### MLX Training Benchmark

Trains YOLO26-pose models with the pure-MLX implementation and measures time, loss, keypoint mAP, and memory on COCO8-Pose.

```bash
python scripts/benchmark_yolo26_pose_training_mlx.py
python scripts/benchmark_yolo26_pose_training_mlx.py --models n s --epochs 5 --batch 2
```

Same CLI flags as the other training scripts (`--models`, `--epochs`, `--batch`, `--lr`, `--output`).

| Setting | Value |
|---------|-------|
| Epochs | 10 |
| Batch size | 4 |
| Learning rate | 0.002 (auto-LR for nc=1: `0.002 × 5 / (4 + 1)`) |
| Optimizer | auto (AdamW for ≤10k iter, MuSGD otherwise) |
| Dataset | COCO8-Pose (8 images) |
| Validation | After training |

> COCO8-Pose has only 8 images (`val == train`), so its keypoint mAP is a *fit* metric for smoke-testing the training loop, not a generalization metric. For an accuracy comparison against the published Ultralytics numbers use **Part 4** (full COCO val2017).

### PyTorch MPS / CPU Training Benchmarks

> **Requires:** `pip install ultralytics torch torchvision`

```bash
python scripts/benchmark_yolo26_pose_training_mps.py --models n s --epochs 5 --batch 2
python scripts/benchmark_yolo26_pose_training_cpu.py --models n s --epochs 5 --batch 2 --threads 4
```

Use the Ultralytics trainer on `coco8-pose` and report keypoint mAP + box mAP. The CPU script adds `--threads` to control `torch.set_num_threads()`.

> **MPS note:** Ultralytics' end-to-end pose loss moves a **float64** `RLE_WEIGHT` array to the training device, and MPS cannot host float64 tensors (`TypeError: Cannot convert a MPS Tensor to float64 dtype`). The MPS benchmark applies `scripts/_mps_pose_perf_patch.py` (`apply_mps_pose_patch()`), which casts that module-level array to float32 before training. The weights are per-keypoint scale factors, so float32 is numerically inconsequential and keeps the MPS run comparable to CPU/MLX. CPU is unaffected.

#### Cross-Backend Training Results

Training time per epoch on COCO8-Pose (10 epochs, batch=4, Apple M3 Pro). Each backend runs one discarded 1-epoch warmup before timing so one-time kernel compilation / process init is not charged to the first model:

| Model | MLX time/epoch (s) | MPS time/epoch (s) | CPU time/epoch (s) | MLX vs MPS | MLX vs CPU | MPS vs CPU |
|-------|---------------------:|---------------------:|---------------------:|-----------:|-----------:|-----------:|
| yolo26n-pose | **0.52** | 1.73 | 1.49 | **3.33×** | **2.87×** | 0.86× |
| yolo26s-pose | **0.79** | 1.95 | 2.38 | **2.47×** | **3.01×** | 1.22× |
| yolo26m-pose | **1.29** | 2.41 | 4.11 | **1.87×** | **3.19×** | 1.71× |
| yolo26l-pose | **1.58** | 2.79 | 4.96 | **1.77×** | **3.14×** | 1.78× |
| yolo26x-pose | **2.43** | 3.98 | 7.72 | **1.64×** | **3.18×** | 1.94× |

MLX is the fastest backend across all five model sizes — `1.64×–3.33×` over PyTorch MPS and `2.87×–3.19×` over PyTorch CPU. The MLX-vs-MPS margin is largest on the smallest model and tapers as the models become compute-bound — the same pattern as inference, since MLX's lower per-dispatch overhead matters most when each step does little work. MPS edges below CPU only on `yolo26n-pose` (0.86×): for that tiny workload the MPS host↔device dispatch/transfer overhead marginally exceeds CPU compute, and MPS overtakes CPU from `s` upward.

> COCO8-Pose has only 8 images, so the post-training keypoint mAP recorded in each JSON is a *fit* metric for smoke-testing the loop, not a generalization metric. PyTorch MPS and CPU are bit-identical here (same Ultralytics trainer, deterministic seed); MLX trains the same 6-component loss with its own initialization. For a real accuracy comparison against the published Ultralytics numbers use **Part 4** (full COCO val2017).

#### Real-Accuracy Fine-Tune (MLX, COCO-pose 1k subset → COCO val2017)

To check the MLX trainer on real data (not the 8-image smoke set), each model was fine-tuned from the official converted weights for 10 epochs on a 1,000-image COCO-pose train subset (`datasets/coco_pose_sub1k.yaml`, batch=4, AdamW, lr=1e-4 with warmup+cosine), then evaluated on the full COCO val2017 person split (`--person-only`):

```bash
python scripts/benchmark_yolo26_pose_training_mlx.py \
  --data datasets/coco_pose_sub1k.yaml --models n s m l x \
  --epochs 10 --batch 4 --lr 0.0001 --no-warmup \
  --output results/ft/yolo26_pose_mlx_finetune_sub1k.json
# Evaluate a fine-tuned checkpoint (name the file yolo26<scale>-pose so the
# loader picks the right scale), e.g.:
cp results/mlx_runs/yolo26m-pose/last.safetensors results/ft/yolo26m-pose.safetensors
python scripts/evaluate_coco_pose_val.py --model yolo26m-pose --data datasets/coco \
  --person-only --weights results/ft/yolo26m-pose.safetensors
```

| Model | Pretrained mAP<sup>pose</sup> 50-95 / 50 | Fine-tuned (1k·10ep) 50-95 / 50 | Δ 50-95 | MLX time/epoch |
|-------|------------------------------------------:|---------------------------------:|--------:|---------------:|
| yolo26n-pose | 56.8 / 82.4 | 55.8 / 81.7 | −1.0 | 100.4s |
| yolo26s-pose | 62.7 / 85.8 | 59.7 / 84.6 | −3.0 | 155.2s |
| yolo26m-pose | 68.6 / 89.3 | 64.9 / 87.9 | −3.7 | 267.8s |
| yolo26l-pose | 69.9 / 90.2 | 66.6 / 88.5 | −3.3 | 317.4s |
| yolo26x-pose | 71.4 / 90.9 | 65.9 / 89.0 | −5.5 | 521.3s |

This is a **training-loop validation, not an accuracy win**: the MLX trainer fine-tunes end-to-end (loss decreases monotonically, train-subset fit mAP50 reaches 0.95–0.97, no NaNs) and produces valid models scoring 56–67 mAP on real val2017. Val mAP drops modestly below the official-pretrained baseline because 10 epochs on 1,000 images cannot match the official ~56k-image schedule, and larger models overfit the small subset faster (hence the larger drop for `x`). At the default short-run LR (0.002) the drop is severe (`n`: 56.8→29.8); lr=1e-4 is the conservative setting used above. For a true accuracy-matching run, train on full `coco-pose` (~56k images) with the standard schedule.

### Collect Results & Generate Charts

> **Requires:** `pip install matplotlib`

```bash
# Combine all benchmark JSONs into one file
python scripts/benchmark_yolo26_pose_collect_results.py

# Generate comparison charts (PNG; use --format pdf / --dpi 300 / --output for variants)
python scripts/benchmark_yolo26_pose_generate_charts.py
```

#### Charts Generated (in `results/charts/`)

| Chart | File | Description |
|-------|------|-------------|
| Inference latency | `yolo26_pose_inference_latency.png` | Latency across backends |
| Inference FPS | `yolo26_pose_inference_fps.png` | Throughput (frames per second) |
| Training time | `yolo26_pose_training_time.png` | Training time across backends |
| Speedup | `yolo26_pose_speedup.png` | MLX vs CPU and MLX vs MPS factors |
| Accuracy | `yolo26_pose_accuracy.png` | Post-training keypoint mAP across backends |
| Memory usage | `yolo26_pose_memory.png` | Peak memory comparison |
| Summary dashboard | `yolo26_pose_summary.png` | 2×2 grid: latency, training time, speedup, accuracy |
| Accuracy vs Official | `yolo26_pose_accuracy_vs_official.png` | MLX keypoint mAP next to the official reference |

> Charts whose underlying benchmark has not been run are skipped automatically (e.g. training/accuracy/memory charts require the training benchmarks first).

---

## Appendix: Pose-Specific Reference

### YOLO26-Pose Model Sizes

COCO Keypoints val2017, 640px, single `person` class. Official numbers from the [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/).

| Model | Params (M) | FLOPs (B) | Official mAP<sup>pose</sup> 50-95 | Official mAP<sup>pose</sup> 50 |
|-------|-----------:|----------:|----------------------------------:|-------------------------------:|
| yolo26n-pose | 2.9 | 7.5 | 57.2 | 83.3 |
| yolo26s-pose | 10.4 | 23.9 | 63.0 | 86.6 |
| yolo26m-pose | 21.5 | 73.1 | 68.8 | 89.6 |
| yolo26l-pose | 25.9 | 91.3 | 70.4 | 90.5 |
| yolo26x-pose | 57.6 | 201.7 | 71.6 | 91.6 |

### Keypoint Decode (Inference)

The head decodes raw keypoint predictions to image space:

```
x = (dx · 2.0 + (anchor_x − 0.5)) · stride
y = (dy · 2.0 + (anchor_y − 0.5)) · stride
visibility = sigmoid(v_logit)
```

This is distinct from the loss-side decode, which stays in grid space (no stride, no sigmoid).

### Dataset Config

`src/yolo26mlx/cfg/datasets/coco8-pose.yaml`: `nc: 1`, names `[person]`, `kpt_shape: [17, 3]`,
`flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]`.

---

## Quick Reference

```bash
# Activate environment
cd yolo-mlx-pose
source .venv/bin/activate

# ── Inference (CLI, save skeleton) ──
yolo-mlx predict --model models/yolo26n-pose.npz --source images/bus.jpg --task pose

# ── Inference benchmark (all models, MLX only) ──
python scripts/benchmark_yolo26_pose_inference.py --skip-mps --skip-cpu

# ── Full COCO val2017 keypoint mAP (all models) ──
python scripts/evaluate_coco_pose_val.py --model all

# ── Quick sanity check (1 model, 50 images) ──
python scripts/evaluate_coco_pose_val.py --model yolo26n-pose --subset 50

# ── MLX training benchmark (all models) ──
python scripts/benchmark_yolo26_pose_training_mlx.py --models n s m l x --epochs 10 --batch 4

# ── MPS / CPU training benchmarks (all models) ──
python scripts/benchmark_yolo26_pose_training_mps.py --models n s m l x --epochs 10 --batch 4
python scripts/benchmark_yolo26_pose_training_cpu.py --models n s m l x --epochs 10 --batch 4

# ── Collect & chart ──
python scripts/benchmark_yolo26_pose_collect_results.py
python scripts/benchmark_yolo26_pose_generate_charts.py
python scripts/benchmark_yolo26_pose_accuracy_vs_official.py
```
