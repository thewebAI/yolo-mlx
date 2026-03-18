# YOLO26 MLX — Inference Benchmarking & COCO Validation Guide

---

## Setup (Step by Step)

### Step 1: Create & Activate Virtual Environment

```bash
cd yolo-mlx

# Create a new virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

### Step 2: Install the Package & Dependencies

```bash
# Install yolo-mlx and its core dependencies (mlx, numpy, pillow, pyyaml, tqdm)
pip install -e .

# Install weight conversion dependencies (needed once to convert .pt → .npz)
pip install -e ".[convert]"

# Install COCO evaluation and chart generation tools
pip install pycocotools matplotlib

# (Optional) Install PyTorch MPS/CPU comparison benchmarks
pip install torchvision                           # optional, for MPS/CPU benchmarks
```

Runtime directories (`datasets/`, `images/`, `models/`, `results/`) are
auto-created by the scripts when needed.

**Core dependencies installed by `pip install -e .`:**

| Package | Version | Purpose |
|---------|---------|---------|
| mlx | >= 0.30.3 | Apple Silicon ML framework |
| numpy | >= 1.24.0 | Array operations |
| pillow | >= 10.0.0 | Image loading |
| pyyaml | >= 6.0 | Config parsing |
| tqdm | >= 4.65.0 | Progress bars |

**Conversion dependencies installed by `pip install -e ".[convert]"`:**

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0.0 | Loading .pt checkpoint files |
| ultralytics | >= 8.0.0 | Deserializing Ultralytics model objects in .pt files |
| safetensors | >= 0.4.0 | Optional safetensors output format |

**Additional dependencies installed separately:**

| Package | Version | Purpose |
|---------|---------|---------|
| pycocotools | latest | Official COCO mAP evaluation |
| matplotlib | latest | Chart generation |
| torchvision | latest | PyTorch MPS/CPU comparison (optional) |

### Step 3: Download PyTorch Models

Download the official YOLO26 pretrained weights (`.pt` files) from Ultralytics:

```bash
# Use the download script
bash scripts/download_yolo26_models.sh

# Or download manually
cd models
curl -L -o yolo26n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
curl -L -o yolo26s.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
curl -L -o yolo26m.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt
curl -L -o yolo26l.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt
curl -L -o yolo26x.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt
cd ..
```

After this step, `models/` should contain:
```
yolo26n.pt  yolo26s.pt  yolo26m.pt  yolo26l.pt  yolo26x.pt
```

### Step 4: Convert Weights to MLX Format (.npz)

Convert the PyTorch `.pt` files to MLX `.npz` format using the built-in converter:

```bash
yolo-mlx converters --help
yolo-mlx converters convert models/yolo26n.pt -o models/yolo26n.npz --verify
```

The `--verify` flag checks that converted weight shapes are correct. Repeat the same command for `yolo26s/m/l/x`. After this step, `models/` should contain both formats:
```
yolo26n.pt  yolo26n.npz
yolo26s.pt  yolo26s.npz
yolo26m.pt  yolo26m.npz
yolo26l.pt  yolo26l.npz
yolo26x.pt  yolo26x.npz
```

### Step 5: Download Test Image (for Inference Benchmark)

```bash
mkdir -p images
curl -L -o images/bus.jpg https://ultralytics.com/images/bus.jpg
```

### Step 6: Download COCO val2017 Dataset (for Validation)

Download the COCO 2017 validation set (5,000 images, ~1 GB images + 241 MB annotations):

```bash
# Use the download script
bash scripts/download_coco_val2017.sh datasets/coco

# Or download manually
mkdir -p datasets/coco/images datasets/coco/annotations datasets/coco/labels

# Download val2017 images (~1 GB)
curl -L -o datasets/coco/images/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip datasets/coco/images/val2017.zip -d datasets/coco/images/
rm datasets/coco/images/val2017.zip

# Download annotations (~241 MB)
curl -L -o datasets/coco/annotations/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip datasets/coco/annotations/annotations_trainval2017.zip -d datasets/coco/
rm datasets/coco/annotations/annotations_trainval2017.zip

# Download YOLO-format labels (pre-converted from Ultralytics)
curl -L -o datasets/coco/labels/val2017.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip
unzip datasets/coco/labels/val2017.zip -d datasets/coco/
rm datasets/coco/labels/val2017.zip
```

The dataset config is already at `configs/coco.yaml`. Final structure:
```
datasets/coco/
├── annotations/instances_val2017.json
├── images/val2017/          # 5,000 images
└── labels/val2017/          # YOLO-format labels
```

---

## Part 1: Inference Benchmarking

Measures end-to-end inference latency on a single image across up to 3 backends: MLX GPU, PyTorch MPS, PyTorch CPU.

**Script:** `scripts/benchmark_yolo26_inference.py`

### Run All Models, All Backends

```bash
python scripts/benchmark_yolo26_inference.py
```

### Common Options

```bash
# Specific models only
python scripts/benchmark_yolo26_inference.py --models n s

# More timed runs for stable results
python scripts/benchmark_yolo26_inference.py --runs 20

# MLX only (skip PyTorch comparisons)
python scripts/benchmark_yolo26_inference.py --skip-mps --skip-cpu

# Custom warmup
python scripts/benchmark_yolo26_inference.py --warmup 5 --runs 15

# Custom output path
python scripts/benchmark_yolo26_inference.py --output my_results.json
```

### What It Measures

| Metric | Description |
|--------|-------------|
| End-to-end latency (ms) | Full `model.predict(image)` including preprocessing + postprocessing |
| Forward-pass-only (ms) | Model inference only (no pre/post processing) |
| FPS | Throughput (1000 / mean_ms) |
| Peak memory (MB) | MLX Metal memory usage |
| Speedup ratios | MLX vs MPS, MLX vs CPU |

### Output

- **Console:** Summary table with latency, FPS, and speedups for all models
- **JSON:** `results/yolo26_inference_three_way.json` (override with `--output`)

### Defaults

| Setting | Value |
|---------|-------|
| Warmup runs | 3 |
| Timed runs | 10 |
| Image size | 640×640 |
| Models | n, s, m, l, x |

---

## Part 2: COCO val2017 Validation (mAP)

Evaluates model accuracy on the full COCO val2017 set (5,000 images) using the official pycocotools COCO evaluation protocol.

**Script:** `scripts/evaluate_coco_val.py`

### Run Full Validation (5,000 images)

```bash
# Single model
python scripts/evaluate_coco_val.py --model yolo26n --data datasets/coco

# All 5 models
python scripts/evaluate_coco_val.py --model all --data datasets/coco
```

### Quick Test (subset)

```bash
# First 100 images only (for quick sanity check)
python scripts/evaluate_coco_val.py --model yolo26n --data datasets/coco --subset 100
```

### Common Options

```bash
# Custom confidence threshold
python scripts/evaluate_coco_val.py --model yolo26n --data datasets/coco --conf 0.001

# Custom NMS IoU threshold
python scripts/evaluate_coco_val.py --model yolo26n --data datasets/coco --iou 0.7

# Custom image size and batch size
python scripts/evaluate_coco_val.py --model yolo26n --data datasets/coco --imgsz 640 --batch 32

# Multiple specific models
python scripts/evaluate_coco_val.py --model yolo26n yolo26s --data datasets/coco

# Verbose output (per-image progress)
python scripts/evaluate_coco_val.py --model yolo26n --data datasets/coco --verbose

# Custom output directory
python scripts/evaluate_coco_val.py --model yolo26n --data datasets/coco --output my_results/
```

### What It Reports

| Metric | Description |
|--------|-------------|
| mAP@0.5:0.95 | Primary COCO metric (averaged over IoU 0.50–0.95) |
| mAP@0.5 | AP at IoU=0.50 |
| mAP@0.75 | AP at IoU=0.75 |
| mAP (small/medium/large) | AP by object size |
| Images/second | Throughput during evaluation |

### Output

- **Console:** Full pycocotools evaluation summary (12-metric table)
- **JSON:** `results/` directory (override with `--output`)

### Defaults

| Setting | Value |
|---------|-------|
| Image size | 640 |
| Confidence threshold | 0.001 |
| NMS IoU threshold | 0.7 |
| Max detections | 300 |
| Batch size | 16 |

---

## Quick Reference

```bash
# Activate environment
cd yolo-mlx
source .venv/bin/activate

# ── Inference benchmark (all models, MLX only) ──
python scripts/benchmark_yolo26_inference.py --skip-mps --skip-cpu

# ── Full COCO validation (all models) ──
python scripts/evaluate_coco_val.py --model all --data datasets/coco

# ── Quick sanity check (1 model, 100 images) ──
python scripts/evaluate_coco_val.py --model yolo26n --data datasets/coco --subset 100
```
