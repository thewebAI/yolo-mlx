# YOLO26 MLX — Training Benchmarking Guide

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
# Install yolo26mlx and its core dependencies (mlx, numpy, pillow, pyyaml, tqdm)
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
| torchvision | latest | PyTorch MPS/CPU training comparison (optional) |

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
yolo26 converters --help
yolo26 converters convert models/yolo26n.pt -o models/yolo26n.npz --verify
```

The `--verify` flag checks that converted weight shapes are correct. Repeat the same command for `yolo26s/m/l/x`. After this step, `models/` should contain both formats:
```
yolo26n.pt  yolo26n.npz
yolo26s.pt  yolo26s.npz
yolo26m.pt  yolo26m.npz
yolo26l.pt  yolo26l.npz
yolo26x.pt  yolo26x.npz
```

### COCO128 Dataset (Auto-Downloaded)

COCO128 (128 images from COCO) is used as the training dataset for benchmarks.

**No manual download needed.** The training scripts automatically download COCO128 (~7 MB) on first run into `datasets/coco128/`. After the first run, the dataset is cached locally and reused.

---

## Training Benchmark Scripts

There are 3 training scripts (one per backend) plus 2 utility scripts:

| Script | Backend | Default Output |
|--------|---------|----------------|
| `benchmark_yolo26_training_mlx.py` | MLX GPU | `results/yolo26_mlx_training_final.json` |
| `benchmark_yolo26_training_mps.py` | PyTorch MPS | `results/yolo26_mps_training_final.json` |
| `benchmark_yolo26_training_cpu.py` | PyTorch CPU | `results/yolo26_cpu_training_final.json` |
| `benchmark_yolo26_collect_results.py` | — | `results/yolo26_benchmark_combined.json` |
| `benchmark_yolo26_generate_charts.py` | — | `results/charts/*.png` |

All scripts accept `--output` to override the default output path.

---

## Part 1: MLX Training Benchmark

Trains YOLO26 models using the pure MLX implementation and measures time, loss, mAP, and memory.

**Script:** `scripts/benchmark_yolo26_training_mlx.py`

### Run All Models

```bash
python scripts/benchmark_yolo26_training_mlx.py
```

### Common Options

```bash
# Specific models only
python scripts/benchmark_yolo26_training_mlx.py --models n s

# Custom epochs and batch size
python scripts/benchmark_yolo26_training_mlx.py --epochs 5 --batch 2

# Custom learning rate
python scripts/benchmark_yolo26_training_mlx.py --lr 0.0001

# Custom output path
python scripts/benchmark_yolo26_training_mlx.py --output my_results.json

# All options combined
python scripts/benchmark_yolo26_training_mlx.py --models n s m l x --epochs 10 --batch 4
```

### What It Measures

| Metric | Description |
|--------|-------------|
| Training time (s) | Total wall-clock time for all epochs |
| Time/epoch (s) | Average time per epoch |
| Final loss | Loss value at end of training |
| mAP@0.5 | Accuracy after training (validation on COCO128) |
| Peak memory (MB) | MLX Metal peak memory usage |

### Defaults

| Setting | Value |
|---------|-------|
| Epochs | 10 |
| Batch size | 4 |
| Learning rate | 0.000119 (MuSGD auto) |
| Optimizer | MuSGD (Muon + Nesterov SGD) |
| Dataset | COCO128 (128 images) |
| Validation | After training (not during) |

---

## Part 2: PyTorch MPS Training Benchmark

> **Requires:** `pip install ultralytics torch torchvision`

```bash
python scripts/benchmark_yolo26_training_mps.py
python scripts/benchmark_yolo26_training_mps.py --models n s --epochs 5 --batch 2
```

Same CLI flags as MLX script (`--models`, `--epochs`, `--batch`, `--lr`, `--output`). Uses PyTorch MPS backend with Ultralytics trainer. Default learning rate is 0.00001 (differs from MLX's MuSGD auto rate).

---

## Part 3: PyTorch CPU Training Benchmark

> **Requires:** `pip install ultralytics torch torchvision`

```bash
python scripts/benchmark_yolo26_training_cpu.py
python scripts/benchmark_yolo26_training_cpu.py --models n s --epochs 5 --batch 2

# Control CPU thread count
python scripts/benchmark_yolo26_training_cpu.py --threads 4
```

Same CLI flags as MPS script, plus `--threads` to control `torch.set_num_threads()`. Default learning rate is 0.00001. Slowest backend — useful as a baseline.

---

## Part 4: Collect Results & Generate Charts

After running one or more training benchmarks, combine results and generate charts:

> **Requires:** `pip install matplotlib`

```bash
# Combine all JSON results into one file
python scripts/benchmark_yolo26_collect_results.py

# Generate comparison charts
python scripts/benchmark_yolo26_generate_charts.py

# PDF format (for publications)
python scripts/benchmark_yolo26_generate_charts.py --format pdf
```

### Charts Generated (in `results/charts/`)

- Inference latency comparison (bar chart)
- Inference throughput (FPS)
- Training time comparison
- Speedup comparison (MLX vs CPU, MLX vs MPS)
- Memory usage comparison
- Accuracy (mAP) comparison

---

## Quick Reference

```bash
# Activate environment
cd yolo-mlx
source .venv/bin/activate

# ── MLX training benchmark (all models) ──
python scripts/benchmark_yolo26_training_mlx.py --models n s m l x --epochs 10 --batch 4

# ── MPS training benchmark (all models) ──
python scripts/benchmark_yolo26_training_mps.py --models n s m l x --epochs 10 --batch 4

# ── CPU training benchmark (all models) ──
python scripts/benchmark_yolo26_training_cpu.py --models n s m l x --epochs 10 --batch 4

# ── Collect & chart ──
python scripts/benchmark_yolo26_collect_results.py
python scripts/benchmark_yolo26_generate_charts.py
```
