# YOLO26 MLX — Tracking Guide

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

# Install tracking dependencies (OpenCV for video I/O, lap for assignment)
pip install -e ".[tracking]"

# Install weight conversion dependencies (needed once to convert .pt → .npz)
pip install -e ".[convert]"
```

Runtime directories (`datasets/`, `images/`, `models/`, `results/`) are
auto-created by the scripts when needed.

**Core dependencies installed by `pip install -e .`:**

| Package | Version | Purpose |
|---------|---------|---------|
| mlx | >= 0.30.3 | Apple Silicon ML framework |
| numpy | >= 2.0.0 | Array operations |
| pillow | >= 10.0.0 | Image loading |
| pyyaml | >= 6.0 | Config parsing |
| tqdm | >= 4.65.0 | Progress bars |

**Conversion dependencies installed by `pip install -e ".[convert]"`:**

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0.0 | Loading .pt checkpoint files |
| ultralytics | >= 8.0.0 | Deserializing Ultralytics model objects in .pt files |
| safetensors | >= 0.4.0 | Optional safetensors output format |

**Tracking dependencies installed by `pip install -e ".[tracking]"`:**

| Package | Version | Purpose |
|---------|---------|--------|
| opencv-python | >= 4.8.0 | Video I/O, display, annotation rendering |
| lap | >= 0.5.0 | Linear assignment (Hungarian algorithm) |
| scipy | >= 1.11.0 | Fallback assignment solver, scientific computing |

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

### Step 5: Download MOT17 Dataset & Create Sample Video

```bash
# Download MOT17 dataset (~5.5 GB) for evaluation and demo videos
bash scripts/download_mot17.sh

# Create a short sample pedestrian video (~3s, 1080p) from MOT17 frames
python scripts/create_sample_video.py
```

This creates `images/pedestrians.mp4` from the first 90 frames of MOT17-09-SDP.

Custom options for the video script:
```bash
# Custom sequence and frame count
python scripts/create_sample_video.py --sequence MOT17-04-SDP --frames 60

# Custom output path
python scripts/create_sample_video.py --output my_video.mp4
```

---

## Tracker Configurations

Two tracker configurations are included, located at `src/yolo26mlx/cfg/trackers/`:

### ByteTrack (`bytetrack.yaml`) — Default

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tracker_type` | bytetrack | Tracker algorithm |
| `track_high_thresh` | 0.25 | High detection confidence threshold |
| `track_low_thresh` | 0.1 | Low detection confidence threshold (second association) |
| `new_track_thresh` | 0.25 | Minimum confidence to create a new track |
| `track_buffer` | 30 | Frames to keep lost tracks before deletion |
| `match_thresh` | 0.8 | IoU matching threshold |
| `fuse_score` | true | Fuse detection confidence with IoU cost |

### BoT-SORT (`botsort.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tracker_type` | botsort | Tracker algorithm |
| `track_high_thresh` | 0.25 | High detection confidence threshold |
| `track_low_thresh` | 0.1 | Low detection confidence threshold |
| `new_track_thresh` | 0.25 | Minimum confidence to create a new track |
| `track_buffer` | 30 | Frames to keep lost tracks |
| `match_thresh` | 0.8 | IoU matching threshold |
| `fuse_score` | true | Fuse detection confidence with IoU cost |
| `gmc_method` | sparseOptFlow | Camera motion compensation method |
| `proximity_thresh` | 0.5 | Proximity threshold for re-id |
| `appearance_thresh` | 0.8 | Appearance threshold for re-id |
| `with_reid` | false | Enable re-identification features |
| `model` | auto | Model path for re-id features |

---

## Part 1: Video Tracking (Python API)

### Basic Video Tracking

```python
from yolo26mlx import YOLO

model = YOLO("models/yolo26n.npz")

# Track pedestrians — saves annotated output to results/pedestrians_tracked.mp4
results = model.track("images/pedestrians.mp4", conf=0.25, save=True)

# Access per-frame results
for r in results:
    if r.boxes.is_track:
        print(r.boxes.id)     # track IDs (persistent across frames)
        print(r.boxes.xyxy)   # bounding boxes
```

### Webcam Tracking

```python
# Real-time tracking from webcam (press 'q' to quit)
results = model.track(0, conf=0.25, show=True)
```

### Frame-by-Frame Control

```python
from yolo26mlx import YOLO

model = YOLO("models/yolo26n.npz")
for result in model.track("video.mp4", stream=True):
    boxes = result.boxes
    if boxes.is_track:
        for tid, box in zip(boxes.id, boxes.xyxy):
            print(f"Track {tid}: {box}")
```

### `model.track()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | str / int / ndarray | **required** | Video path, webcam index (0), or numpy frame |
| `tracker` | str | `"bytetrack.yaml"` | Tracker config YAML (`bytetrack.yaml` or `botsort.yaml`) |
| `conf` | float | `0.25` | Detection confidence threshold |
| `imgsz` | int | `640` | Input image size |
| `persist` | bool | `False` | Keep tracker state between calls (for frame-by-frame API) |
| `stream` | bool | `False` | Return generator instead of list |
| `show` | bool | `False` | Display annotated frames with `cv2.imshow` |
| `save` | bool | `False` | Save annotated video to `results/` |
| `vid_stride` | int | `1` | Process every Nth frame |

### Output Locations

| Output | Location |
|--------|----------|
| Annotated video | `results/<video_name>_tracked.mp4` |
| Displayed frames | `cv2.imshow` window (when `show=True`) |

---

## Part 2: Video Tracking (CLI)

The command-line interface provides the same tracking functionality without writing Python code.

### Basic Usage

```bash
# Track a video file
yolo-mlx track --model models/yolo26n.npz --source images/pedestrians.mp4 --save

# Track with display
yolo-mlx track --model models/yolo26n.npz --source images/pedestrians.mp4 --show --save

# Webcam tracking
yolo-mlx track --model models/yolo26n.npz --source 0 --show
```

### CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | **required** | Path to model weights (.npz, .safetensors, .pt) |
| `--source` | str | **required** | Video path, image path, or webcam index (0) |
| `--tracker` | str | `bytetrack.yaml` | Tracker config YAML |
| `--conf` | float | `0.25` | Confidence threshold |
| `--imgsz` | int | `640` | Input image size |
| `--show` | flag | — | Display results with cv2.imshow |
| `--save` | flag | — | Save annotated video to disk |
| `--vid-stride` | int | `1` | Process every Nth frame |
| `-q` / `--quiet` | flag | — | Suppress informational logs |

### Common Examples

```bash
# Use BoT-SORT instead of ByteTrack
yolo-mlx track --model models/yolo26n.npz --source video.mp4 --tracker botsort.yaml --save

# Higher confidence threshold to reduce false positives
yolo-mlx track --model models/yolo26n.npz --source video.mp4 --conf 0.5 --save

# Skip frames for faster processing (process every 2nd frame)
yolo-mlx track --model models/yolo26n.npz --source video.mp4 --vid-stride 2 --save

# Larger input size for better accuracy
yolo-mlx track --model models/yolo26l.npz --source video.mp4 --imgsz 1280 --save
```

---

## Part 3: Tracking Demo Script

A standalone demo script is provided for quick experiments:

**Script:** `scripts/track_demo.py`

```bash
# Video file tracking (display + save)
python scripts/track_demo.py --model models/yolo26n.npz --source images/pedestrians.mp4 --show --save

# Webcam tracking
python scripts/track_demo.py --model models/yolo26n.npz --source 0 --show

# Frame-by-frame custom processing mode
python scripts/track_demo.py --model models/yolo26n.npz --source images/pedestrians.mp4 --mode framewise --show
```

---

## Part 4: Training a Detector for Tracking

Tracking uses standard detection models — no separate tracking-specific training is needed.
Any YOLO26 model trained on detection can be used directly with `model.track()`.
To improve tracking on a custom domain, fine-tune a detection model on the objects
you want to track, then use it for tracking.

```python
from yolo26mlx import YOLO

# Step 1: Fine-tune on your detection dataset
model = YOLO("models/yolo26n.npz")
results = model.train(
    data="coco128",
    epochs=10,
    batch=4,
    imgsz=640,
    project="runs/train",
    name="my_detector",
)

# Step 2: Use the fine-tuned model for tracking
model = YOLO("runs/train/my_detector/best.safetensors")
results = model.track("video.mp4", conf=0.25, save=True)
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Number of training epochs |
| `batch` | 16 | Batch size |
| `imgsz` | 640 | Input image size |
| `patience` | 50 | Early stopping patience (epochs) |
| `save_period` | -1 | Save checkpoint every N epochs (-1 to disable) |

### Output Locations

| Output | Location |
|--------|----------|
| Training checkpoints | `runs/train/<name>/best.safetensors` |
| Training logs | Console (INFO level) |
| Tracked output video | `results/<video_name>_tracked.mp4` |

---

## Part 5: MOT17 Evaluation

Evaluate tracking accuracy on the MOT17 benchmark dataset using standard MOT metrics.

### Scripts

| Script | Backend | Description |
|--------|---------|-------------|
| `scripts/evaluate_mot17.py` | MLX GPU | Evaluate YOLO26 MLX models |
| `scripts/evaluate_mot17_pytorch.py` | PyTorch MPS/CPU | Evaluate for comparison |
| `scripts/run_mot17_benchmark.sh` | — | End-to-end benchmark runner |

### Run MLX Evaluation

```bash
# Single model (default: yolo26n)
python scripts/evaluate_mot17.py

# Specific model
python scripts/evaluate_mot17.py --model yolo26s

# All models
python scripts/evaluate_mot17.py --model all

# Single sequence (quick test)
python scripts/evaluate_mot17.py --sequences MOT17-09-SDP

# Use BoT-SORT
python scripts/evaluate_mot17.py --tracker botsort
```

### Run PyTorch Comparison

```bash
# MPS backend
python scripts/evaluate_mot17_pytorch.py --model all --device mps

# CPU backend
python scripts/evaluate_mot17_pytorch.py --model all --device cpu
```

### Run Full Benchmark (Shell Script)

```bash
# Full benchmark (all models, bytetrack)
bash scripts/run_mot17_benchmark.sh

# Quick run (yolo26n, single sequence)
bash scripts/run_mot17_benchmark.sh --quick

# Use BoT-SORT
bash scripts/run_mot17_benchmark.sh --tracker botsort

# Single model
bash scripts/run_mot17_benchmark.sh --model yolo26s
```

### Evaluation Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | `yolo26n` | Model variant (yolo26n/s/m/l/x) or `all` |
| `--data` | str | `datasets/MOT17` | Path to MOT17 dataset |
| `--tracker` | str | `bytetrack` | Tracker type (`bytetrack` or `botsort`) |
| `--imgsz` | int | `1440` | Input image size |
| `--conf` | float | `0.25` | Detection confidence threshold |
| `--iou` | float | `0.7` | NMS IoU threshold |
| `--sequences` | str | `all` | Comma-separated sequence names, or `all` |
| `--eval-class` | int | `1` | MOT class to evaluate (1=pedestrian) |
| `--save-txt` | flag | `True` | Save MOTChallenge result `.txt` files |
| `--output` | str | `results/tracking` | Output directory |
| `--verbose` | flag | — | Verbose logging |
| `--device` | str | `mps` | PyTorch device (PyTorch script only: `mps` or `cpu`) |

### Metrics Computed

| Metric | Description |
|--------|-------------|
| MOTA | Multi-Object Tracking Accuracy (higher is better) |
| IDF1 | ID F1-Score — measures identity preservation (higher is better) |
| MT | Mostly Tracked — tracks covered for ≥80% of lifespan |
| ML | Mostly Lost — tracks covered for ≤20% of lifespan |
| FP | False Positives |
| FN | False Negatives |
| IDSW | Identity Switches |
| Frag | Fragmentations |
| FPS | Frames per second (throughput) |

### MOT17 Evaluation Defaults

| Setting | Value |
|---------|-------|
| Image size | 1440 |
| Confidence threshold | 0.25 |
| IoU threshold | 0.7 |
| Tracker | ByteTrack |
| Dataset | MOT17 train (7 SDP sequences) |
| Eval class | 1 (pedestrian) |

### Output Locations

| Output | Location |
|--------|----------|
| MLX JSON results | `results/tracking/yolo26<variant>_bytetrack_mot17_results.json` |
| PyTorch JSON results | `results/tracking/yolo26<variant>_bytetrack_mot17_pytorch_<device>_results.json` |
| MOTChallenge `.txt` files | `results/tracking/<tracker>/yolo26<variant>/<sequence>.txt` |

---

## Part 6: Tracking Benchmark Charts

After running MOT17 evaluations on all backends (MLX, PyTorch MPS, PyTorch CPU), generate comparison charts.

### Scripts

| Script | Description |
|--------|-------------|
| `scripts/benchmark_tracking_collect_results.py` | Aggregates per-model JSON files into one combined JSON |
| `scripts/benchmark_tracking_generate_charts.py` | Generates 6 comparison charts from the combined JSON |

### Step 1: Collect Results

```bash
# Collect bytetrack results (default)
python scripts/benchmark_tracking_collect_results.py

# Collect botsort results
python scripts/benchmark_tracking_collect_results.py --tracker botsort
```

**Output:** `results/yolo26_tracking_benchmark_combined.json`

### Step 2: Generate Charts

```bash
# Generate PNG charts (default)
python scripts/benchmark_tracking_generate_charts.py

# PDF format for publications
python scripts/benchmark_tracking_generate_charts.py --format pdf

# Custom output directory
python scripts/benchmark_tracking_generate_charts.py --output my_charts/
```

### Charts Generated

| Chart | File | Description |
|-------|------|-------------|
| MOTA comparison | `yolo26_tracking_mota.png` | Grouped bar chart of MOTA across models and backends |
| IDF1 comparison | `yolo26_tracking_idf1.png` | Grouped bar chart of IDF1 across models and backends |
| FPS comparison | `yolo26_tracking_fps.png` | End-to-end throughput (detection + tracking) |
| Speedup | `yolo26_tracking_speedup.png` | MLX vs CPU and MLX vs MPS speedup factors |
| Overhead breakdown | `yolo26_tracking_overhead.png` | Stacked bars: detection vs tracking time (MLX) |
| Summary dashboard | `yolo26_tracking_summary.png` | 2x2 grid with MOTA, IDF1, FPS, and speedup |

### Output Locations

| Output | Location |
|--------|----------|
| Combined results JSON | `results/yolo26_tracking_benchmark_combined.json` |
| Chart images | `results/charts/yolo26_tracking_*.png` |

### Benchmark Results (MOT17-09-SDP, ByteTrack)

| Model | MLX MOTA | MPS MOTA | CPU MOTA | MLX FPS | MPS FPS | CPU FPS | MLX vs CPU |
|-------|----------|----------|----------|---------|---------|---------|------------|
| yolo26n | **46.6** | 45.2 | 45.2 | **37.2** | 34.1 | 8.3 | **4.5×** |
| yolo26s | **46.6** | 44.9 | 44.9 | 21.5 | 22.1 | 4.3 | **5.0×** |
| yolo26m | **45.6** | 38.2 | 38.2 | **10.6** | 10.5 | 2.2 | **4.8×** |
| yolo26l | **48.5** | 42.2 | 42.2 | 8.8 | 8.9 | 1.6 | **5.5×** |
| yolo26x | **38.7** | 35.1 | 35.1 | **4.7** | 3.9 | 1.0 | **4.7×** |

MLX matches or exceeds PyTorch MPS tracking speed. MLX is faster for n, m, and x models; tied with MPS for s and l. Both are **4.5–5.5× faster** than PyTorch CPU. Tracking overhead is ~3–5 ms/frame thanks to batched Kalman updates and batch-precomputed coordinates.

---

## Quick Reference

```bash
# Activate environment
cd yolo-mlx
source .venv/bin/activate

# ── Setup: Download models & convert ──
bash scripts/download_yolo26_models.sh
yolo-mlx converters convert models/yolo26n.pt -o models/yolo26n.npz --verify

# ── Setup: Download MOT17 & create sample video ──
bash scripts/download_mot17.sh
python scripts/create_sample_video.py

# ── Track a video (Python) ──
python -c "
from yolo26mlx import YOLO
model = YOLO('models/yolo26n.npz')
model.track('images/pedestrians.mp4', conf=0.25, save=True)
"

# ── Track a video (CLI) ──
yolo-mlx track --model models/yolo26n.npz --source images/pedestrians.mp4 --save

# ── Track webcam (CLI) ──
yolo-mlx track --model models/yolo26n.npz --source 0 --show

# ── MOT17 evaluation (MLX, single model) ──
python scripts/evaluate_mot17.py --model yolo26n

# ── MOT17 evaluation (all models) ──
python scripts/evaluate_mot17.py --model all

# ── MOT17 evaluation (PyTorch MPS comparison) ──
python scripts/evaluate_mot17_pytorch.py --model all --device mps

# ── Full MOT17 benchmark ──
bash scripts/run_mot17_benchmark.sh

# ── Collect results & generate charts ──
python scripts/benchmark_tracking_collect_results.py
python scripts/benchmark_tracking_generate_charts.py
```
