#!/bin/bash
# Copyright (c) 2026 webAI, Inc.
# Run full MOT17 benchmark — Step 19 of the tracking implementation plan.
#
# Downloads the MOT17 dataset (if needed), evaluates all YOLO26 model variants
# with ByteTrack, and saves combined results.
#
# Usage:
#   bash scripts/run_mot17_benchmark.sh                   # Full benchmark (all models, bytetrack)
#   bash scripts/run_mot17_benchmark.sh --quick            # Quick: yolo26n, single sequence
#   bash scripts/run_mot17_benchmark.sh --tracker botsort  # Use BoT-SORT instead
#   bash scripts/run_mot17_benchmark.sh --model yolo26s    # Single model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/datasets/MOT17"
RESULTS_DIR="${PROJECT_DIR}/results/tracking"

# Defaults
MODEL="all"
TRACKER="bytetrack"
SEQUENCES="all"
IMGSZ=1440
CONF=0.25
QUICK=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --tracker)
            TRACKER="$2"
            shift 2
            ;;
        --sequences)
            SEQUENCES="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --conf)
            CONF="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--model MODEL] [--tracker TRACKER] [--sequences SEQS]"
            exit 1
            ;;
    esac
done

if [ "$QUICK" = true ]; then
    MODEL="yolo26n"
    SEQUENCES="MOT17-02-SDP"
fi

echo "=================================================="
echo "YOLO26 MLX — MOT17 Benchmark Runner"
echo "=================================================="
echo "  Model:      $MODEL"
echo "  Tracker:    $TRACKER"
echo "  Sequences:  $SEQUENCES"
echo "  Image size: $IMGSZ"
echo "  Confidence: $CONF"
echo "  Data dir:   $DATA_DIR"
echo "  Results:    $RESULTS_DIR"
echo ""

# -------------------------------------------------------
# 1. Ensure MOT17 dataset is downloaded
# -------------------------------------------------------
if [ ! -d "$DATA_DIR/train" ] || [ "$(ls -d "$DATA_DIR"/train/MOT17-*-SDP 2>/dev/null | wc -l | tr -d ' ')" -lt 7 ]; then
    echo "[Step 1/3] MOT17 dataset not found — downloading..."
    bash "$SCRIPT_DIR/download_mot17.sh" "$DATA_DIR"
    echo ""
else
    echo "[Step 1/3] MOT17 dataset found at $DATA_DIR"
    echo "           $(ls -d "$DATA_DIR"/train/MOT17-*-SDP | wc -l | tr -d ' ') SDP sequences available"
    echo ""
fi

# -------------------------------------------------------
# 2. Check model weights
# -------------------------------------------------------
echo "[Step 2/3] Checking model weights..."
MODELS_DIR="$PROJECT_DIR/models"
if [ "$MODEL" = "all" ]; then
    VARIANTS="yolo26n yolo26s yolo26m yolo26l yolo26x"
else
    VARIANTS="$MODEL"
fi

MISSING=0
for v in $VARIANTS; do
    if [ -f "$MODELS_DIR/${v}.safetensors" ] || [ -f "$MODELS_DIR/${v}.npz" ] || [ -f "$MODELS_DIR/${v}.pt" ]; then
        echo "  ✓ $v"
    else
        echo "  ✗ $v — weights not found!"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING model(s) missing. Place weights in $MODELS_DIR"
    echo "       or run: bash scripts/download_yolo26_models.sh"
    exit 1
fi
echo ""

# -------------------------------------------------------
# 3. Run evaluation
# -------------------------------------------------------
echo "[Step 3/3] Running MOT17 evaluation..."
echo ""

cd "$PROJECT_DIR/scripts"

python evaluate_mot17.py \
    --model "$MODEL" \
    --data "$DATA_DIR" \
    --tracker "$TRACKER" \
    --imgsz "$IMGSZ" \
    --conf "$CONF" \
    --sequences "$SEQUENCES" \
    --output "$RESULTS_DIR" \
    --save-txt

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=================================================="
    echo "Benchmark complete!"
    echo "=================================================="
    echo ""
    echo "Results saved to:"
    ls -1 "$RESULTS_DIR"/*.json 2>/dev/null | while read -r f; do
        echo "  $(basename "$f")"
    done
    echo ""
    echo "MOTChallenge format predictions:"
    find "$RESULTS_DIR/$TRACKER" -name "*.txt" 2>/dev/null | while read -r f; do
        echo "  ${f#$RESULTS_DIR/}"
    done
else
    echo "ERROR: Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
