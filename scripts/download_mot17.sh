#!/bin/bash
# Copyright (c) 2026 webAI, Inc.
# Download MOT17 dataset for YOLO26 MLX tracking evaluation
#
# Usage:
#   bash scripts/download_mot17.sh [target_dir]
#
# Downloads the MOT17 training set (~5.5 GB) which contains
# 7 sequences × 3 detector variants = 21 folders with public
# ground truth annotations for local evaluation.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${1:-$SCRIPT_DIR/../datasets/MOT17}"
MOT17_URL="https://motchallenge.net/data/MOT17.zip"

echo "=================================================="
echo "MOT17 Dataset Download Script"
echo "=================================================="
echo "Target directory: $DATASET_DIR"
echo ""

# Create target directory
mkdir -p "$DATASET_DIR"
DATASET_DIR=$(cd "$DATASET_DIR" && pwd)
cd "$DATASET_DIR"

# Check if already downloaded
if [ -d "train" ] && [ "$(ls -d train/MOT17-*-SDP 2>/dev/null | wc -l)" -ge 7 ]; then
    echo "[INFO] MOT17 training set already exists ($(ls -d train/MOT17-* | wc -l) folders), skipping download."
    echo ""
    echo "Sequences found:"
    ls -d train/MOT17-*-SDP 2>/dev/null | while read -r seq; do
        name=$(basename "$seq")
        n_frames=$(ls "$seq/img1/" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $name  ($n_frames frames)"
    done
    echo ""
    echo "Done — dataset is ready at: $DATASET_DIR"
    exit 0
fi

echo "[1/2] Downloading MOT17 (~5.5 GB)..."
echo "      URL: $MOT17_URL"
echo "      This may take a while depending on your connection."
echo ""

curl -L -o MOT17.zip "$MOT17_URL"

echo ""
echo "[2/2] Extracting..."

unzip -q MOT17.zip

# The zip extracts to MOT17/ subdirectory — move contents up if needed
if [ -d "MOT17/train" ] && [ ! -d "train" ]; then
    mv MOT17/train train
fi
if [ -d "MOT17/test" ] && [ ! -d "test" ]; then
    mv MOT17/test test
fi
# Clean up nested directory if it's now empty
if [ -d "MOT17" ]; then
    rmdir MOT17 2>/dev/null || true
fi

rm -f MOT17.zip

echo ""
echo "=================================================="
echo "MOT17 Download Complete"
echo "=================================================="
echo ""
echo "Dataset location: $DATASET_DIR"
echo ""

# Show summary
if [ -d "train" ]; then
    echo "Training sequences (with ground truth):"
    ls -d train/MOT17-*-SDP 2>/dev/null | while read -r seq; do
        name=$(basename "$seq")
        n_frames=$(ls "$seq/img1/" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $name  ($n_frames frames)"
    done
    echo ""
    echo "Total training folders: $(ls -d train/MOT17-* | wc -l | tr -d ' ')"
fi

if [ -d "test" ]; then
    echo "Test sequences: $(ls -d test/MOT17-* | wc -l | tr -d ' ') folders (no ground truth)"
fi

echo ""
echo "Note: For evaluation, use only -SDP variants (7 unique sequences)"
echo "      since images and ground truth are identical across detector variants."
