#!/bin/bash
# Copyright (c) 2026 webAI, Inc.
# Download COCO val2017 dataset for YOLO26 MLX evaluation

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${1:-$SCRIPT_DIR/../datasets/coco}"
BASE_URL="http://images.cocodataset.org"

echo "=================================================="
echo "COCO val2017 Dataset Download Script"
echo "=================================================="
echo "Target directory: $DATASET_DIR"
echo ""

# Create directories
mkdir -p "$DATASET_DIR/images"
mkdir -p "$DATASET_DIR/annotations"
mkdir -p "$DATASET_DIR/labels/val2017"

# Resolve to absolute path before cd (so later references still work)
DATASET_DIR=$(cd "$DATASET_DIR" && pwd)
cd "$DATASET_DIR"

# Normalize labels layout if extracted under nested "coco/labels/val2017".
normalize_labels_layout() {
    local nested_dir="coco/labels/val2017"
    local target_dir="labels/val2017"
    if [ -d "$nested_dir" ]; then
        # Move only when target is missing or empty.
        if [ ! -d "$target_dir" ] || [ -z "$(ls -A "$target_dir" 2>/dev/null)" ]; then
            mkdir -p "$target_dir"
            mv "$nested_dir"/* "$target_dir"/
        fi
    fi
}

# Download validation images (1GB, 5000 images)
if [ -d "images/val2017" ] && [ "$(ls -A images/val2017 2>/dev/null)" ]; then
    echo "[INFO] val2017 images already exist, skipping download"
else
    echo "[1/3] Downloading val2017 images (1GB)..."
    curl -L -o val2017.zip "$BASE_URL/zips/val2017.zip"
    echo "      Extracting images..."
    unzip -q val2017.zip -d images/
    rm val2017.zip
    echo "      Done! $(ls images/val2017 | wc -l) images"
fi

# Download annotations (241MB)
if [ -f "annotations/instances_val2017.json" ]; then
    echo "[INFO] Annotations already exist, skipping download"
else
    echo "[2/3] Downloading annotations (241MB)..."
    curl -L -o annotations.zip "$BASE_URL/annotations/annotations_trainval2017.zip"
    echo "      Extracting annotations..."
    unzip -q -o annotations.zip
    rm annotations.zip
    echo "      Done!"
fi

# Download labels in YOLO format (from Ultralytics)
LABELS_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip"
normalize_labels_layout
if [ -f "labels/val2017/000000000139.txt" ]; then
    echo "[INFO] YOLO-format labels already exist, skipping download"
else
    echo "[3/3] Downloading YOLO-format labels..."
    curl -L -o labels.zip "$LABELS_URL"
    echo "      Extracting labels..."
    unzip -q -o labels.zip
    rm labels.zip
    normalize_labels_layout
    echo "      Done!"
fi

# Create val2017.txt file list
echo ""
echo "Creating image list file..."
ls "$DATASET_DIR/images/val2017/"*.jpg > val2017.txt
echo "Created val2017.txt with $(wc -l < val2017.txt) entries"

# Verify download
echo ""
echo "=================================================="
echo "Download Complete!"
echo "=================================================="
echo ""
echo "Dataset structure:"
echo "$DATASET_DIR/"
ls -la "$DATASET_DIR"

echo ""
echo "Summary:"
echo "  - Images: $(ls images/val2017 2>/dev/null | wc -l) files"
echo "  - Annotations: instances_val2017.json"
echo "  - Labels: YOLO format in labels/val2017/"
echo ""
echo "To use with evaluation script:"
echo "  python scripts/evaluate_coco_val.py --model yolo26n --data $DATASET_DIR"
