#!/bin/bash
# Copyright (c) 2026 webAI, Inc.
# =============================================================================
# YOLO26 Model Download Script
# Downloads YOLO26 pretrained detection models from Ultralytics GitHub releases
#
# Reference: https://docs.ultralytics.com/models/yolo26/
# =============================================================================

set -e  # Exit on error

# Configuration
MODELS=("yolo26n" "yolo26s" "yolo26m" "yolo26l" "yolo26x")
BASE_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0"

# Get script directory and set model directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Banner
echo "=============================================="
echo "  YOLO26 Model Download Script"
echo "  Models: ${MODELS[*]}"
echo "=============================================="
echo ""

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"
echo -e "${GREEN}✓${NC} Model directory: ${MODEL_DIR}"
echo ""

# Parse arguments
DOWNLOAD_ALL=true
SPECIFIC_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            SPECIFIC_MODEL="$2"
            DOWNLOAD_ALL=false
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -m, --model MODEL   Download specific model (n, s, m, l, x)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                  # Download all models"
            echo "  $0 -m n             # Download only yolo26n"
            echo "  $0 --model s        # Download only yolo26s"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to download a model
download_model() {
    local model_name=$1
    local url="${BASE_URL}/${model_name}.pt"
    local output_path="${MODEL_DIR}/${model_name}.pt"

    if [[ -f "$output_path" ]]; then
        echo -e "${YELLOW}⏭${NC}  ${model_name}.pt already exists, skipping..."
        return 0
    fi

    echo -e "📥 Downloading ${model_name}.pt..."

    if curl -L --fail --progress-bar -o "$output_path" "$url"; then
        # Verify file size (should be > 1MB for valid model)
        local file_size=$(stat -f%z "$output_path" 2>/dev/null || stat --printf="%s" "$output_path" 2>/dev/null)
        if [[ $file_size -lt 1000000 ]]; then
            echo -e "${RED}✗${NC}  ${model_name}.pt download failed (file too small)"
            rm -f "$output_path"
            return 1
        fi
        echo -e "${GREEN}✓${NC}  ${model_name}.pt downloaded successfully ($(numfmt --to=iec-i --suffix=B $file_size 2>/dev/null || echo "${file_size} bytes"))"
    else
        echo -e "${RED}✗${NC}  Failed to download ${model_name}.pt"
        rm -f "$output_path"
        return 1
    fi
}

# Download models
SUCCESS_COUNT=0
FAIL_COUNT=0

if [[ "$DOWNLOAD_ALL" == true ]]; then
    echo "Downloading all YOLO26 models..."
    echo ""

    for model in "${MODELS[@]}"; do
        if download_model "$model"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
else
    # Download specific model
    model_name="yolo26${SPECIFIC_MODEL}"

    # Validate model size
    valid=false
    for m in "${MODELS[@]}"; do
        if [[ "$m" == "$model_name" ]]; then
            valid=true
            break
        fi
    done

    if [[ "$valid" == false ]]; then
        echo -e "${RED}Invalid model: ${SPECIFIC_MODEL}${NC}"
        echo "Valid options: n, s, m, l, x"
        exit 1
    fi

    if download_model "$model_name"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
fi

# Summary
echo ""
echo "=============================================="
echo "  Download Summary"
echo "=============================================="
echo -e "${GREEN}✓ Successful:${NC} ${SUCCESS_COUNT}"
if [[ $FAIL_COUNT -gt 0 ]]; then
    echo -e "${RED}✗ Failed:${NC}     ${FAIL_COUNT}"
fi
echo ""

# List downloaded models
echo "Downloaded models in ${MODEL_DIR}/:"
ls -lh "${MODEL_DIR}"/*.pt 2>/dev/null || echo "  (none)"
echo ""

# Exit with error if any downloads failed
if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi

echo -e "${GREEN}✓ All downloads complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Convert weights:           yolo26 converters convert models/yolo26n.pt -o models/yolo26n.npz --verify"
echo "  2. Run inference benchmark:   python scripts/benchmark_yolo26_inference.py"
echo "  3. Run training benchmarks:   python scripts/benchmark_yolo26_training_mlx.py"
