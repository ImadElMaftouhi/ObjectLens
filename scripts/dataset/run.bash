#!/bin/bash
#
# Complete dataset download and setup script for ObjectLens project
# Downloads and processes both ImageNet and Pottery datasets
#
# Usage:
#   bash scripts/dataset/run.bash [options]
#
# Options:
#   --skip-imagenet    Skip ImageNet dataset steps
#   --skip-pottery     Skip Pottery dataset steps
#   --skip-precompute  Skip feature precomputation
#   --skip-eval        Skip retrieval evaluation
#   --help             Show this help message
#

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse command line arguments
SKIP_IMAGENET=false
SKIP_POTTERY=false
SKIP_PRECOMPUTE=false
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-imagenet)
            SKIP_IMAGENET=true
            shift
            ;;
        --skip-pottery)
            SKIP_POTTERY=true
            shift
            ;;
        --skip-precompute)
            SKIP_PRECOMPUTE=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-imagenet    Skip ImageNet dataset steps"
            echo "  --skip-pottery     Skip Pottery dataset steps"
            echo "  --skip-precompute  Skip feature precomputation"
            echo "  --skip-eval        Skip retrieval evaluation"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to print colored messages
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run a command with error handling
run_step() {
    local step_name="$1"
    local command="$2"
    
    info "Starting: $step_name"
    if eval "$command"; then
        success "Completed: $step_name"
        return 0
    else
        error "Failed: $step_name"
        return 1
    fi
}

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    error "Virtual environment not found at $PROJECT_ROOT/.venv"
    error "Please create it first: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
info "Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

# Verify Python is available
if ! command -v python &> /dev/null; then
    error "Python not found in virtual environment"
    exit 1
fi

info "Using Python: $(which python)"
info "Python version: $(python --version)"

# Change to project root
cd "$PROJECT_ROOT"

echo ""
echo "=========================================="
echo "  ObjectLens Dataset Setup"
echo "=========================================="
echo ""

# ============================================
# ImageNet Dataset Pipeline
# ============================================
if [ "$SKIP_IMAGENET" = false ]; then
    echo ""
    info "=== ImageNet Dataset Pipeline ==="
    echo ""
    
    # Step 1: Download ImageNet Winter 2021
    run_step "Download ImageNet Winter 2021" \
        "python scripts/dataset/imagenet_01_download_winter21.py" || exit 1
    
    # Step 2: Verify downloads
    run_step "Verify ImageNet downloads" \
        "python scripts/dataset/imagenet_02_verify_download.py" || exit 1
    
    # Step 3: Build YOLO dataset
    run_step "Build YOLO dataset" \
        "python scripts/dataset/imagenet_03_build_yolo_dataset.py" || exit 1
    
    # Step 4: Precompute features (optional)
    if [ "$SKIP_PRECOMPUTE" = false ]; then
        run_step "Precompute ImageNet features" \
            "python scripts/preprocessing/imagenet_04_precompute_features.py" || exit 1
    else
        warning "Skipping feature precomputation (--skip-precompute)"
    fi
    
    success "ImageNet dataset pipeline completed!"
else
    warning "Skipping ImageNet dataset pipeline (--skip-imagenet)"
fi

# ============================================
# Pottery Dataset Pipeline
# ============================================
if [ "$SKIP_POTTERY" = false ]; then
    echo ""
    info "=== Pottery 3D Dataset Pipeline ==="
    echo ""
    
    # Step 1: Download and extract pottery dataset
    run_step "Download Pottery dataset" \
        "python scripts/dataset/pottery_01_download.py" || exit 1
    
    # Step 2: Build catalog
    run_step "Build Pottery catalog" \
        "python scripts/dataset/pottery_02_build_catalog.py" || exit 1
    
    # Step 3: Split catalog
    run_step "Split Pottery catalog" \
        "python scripts/dataset/pottery_03_split_catalog.py" || exit 1
    
    # Step 4: Initialize MongoDB
    run_step "Initialize MongoDB for Pottery" \
        "python scripts/dataset/pottery_04_init_mongodb.py" || exit 1
    
    # Step 5: Index models
    run_step "Index Pottery models" \
        "python scripts/dataset/pottery_05_index_models.py --image-size 256 --l2-normalize" || exit 1
    
    # Step 6: Run retrieval evaluation (optional)
    if [ "$SKIP_EVAL" = false ]; then
        run_step "Run Pottery retrieval evaluation" \
            "python scripts/dataset/pottery_06_evaluate_retrieval.py --both --image-size 256 --depth-rotation-set grid24 --l2-normalize" || exit 1
    else
        warning "Skipping retrieval evaluation (--skip-eval)"
    fi
    
    success "Pottery dataset pipeline completed!"
else
    warning "Skipping Pottery dataset pipeline (--skip-pottery)"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
success "All dataset setup completed successfully!"
echo "=========================================="
# echo ""
# info "Next steps:"
# info "  1. Start MongoDB: docker-compose up -d mongo"
# info "  2. Start backend:  docker-compose up backend"
# info "  3. Start frontend: docker-compose up frontend"
# echo ""
