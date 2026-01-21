#!/bin/bash
#
# Run ObjectLens setup pipeline
#

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "  ObjectLens Setup Pipeline"
echo "=========================================="

# Dataset scripts
echo -e "${YELLOW}[1/8]${NC} Downloading ImageNet Winter21..."
python scripts/dataset/imagenet_01_download_winter21.py

echo -e "${YELLOW}[2/8]${NC} Verifying download..."
python scripts/dataset/imagenet_02_verify_download.py

echo -e "${YELLOW}[3/8]${NC} Building YOLO dataset..."
python scripts/dataset/imagenet_03_build_yolo_dataset.py

# Preprocessing
echo -e "${YELLOW}[4/8]${NC} Precomputing features..."
python scripts/preprocessing/imagenet_04_precompute_features.py

# Pottery scripts
echo -e "${YELLOW}[5/8]${NC} Downloading pottery dataset..."
python scripts/dataset/pottery_01_download.py

echo -e "${YELLOW}[6/8]${NC} Building pottery catalog..."
python scripts/dataset/pottery_02_build_catalog.py

echo -e "${YELLOW}[7/8]${NC} Splitting pottery catalog..."
python scripts/dataset/pottery_03_split_catalog.py

# echo -e "${YELLOW}[7/8]${NC} Indexing pottery catalog..."
# python scripts/indexing/pottery_04_index_catalog.py

# echo -e "${YELLOW}[7/8]${NC} Evaluate retrieval..."
# python scripts/indexing/pottery_05_evaluate_retrieval.py

# Database setup
echo -e "${YELLOW}[8/8]${NC} Setting up database and index..."
bash scripts/indexing/setup_db_and_index.sh

echo -e "${GREEN}✅ Pipeline complete!${NC}"