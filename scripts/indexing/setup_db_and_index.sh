#!/bin/bash
#
# Setup MongoDB and load FAISS metadata
#
# This script:
# 1. Starts MongoDB container
# 2. Waits for MongoDB to be ready
# 3. Loads FAISS metadata into MongoDB
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  MongoDB Setup for ObjectLens"
echo "=========================================="
echo ""

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Step 1: Start MongoDB
echo -e "${YELLOW}[1/3]${NC} Starting MongoDB container..."
docker-compose up -d mongo

# Wait for MongoDB to be ready
echo -e "${YELLOW}[2/3]${NC} Waiting for MongoDB to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if docker exec objectlens-mongo mongosh --quiet --eval "db.runCommand({ ping: 1 }).ok" | grep -q "1"; then
        echo -e "${GREEN}✅ MongoDB is ready!${NC}"
        break
    fi
    attempt=$((attempt + 1))
    echo "   Waiting... (${attempt}/${max_attempts})"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}❌ MongoDB failed to start${NC}"
    exit 1
fi

# Step 2: Check if FAISS index exists
FAISS_METADATA="$PROJECT_ROOT/data/index/faiss/metadata.json"
OBJECT_MAPPING="$PROJECT_ROOT/data/index/metadata/object_mapping.json"

if [ ! -f "$FAISS_METADATA" ] || [ ! -f "$OBJECT_MAPPING" ]; then
    echo -e "${RED}❌ FAISS index not found!${NC}"
    echo "   Please run: python scripts/preprocessing/imagenet_04_precompute_features.py"
    exit 1
fi

echo -e "${GREEN}✅ FAISS index found${NC}"

# Step 3: Load data into MongoDB
echo ""
echo -e "${YELLOW}[3/3]${NC} Loading FAISS metadata into MongoDB..."
echo ""

python scripts/indexing/load_to_mongodb.py \
    --mapping-path "$OBJECT_MAPPING" \
    --faiss-metadata-path "$FAISS_METADATA" \
    --mongo-uri "mongodb://localhost:27017/objectlens"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "  ✅ Setup Complete!"
    echo "==========================================${NC}"
    echo ""
    echo "MongoDB is ready with:"
    echo "  - Image metadata"
    echo "  - Object metadata"
    echo "  - FAISS index references"
    echo ""
    echo "Next steps:"
    echo "  1. Test backend API: python -m uvicorn backend.main:app --reload"
    echo "  2. Test search endpoints"
    echo "  3. Integrate with frontend"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Failed to load data${NC}"
    exit 1
fi
