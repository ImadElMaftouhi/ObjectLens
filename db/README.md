# MongoDB Setup and Data Loading

This guide explains how to set up MongoDB and load FAISS metadata into it.

## Overview

MongoDB stores **metadata only** - not the actual images:
- **Images**: Stored on disk/mounted volumes (not in MongoDB)
- **Metadata**: Image paths, object bounding boxes, class info, FAISS IDs
- **FAISS Index**: Stored on disk (vectors.npy, index.faiss)

What MongoDB stores
- **Object metadata**: bounding boxes, class IDs, class names, confidence
- **FAISS references**: faiss_id linking to FAISS index

## Quick Start

### 1. Start MongoDB Container

```bash
# Start MongoDB only
docker-compose up -d mongo

# Check if it's running
docker-compose ps

# View logs
docker-compose logs mongo
```

### 2. Build FAISS Index (if not done already)

```bash
# This should already be done, but if not:
python scripts/preprocessing/imagenet_04_precompute_features.py
```

This creates:
- `data/index/faiss/` - FAISS index files
- `data/index/metadata/object_mapping.json` - Object metadata

### 3. Load Metadata into MongoDB

```bash
# Load from default paths
python scripts/indexing/load_to_mongodb.py

# Or specify custom paths
python scripts/indexing/load_to_mongodb.py \
    --mapping-path data/index/metadata/object_mapping.json \
    --faiss-metadata-path data/index/faiss/metadata.json

# Drop existing data and reload
python scripts/indexing/load_to_mongodb.py --drop-existing

# Skip individual objects collection (faster, images collection only)
python scripts/indexing/load_to_mongodb.py --skip-objects-collection
```

### 4. Verify Data in MongoDB

```bash
# Connect to MongoDB
docker exec -it objectlens-mongo mongosh objectlens

# Check collections
show collections

# Count documents
db.images.countDocuments()
db.objects.countDocuments()

# View sample document
db.images.findOne()
db.objects.findOne()

# View index metadata
db.index_metadata.findOne()
```

## MongoDB Schema

### `images` Collection

Groups objects by image path:

```javascript
{
  "_id": "train/n00007846_103856.JPEG",  // image_path as _id
  "image_path": "train/n00007846_103856.JPEG",
  "split": "train",
  "num_objects": 2,
  "objects": [
    {
      "object_idx": 0,
      "faiss_id": 123,  // Reference to FAISS index
      "bbox": [x1, y1, x2, y2],
      "class_id": 10,
      "class_name": "person",
      "confidence": 0.95
    },
    // ... more objects
  ],
  "indexed_at": ISODate("2026-01-19T...")
}
```

### `objects` Collection (Optional)

Individual objects for direct queries:

```javascript
{
  "_id": "train/n00007846_103856.JPEG__0",  // image_path__object_idx
  "faiss_id": 123,  // Reference to FAISS index
  "image_path": "train/n00007846_103856.JPEG",
  "object_idx": 0,
  "bbox": [x1, y1, x2, y2],
  "class_id": 10,
  "class_name": "person",
  "confidence": 0.95
}
```

### `index_metadata` Collection

FAISS index information:

```javascript
{
  "_id": "faiss_index",
  "num_vectors": 7713,
  "dimension": 176,
  "metric": "cosine",
  "index_type": "IndexFlatIP",
  "updated_at": ISODate("2026-01-19T..."),
  "vectors_file": "vectors.npy",
  "ids_file": "ids.npy",
  "index_file": "index.faiss"
}
```

## Indexes Created

### `images` Collection
- `image_path` (unique)
- `split`
- `objects.class_id`
- `objects.class_name`
- `objects.faiss_id`

### `objects` Collection
- `faiss_id` (unique)
- `image_path`
- `object_idx`
- `class_id`
- `class_name`

## Local Development (Without Docker)

If you want to run MongoDB locally:

```bash
# Install MongoDB locally (macOS)
brew install mongodb-community

# Start MongoDB
brew services start mongodb-community

# Update MONGO_URI in .env
MONGO_URI=mongodb://localhost:27017/objectlens

# Load data
python scripts/indexing/load_to_mongodb.py
```

## Troubleshooting

### MongoDB Connection Failed

```bash
# Check if MongoDB is running
docker-compose ps mongo

# Check MongoDB logs
docker-compose logs mongo

# Test connection
docker exec -it objectlens-mongo mongosh objectlens --eval "db.runCommand({ ping: 1 })"
```

### Data Not Loading

1. Verify FAISS index exists:
   ```bash
   ls -la data/index/faiss/
   ls -la data/index/metadata/
   ```

2. Check MongoDB connection:
   ```python
   from backend.db.mongo import get_collection
   col = get_collection("images")
   print(col.count_documents({}))
   ```

3. View loading script output:
   ```bash
   python scripts/indexing/load_to_mongodb.py --drop-existing
   ```

### Drop All Data

```bash
# Connect to MongoDB
docker exec -it objectlens-mongo mongosh objectlens

# Drop collections
db.images.drop()
db.objects.drop()
db.index_metadata.drop()
```

## Next Steps

After loading data:
1. ✅ MongoDB contains metadata
2. ✅ FAISS index is on disk
3. ⏭️  Update backend to use FAISS for search
4. ⏭️  Test search endpoints
5. ⏭️  Integrate with frontend
