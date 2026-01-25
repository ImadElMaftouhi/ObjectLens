# Quick Start: MongoDB Setup

## One-Command Setup (Bash/Linux)

```bash
bash scripts/indexing/setup_mongodb.sh
```

## Manual Setup (Windows/PowerShell)

### Step 1: Start MongoDB

```powershell
docker-compose up -d mongo
```

Wait ~10 seconds for MongoDB to initialize.

### Step 2: Verify MongoDB is Ready

```powershell
docker exec objectlens-mongo mongosh objectlens --eval "db.runCommand({ ping: 1 })"
```

Should return `{ ok: 1 }`

### Step 3: Load FAISS Metadata

**For local development** (running script from host, MongoDB in Docker):

```powershell
# Use localhost to connect to MongoDB container
python scripts/indexing/load_to_mongodb.py --mongo-uri mongodb://localhost:27017/objectlens
```

**Or set environment variable:**

```powershell
$env:MONGO_URI="mongodb://localhost:27017/objectlens"
python scripts/indexing/load_to_mongodb.py
```

**For Docker environment** (script running in container):

```powershell
python scripts/indexing/load_to_mongodb.py
```

This will:
- ✅ Load object metadata from `data/index/metadata/object_mapping.json`
- ✅ Load FAISS index info from `data/index/faiss/metadata.json`
- ✅ Create `images` collection (grouped by image)
- ✅ Create `objects` collection (individual objects)
- ✅ Create `index_metadata` collection (FAISS info)

### Step 4: Verify Data

```powershell
# Connect to MongoDB
docker exec -it objectlens-mongo mongosh objectlens

# Check counts
db.images.countDocuments()
db.objects.countDocuments()

# View sample
db.images.findOne()
```

## What Gets Stored in MongoDB?

✅ **Stored in MongoDB:**
- Image paths
- Object bounding boxes
- Class IDs and names
- Confidence scores
- FAISS IDs (references to FAISS index)
- Metadata (split, dimensions, etc.)

❌ **NOT Stored in MongoDB:**
- Actual images (stored on disk/volumes)
- Feature vectors (stored in FAISS index on disk)

## File Structure

```
data/
├── index/
│   ├── faiss/
│   │   ├── vectors.npy        # Feature vectors (disk)
│   │   ├── index.faiss        # FAISS index (disk)
│   │   ├── ids.npy            # ID mapping (disk)
│   │   └── metadata.json      # Index metadata
│   └── metadata/
│       └── object_mapping.json # Object metadata
└── imagenet_yolo15/           # Images (disk/volume)
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
```

## MongoDB Collections

1. **`images`** - Groups objects by image path
2. **`objects`** - Individual objects (optional, for direct queries)
3. **`index_metadata`** - FAISS index information

## Next Steps

After MongoDB is set up:
1. ✅ Test backend connection to MongoDB
2. ✅ Update search router to use FAISS
3. ✅ Test search endpoints
4. ✅ Integrate with frontend
