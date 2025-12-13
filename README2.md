# ObjectLens — Detection + Object-Centric CBIR (Top-K)

This repo implements an MVP pipeline:

1. **User uploads an image (frontend)**
2. **Backend runs YOLO** to detect objects
3. Frontend shows **cropped objects** and user selects one
4. Selected crop is sent to backend **Top-K search**
5. Backend extracts features from the crop and does **cosine similarity** vs a cache of pre-indexed object vectors stored in **MongoDB**
6. Backend returns **Top-K results** + URLs so frontend can display the images

---

## Architecture Overview

### Frontend (React / Vite)

- Upload image
- Call `/api/detect` to get detections + small thumbnails
- Locally crop original image by bbox (`cropToBlob`) and let user select one object
- Send selected crop to:
  - `/api/search/topk` (actual retrieval)
  - optionally `/api/search/select-object` (debug endpoint)

### Backend (FastAPI)

- `/api/detect` → YOLO detection
- `/api/search/topk` → extracts features from crop, similarity search in memory
- `/api/search/reload-cache` → loads object vectors from MongoDB into RAM
- `/files/...` → serves images to the frontend (static mount)

### MongoDB

Stores one document per dataset image:

- `image_path` (string)
- `split` (val/train)
- `width`, `height`
- `objects[]` list, each object includes:
  - bbox, class_id, class_name
  - readable features dict
  - `final_vector` (L2 normalized vector used for retrieval)

---

## Key Data Flow

### 1) Detection

Frontend sends original uploaded image to:

- `POST /api/detect` (multipart form field name = `file`)

Backend returns:

- detections (bbox_xyxy + bbox_xywh, class_id/name, confidence)
- thumbnail data URLs (for UI preview)

### 2) Cropping (frontend)

Frontend crops the original image locally using the bbox:

- `cropToBlob(imageUrl, det.bbox)`  
  This ensures we send **ONLY the selected object crop** to the backend.

### 3) Retrieval (Top-K)

Frontend sends the selected crop to:

- `POST /api/search/topk?top_k=20` (multipart form field name = `file`)

Backend:

- extracts features from the crop
- builds a query vector (final_vector)
- cosine similarity = dot product against cached vectors loaded from MongoDB
- returns:
  - `best_images[]` (unique images with best score)
  - `best_objects[]` (raw object hits, debug)

---

## Important: Indexing Step (Required)

Top-K search depends on having pre-computed vectors in MongoDB.

### Dataset format expected

Under `DATASET_ROOT` (default `/data/imagenet_yolo15`):
