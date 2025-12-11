# Python API - ML Operations

This directory contains the Python API (FastAPI or Flask) that handles all machine learning operations.

## Structure

- `app/main.py` - Application entry point
- `app/routes/` - API endpoints:
  - `detection.py` - POST `/detect` endpoint
  - `search.py` - POST `/search` endpoint
  - `image.py` - GET `/image/{id}` endpoint
- `app/services/` - ML service implementations:
  - `yolo_service.py` - YOLOv8 detection
  - `feature_extraction.py` - EfficientNet feature extraction
  - `faiss_service.py` - Faiss similarity search
  - `image_processing.py` - Image cropping and preprocessing
- `app/models/` - Pydantic models (FastAPI) or schemas
- `models/yolo/` - Trained YOLOv8n model files
- `models/efficientnet/` - EfficientNet model files (if needed)
- `indexes/` - Faiss index files and metadata

## API Endpoints

### POST `/detect`
Upload an image and get detected objects with bounding boxes.

**Request:**
- Multipart form data with image file

**Response:**
```json
{
  "detections": [
    {
      "bbox": [x, y, width, height],
      "class": "person",
      "confidence": 0.95,
      "thumbnail": "base64_encoded_image"
    }
  ]
}
```

### POST `/search`
Search for similar objects using a cropped object image.

**Request:**
- Multipart form data with cropped object image
- Optional: `top_k` parameter (default: 20)

**Response:**
```json
{
  "results": [
    {
      "image_id": "123",
      "image_path": "path/to/image.jpg",
      "bbox": [x, y, width, height],
      "similarity": 0.92,
      "class": "person"
    }
  ]
}
```

### GET `/image/{id}`
Get image metadata and serve image.

**Response:**
- Image file or metadata JSON

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file from `.env.example`

4. Run the API:
```bash
# FastAPI
uvicorn app.main:app --reload

# Flask
python app/main.py
```

## Dependencies

- FastAPI or Flask
- Ultralytics (YOLOv8)
- timm (EfficientNet)
- faiss-cpu or faiss-gpu
- Pillow
- numpy

