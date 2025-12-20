# ObjectLens - A Content-Based Image Retrieval System

A content-based image retrieval system that uses YOLOv8n object detection and classical feature extraction to find visually similar objects in a pre-indexed ImageNet dataset.

## Overview

ObjectLens enables users to:
1. Upload an image and detect objects using YOLOv8n
2. Select a specific detected object
3. Retrieve the most visually similar objects from a pre-indexed dataset using classical feature descriptors (HSV histogram, Tamura texture, Gabor filters, Fourier descriptors, and orientation histograms)

## Architecture

```
Frontend (React + Vite)
    ↓ HTTP
FastAPI Backend
    ├─ YOLOv8n Object Detection
    ├─ Feature Extraction Service
    └─ Similarity Search (Cosine/Euclidean)
    ↓
MongoDB (Object Metadata & Features)
```

## Technology Stack

- **Frontend**: React 19, Vite
- **Backend**: FastAPI (Python)
- **Database**: MongoDB
- **ML Models**: YOLOv8n (Ultralytics)
- **Feature Extraction**: Classical descriptors (HSV, Tamura, Gabor, Fourier, Orientation)
- **Containerization**: Docker Compose

## Project Structure

```
ObjectLens/
├── api/                    # FastAPI backend
│   ├── app/
│   │   ├── routers/       # API endpoints
│   │   ├── services/      # YOLO, feature extraction, similarity
│   │   └── main.py
│   └── models/yolo/       # YOLO model weights
├── frontend/              # React application
├── db/                    # MongoDB initialization
├── scripts/               # Preprocessing and utility scripts
├── docs/                  # Project documentation
└── docker-compose.yml     # Container orchestration
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Git

### Running with Docker Compose

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ObjectLens
   ```

2. **Start all services**
   ```bash
   docker-compose up --build
   ```

   This will:
   - Start MongoDB on port 27017
   - Run the indexer to populate the database (one-time, if empty)
   - Start the FastAPI backend on port 8000
   - Start the React frontend on port 5173

3. **Access the application**
   - Frontend: http://localhost:5173
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Manual Setup (Development)

#### Backend Setup

```bash
cd api
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables (create .env file)
# MONGO_URI=mongodb://localhost:27017/objectlens
# YOLO_WEIGHTS=./models/yolo/best.pt
# DATASET_ROOT=../imagenet_yolo15

uvicorn app.main:app --reload
```

#### Frontend Setup

```bash
cd frontend
npm install

# Set environment variables (create .env file)
# VITE_API_BASE=http://localhost:8000

npm run dev
```

#### MongoDB Setup

```bash
# Using Docker
docker-compose up mongo

# Or install MongoDB locally and run
mongod
```

## API Endpoints

### Detection

- **POST** `/api/detect`
  - Upload an image to detect objects
  - Returns: List of detected objects with bounding boxes, class names, and confidence scores

### Search

- **POST** `/api/search/topk`
  - Upload an object crop to find similar objects
  - Query parameters:
    - `top_k` (default: 20): Number of results to return
    - `metric` (default: "cosine"): Similarity metric ("cosine" or "euclidean")
    - `same_class_only` (default: true): Filter to same class only
  - Form data:
    - `file`: Image file (object crop)
    - `query_class`: Optional class name for filtering
  - Returns: Top-K similar images with scores and metadata

- **POST** `/api/search/reload-cache`
  - Reload the in-memory feature cache from MongoDB
  - Use after running the indexing script

- **POST** `/api/search/select-object`
  - Debug endpoint for object selection

### Health Checks

- **GET** `/health` - API health status
- **GET** `/health/dataset` - Dataset availability check

### Static Files

- **GET** `/dataset/{path}` - Serve images from the dataset

## Indexing (One-time Setup)

Before using the search functionality, the dataset must be indexed:

1. **Ensure MongoDB is running**
   ```bash
   docker-compose up mongo
   ```

2. **Run the indexer**
   ```bash
   docker-compose up indexer
   ```

   Or manually:
   ```bash
   cd api
   python app/cli/run_indexer.py
   ```

   This will:
   - Detect objects in all dataset images using YOLOv8n
   - Extract features for each detected object
   - Store metadata and features in MongoDB

3. **Reload the cache** (if running manually)
   ```bash
   curl -X POST http://localhost:8000/api/search/reload-cache
   ```

## Dataset

The system expects a YOLO-formatted dataset in `imagenet_yolo15/`:
- `images/train/` - Training images
- `images/val/` - Validation images
- `labels/train/` - YOLO format labels
- `labels/val/` - YOLO format labels

The dataset should contain 15 ImageNet synsets with approximately 1,000 images per class.

## Features

- **Object Detection**: Fine-tuned YOLOv8n on 15 classes
- **Feature Extraction**: Multi-descriptor approach combining:
  - HSV color histogram
  - Dominant colors (K-means)
  - Tamura texture features
  - Gabor filters
  - Fourier descriptors
  - Orientation histogram
- **Similarity Search**: Cosine or Euclidean distance on normalized feature vectors
- **Class Filtering**: Optional filtering to same-class objects only
- **Region-Based CBIR**: Features extracted from cropped object regions (not whole images)

## Development

### Environment Variables

#### Backend (`.env` in `api/`)
```env
MONGO_URI=mongodb://localhost:27017/objectlens
YOLO_WEIGHTS=./models/yolo/best.pt
YOLO_CONF=0.25
YOLO_IOU=0.45
YOLO_IMGSZ=640
DATASET_ROOT=../imagenet_yolo15
DATASET_SPLIT=val
CORS_ORIGINS=http://localhost:5173
TOPK_DEFAULT=20
```

#### Frontend (`.env` in `frontend/`)
```env
VITE_API_BASE=http://localhost:8000
```

## Documentation

- [Project Overview](docs/overview.md)
- [Project Map](docs/project_map.md)
- [Folder Structure](docs/FOLDER_STRUCTURE.md)
- [Logic Flow](docs/logic_flow.md)

## License
Apache License - Version 2.0, January 2004

## Acknowledgments

- YOLOv8 by Ultralytics
- ImageNet dataset