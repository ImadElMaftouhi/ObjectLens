# ObjectLens

**ObjectLens** is a multimedia mining and indexing system that enables efficient object-based image retrieval using deep learning and similarity search. The system supports both 2D image search (using ImageNet/YOLO) and 3D model retrieval (using pottery dataset), combining computer vision, feature extraction, and vector similarity search.

## 🎯 Overview

ObjectLens allows users to:
- **Upload an image** and detect objects within it using YOLO
- **Search for similar objects** across large datasets using FAISS-powered vector similarity
- **Filter results by class** for more precise retrieval
- **Visualize 2D and 3D results** through an interactive web interface
- **Index and retrieve 3D models** based on geometric and visual features

## 🏗️ Architecture

```
ObjectLens/
├── backend/          # FastAPI server with ML pipelines
├── frontend/         # React + Vite web interface
├── scripts/          # Dataset preparation and indexing scripts
├── data/             # Datasets (ImageNet, Pottery)
├── db/               # MongoDB configuration
└── docker-compose.yml
```

### Technology Stack

**Backend:**
- FastAPI for REST API
- YOLOv8 (Ultralytics) for object detection
- FAISS for fast similarity search
- MongoDB for metadata storage
- OpenCV, scikit-image, and custom descriptors for feature extraction

**Frontend:**
- React 19 with Vite
- Three.js for 3D model visualization
- Tailwind CSS for styling
- React Router for navigation

**Databases:**
- MongoDB for storing object metadata and features
- FAISS indices for vector similarity search

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip (Make sure to create a local python env : `python -m venv .venv`)
- **Node.js 16+** with npm
- **Docker & Docker Compose** (optional, for containerized deployment)
- **Git** for cloning the repository

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ObjectLens
   ```

2. **Set up environment variables:**
   ```bash
   # Copy and configure .env file
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run the setup pipeline:**
   ```bash
   # This downloads datasets, builds catalogs, and sets up indices
   bash run_pipeline.sh
   ```

   The pipeline performs:
   - Downloads ImageNet Winter21 dataset
   - Verifies and builds YOLO dataset
   - Precomputes image features
   - Downloads pottery 3D models
   - Builds and splits pottery catalog
   - Sets up MongoDB and FAISS indices

### Running the Application

#### Option 1: Docker Compose (Recommended)

```bash
# Start MongoDB
docker-compose up -d mongo

# Start backend (from project root)
cd backend
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend (in a new terminal)
cd frontend
npm install
npm run dev
```

Access the application at `http://localhost:5173`

#### Option 2: Manual Setup

**Backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ..; uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**MongoDB:**
```bash
# Using Docker
docker run -d -p 27017:27017 --name objectlens-mongo mongo:latest

# Or install MongoDB locally
```

## 📊 Datasets

### 2D Dataset: ImageNet Winter21

- **Purpose:** Object detection and 2D image retrieval
- **Classes:** Multiple object categories
- **Format:** JPEG images with YOLO annotations
- **Location:** `data/imagenet_4_yolo/`

### 3D Dataset: Pottery Models

- **Purpose:** 3D model retrieval and visualization
- **Classes:** Amphora, Hydria, Krater, Kylix, and more
- **Format:** OBJ files with textures
- **Location:** `data/raw/3DModels/`

## 🔍 How It Works

### 2D Image Search Pipeline

1. **Upload & Detection:** User uploads an image → YOLO detects objects → user selects an object
2. **Feature Extraction:** System crops the object → extracts deep features (weighted, L2-normalized vector)
3. **Similarity Search:** FAISS searches the index → returns top-k nearest neighbors
4. **Result Retrieval:** System looks up metadata in MongoDB → returns images with highlighted matching objects

### 3D Model Retrieval Pipeline

1. **Feature Extraction:** Extract geometric descriptors (shape, curvature, distribution)
2. **Indexing:** Build FAISS index from 3D feature vectors
3. **Query:** User searches by uploading a 3D model or selecting from catalog
4. **Visualization:** Results displayed with Three.js 3D viewer

## 🛠️ API Endpoints

### Health Check
```bash
GET /health
GET /health/dataset
```

### Object Detection
```bash
POST /api/detect
# Upload image, returns detected objects with bounding boxes
```

### Search
```bash
POST /api/search/topk?top_k=10&metric=cosine&same_class_only=false
# Parameters:
# - top_k: Number of results to return
# - metric: 'cosine' or 'euclidean'
# - same_class_only: Filter by object class
```

### Sample Data
```bash
GET /api/samples/random?count=10
# Returns random sample images from dataset
```

See [`backend/test_commands.md`](backend/test_commands.md) for detailed API testing examples.

## 📁 Project Structure

### Backend (`/backend`)

```
backend/
├── main.py              # FastAPI application entry point
├── routers/             # API route handlers
│   ├── detect.py        # Object detection endpoints
│   ├── search.py        # Similarity search endpoints
│   └── samples.py       # Sample data endpoints
├── services/            # Business logic
│   ├── yolo_service.py  # YOLO detection service
│   ├── feature_extraction.py  # Feature extraction
│   ├── faiss_service.py # FAISS index management
│   └── compute_similarity.py  # Similarity computation
├── core/                # Configuration and utilities
├── db/                  # Database models and connections
└── schemas.py           # Pydantic models
```

### Frontend (`/frontend`)

```
frontend/
├── src/
│   ├── main.jsx         # Application entry point
│   ├── App.jsx          # Root component with routing
│   ├── Home.jsx         # Main search interface
│   ├── Pr2d.jsx         # 2D preview page
│   ├── Pr3d.jsx         # 3D preview page
│   ├── api.js           # API client
│   └── components/
│       └── ModelViewer.jsx  # 3D model viewer
└── public/              # Static assets
```

See [`frontend/README.md`](frontend/README.md) for detailed frontend documentation.

### Scripts (`/scripts`)

```
scripts/
├── dataset/             # Dataset download and preparation
│   ├── imagenet_*.py    # ImageNet pipeline scripts
│   └── pottery_*.py     # Pottery dataset scripts
├── preprocessing/       # Feature precomputation
└── indexing/            # Index building and evaluation
```

## 🧪 Testing

### Backend Tests

```bash
# Test feature extraction pipeline
python backend/test_feature_pipeline.py

# Test 3D retrieval
python backend/test_3D_retrieval.py

# Test system flow
python backend/test_system_flow.py
```

### API Testing

```bash
# Using curl
curl -X POST "http://localhost:8000/api/search/topk?top_k=10" \
  -F "file=@path/to/image.jpg"

# Using Python
python backend/test_search_endpoint.py
```

## 🎨 Features

### Current Features

- ✅ Object detection with YOLOv8
- ✅ FAISS-powered similarity search
- ✅ Class-based filtering
- ✅ 2D image retrieval
- ✅ 3D model retrieval
- ✅ Interactive web interface
- ✅ MongoDB metadata storage
- ✅ Docker support

### Planned Features

- 🔄 Batch upload and processing
- 🔄 Advanced 3D feature descriptors
- 🔄 Real-time collaborative search
- 🔄 Export and annotation tools

## 🔧 Configuration

Key configuration files:

- **`.env`** - Environment variables (database URLs, API keys, dataset paths)
- **`backend/core/config.py`** - Backend settings
- **`frontend/src/api.js`** - API base URL configuration
- **`docker-compose.yml`** - Container orchestration

## 📚 Documentation

- [Frontend Documentation](frontend/README.md) - Detailed frontend guide
- [Backend Flow](backend/readme.md) - System architecture overview
- [Test Commands](backend/test_commands.md) - API testing examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is part of the M232 Multimedia Mining & Indexing course at IASD.

## 🙏 Acknowledgments

- **ImageNet** for the image dataset
- **Ultralytics** for YOLOv8
- **FAISS** by Meta AI for similarity search
- **Three.js** for 3D visualization

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Built for MST.IASD.232 - multimedia mining and indexing course**
