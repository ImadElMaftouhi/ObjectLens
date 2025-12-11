# Content-Based Image Retrieval System

A web application for content-based image retrieval using YOLOv8n object detection and EfficientNet feature extraction.

## Project Structure

```
mini-projet-1/
├── frontend/          # React frontend application
├── backend/           # Main backend server (Node.js/Express, Laravel, etc.)
├── python-api/        # Python ML API (FastAPI/Flask)
├── scripts/           # Preprocessing and utility scripts
├── dataset/           # ImageNet dataset (15 synsets)
├── imagenet_yolo15/   # YOLO formatted dataset
└── docs/              # Project documentation
```

## Quick Start

### 1. Frontend Setup (React)

```bash
cd frontend
npm install
# Create .env file with:
# REACT_APP_API_URL=http://localhost:3001/api
npm start
```

### 2. Backend Setup

Choose your backend technology and follow the setup instructions in `backend/README.md`.

**Node.js/Express:**
```bash
cd backend
npm install
# Create .env file
npm start
```

### 3. Python API Setup

```bash
cd python-api
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Create .env file
uvicorn app.main:app --reload
```

### 4. Preprocessing (One-time setup)

```bash
# Extract features for all objects
python scripts/preprocessing/extract_features.py

# # Build Faiss index
# python scripts/preprocessing/build_faiss_index.py

# Generate thumbnails
python scripts/preprocessing/generate_thumbnails.py

# Populate database
python scripts/database/populate_db.py
```

## Architecture

```
Frontend (React)
    ↓ HTTP
Backend
    ↓ HTTP
Python API (FastAPI/Flask)
    ├─ YOLOv8 Detection
    └─ Feature Extraction
```

## API Endpoints

### Python API

- `POST /detect` - Upload image, get detected objects
- `POST /search` - Search for similar objects
- `GET /image/{id}` - Get image metadata

### Backend API

- `POST /api/upload` - Upload image
- `POST /api/detect` - Trigger detection (proxies to Python API)
- `POST /api/search` - Trigger search (proxies to Python API)
- `GET /api/images/:id` - Get image

## Environment Variables

See `.env.example` files in each directory:
- `frontend/.env.example`
- `backend/.env.example`
- `python-api/.env.example`

## Development Workflow

1. **Frontend**: React app runs on `http://localhost:3000`
2. **Backend**: Main server runs on `http://localhost:3001`
3. **Python API**: ML API runs on `http://localhost:8000`

## Technology Stack

- **Frontend**: React, React Router, Axios
- **Backend**: Node.js/Express, Laravel, or Django (your choice)
- **Python API**: FastAPI or Flask
- **ML**: YOLOv8n, EfficientNet-B0, Faiss
- **Database**: PostgreSQL or SQLite

## Documentation

- [Project Overview](docs/project.md)
- [Project Map](docs/project_map.md)
- [Folder Structure](FOLDER_STRUCTURE.md)

## Next Steps

1. Choose your backend technology
2. Set up each component following the README in each directory
3. Run preprocessing scripts to build indexes
4. Start all three services
5. Test the complete workflow

## License

See LICENSE file for details.

