# Project Folder Structure

```
ObjectLens/
│
├── frontend/                          # React + Vite Frontend (to be created)
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/                # Reusable React components
│   │   │   ├── common/
│   │   │   │   ├── Header.jsx
│   │   │   │   ├── Footer.jsx
│   │   │   │   ├── LoadingSpinner.jsx
│   │   │   │   └── ErrorMessage.jsx
│   │   │   ├── upload/
│   │   │   │   ├── ImageUpload.jsx
│   │   │   │   └── DragDropZone.jsx
│   │   │   ├── detection/
│   │   │   │   ├── DetectionResults.jsx
│   │   │   │   ├── BoundingBox.jsx
│   │   │   │   ├── ObjectThumbnail.jsx
│   │   │   │   └── ObjectSelector.jsx
│   │   │   └── results/
│   │   │       ├── ResultsGallery.jsx
│   │   │       ├── ResultCard.jsx
│   │   │       └── HighlightedObject.jsx
│   │   ├── pages/
│   │   │   ├── HomePage.jsx
│   │   │   ├── UploadPage.jsx
│   │   │   ├── ResultsPage.jsx
│   │   │   └── NotFoundPage.jsx
│   │   ├── services/
│   │   │   ├── api.js                 # API client (calls http://localhost:8000/api/...)
│   │   │   ├── detectionService.js
│   │   │   ├── searchService.js
│   │   │   └── imageService.js
│   │   ├── hooks/
│   │   │   ├── useImageUpload.js
│   │   │   ├── useDetection.js
│   │   │   └── useSearch.js
│   │   ├── utils/
│   │   │   ├── constants.js
│   │   │   ├── helpers.js
│   │   │   └── validators.js
│   │   ├── styles/
│   │   │   ├── index.css
│   │   │   └── variables.css
│   │   ├── App.jsx
│   │   ├── index.jsx
│   │   └── routes.jsx
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   ├── .env
│   ├── .env.example
│   ├── .gitignore
│   └── README.md
│
├── api/                        # FastAPI Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI entry point
│   │   ├── config.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── detection.py           # POST /api/detect
│   │   │   ├── search.py              # POST /api/search
│   │   │   └── image.py               # GET  /api/image/{path}
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── yolo_service.py        # YOLO detection
│   │   │   ├── feature_service.py     # Classical features (HSV, Tamura, etc.)
│   │   │   ├── faiss_service.py       # Faiss similarity search
│   │   │   └── image_processing.py    # Image utilities
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── image_utils.py
│   │   └── middleware/
│   │       ├── __init__.py
│   │       └── cors.py
│   ├── models/
│   │   ├── yolo/                      # YOLO models
│   │   │   ├── weights/
│   │   │   │   ├── best.pt
│   │   │   │   └── last.pt
│   │   │   └── results.csv
│   │   └── efficientnet/              # EfficientNet (if needed)
│   ├── indexes/                       # Faiss indexes
│   │   ├── faiss_index.index
│   │   └── metadata.json
│   ├── requirements.txt
│   ├── .env
│   ├── .env.example
│   ├── .gitignore
│   ├── README.md
│   └── yolo/                          # YOLO training artifacts
│       ├── model/
│       │   ├── args.yaml
│       │   ├── results.csv
│       │   ├── weights/
│       │   │   ├── best.pt
│       │   │   └── last.pt
│       │   └── kaggle/
│       ├── notebooks/
│       │   └── train_yolo.ipynb
│       └── scripts/
│           ├── download_synsets.py
│           ├── prepare_yolo_imagenet15.py
│           ├── test_yolo.py
│           └── verify_downloads.py
│
├── features/                          # Pre-computed classical features (to be generated)
│   └── all/
│       ├── WNIDS1.json
│       └── WNIDS2.json
│       └── ...
│
├── scripts/                           # Preprocessing & utility scripts
│   └── dataset/                        # Everything that touches the raw dataset
│   │   ├── download_synsets.py
│   │   ├── verify_downloads.py
│   │   ├── build_yolo_dataset.py     # ex prepare_yolo_imagenet15.py
│   │   └── verification_report.json    # output of verify_downloads.py
│   ├── preprocessing/
│   │   ├── compute_similarity.py      # Compute feature similarity
│   │   ├── feature_extraction.py      # Extract features from objects
│   │   ├── precompute_features.py     # Pre-compute all features
│   │   └── __pycache__/

│   └── README.md
│
├── docs/                              # Documentation
│   ├── documentation.md
│   ├── ennoncer.md
│   ├── FOLDER_STRUCTURE.md
│   ├── logic_flow.md
│   ├── overview.md
│   ├── project_map.md
│   └── README.md
│
├── dataset/                           # ImageNet dataset (raw images + XML bboxes)
│   └── [synset folders]/
│
├── imagenet_yolo15/                   # YOLO-formatted dataset
│   ├── images/
│   └── labels/
│
├── .gitignore
├── README.md
├── notes.md
└── yolo.zip, yolov8n_imagenet152.zip  # Archived models
```

## Key Directories Explanation

### Frontend (`frontend/`)
- **React application** with Vite build tool
- Component-based architecture (common, upload, detection, results)
- Service layer for API calls to FastAPI backend
- Custom React hooks for reusable logic
- Environment: `VITE_API_URL=http://localhost:8000/api`

### Python API (`api/`)
- **FastAPI application** serving all ML operations
- Routes:
  - `POST /api/detect` - YOLO object detection
  - `POST /api/search` - Similarity-based search
  - `GET /api/image/{path}` - Serve images
- Contains YOLO models and feature extraction services
- Faiss indexes for fast similarity search

### Scripts (`scripts/`)
- Preprocessing scripts for one-time operations
- Feature extraction, index building
- Independent utility functions

### Dataset & Models
- `dataset/` - Raw ImageNet images with XML annotations
- `imagenet_yolo15/` - YOLO-formatted training dataset
- `api/models/` - Trained YOLO and feature extraction models
- `api/indexes/` - Faiss indexes for similarity search

## Environment Variables

### Frontend `.env`
```
VITE_API_URL=http://localhost:8000/api
```

### Python API `.env`
```
FASTAPI_PORT=8000
FASTAPI_HOST=0.0.0.0
MODEL_PATH=./models/yolo/best.pt
FAISS_INDEX_PATH=./indexes/faiss_index.index
METADATA_PATH=./indexes/metadata.json
DATASET_PATH=../dataset
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
```

