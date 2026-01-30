from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.core.config import settings
from backend.routers.detect import router as detect_router
from backend.routers.search import router as search_router
from backend.routers.samples import router as samples_router

import logging
logger = logging.getLogger(__name__)

app = FastAPI(title="ObjectLens API")

origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve 2D dataset images (ImageNet)
# Example URL: /dataset/images/val/xxx.jpg
dataset_dir = settings.get_imagenet_dataset_path()
logger.info(f"Mounting 2D dataset directory: {dataset_dir}")
if dataset_dir.exists():
    app.mount(
        "/dataset",
        StaticFiles(directory=str(dataset_dir), check_dir=False),
        name="dataset",
    )
else:
    logger.warning(f"2D dataset directory does not exist: {dataset_dir}")

# Serve 3D model files (Pottery dataset)
# Example URL: /raw/3DModels/Amphora/Amphora_1.obj
raw_dir = settings.get_3D_dataset_path()
logger.info(f"Mounting 3D dataset directory: {raw_dir}")
if raw_dir.exists():
    app.mount(
        "/raw",
        StaticFiles(directory=str(raw_dir), check_dir=False),
        name="raw",
    )
else:
    logger.warning(f"3D dataset directory does not exist: {raw_dir}")

app.include_router(detect_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(samples_router, prefix="/api")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/health/dataset")
def health_dataset():
    dataset_root = settings.get_imagenet_dataset_path()
    raw_root = settings.get_3D_dataset_path()
    return {
        "2d_dataset_root": str(dataset_root),
        "2d_dataset_exists": dataset_root.exists(),
        "3d_dataset_root": str(raw_root),
        "3d_dataset_exists": raw_root.exists(),
    }