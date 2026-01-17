from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.core.config import settings
from app.routers.detect import router as detect_router
from app.routers.search import router as search_router
from app.routers.samples import router as samples_router

app = FastAPI(title="ObjectLens API")

origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve dataset images directly (Option A)
# Example URL: /dataset/images/val/xxx.jpg
app.mount(
    "/dataset",
    StaticFiles(directory=settings.DATASET_ROOT, check_dir=False),
    name="dataset",
)

# Serve raw 3D files mounted into the backend at settings.RAW_DATA_ROOT
app.mount(
    "/raw",
    StaticFiles(directory=settings.RAW_DATA_ROOT, check_dir=False),
    name="raw",
)

app.include_router(detect_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(samples_router, prefix="/api")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/health/dataset")
def health_dataset():
    root = Path(settings.DATASET_ROOT)
    return {"dataset_root": str(root), "exists": root.exists()}
