from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.core.config import settings
from backend.routers.detect import router as detect_router
from backend.routers.search import router as search_router
from backend.routers.samples import router as samples_router

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
dataset_dir = Path(settings.DATASET_ROOT).resolve()
print(f"Mounting dataset directory: {dataset_dir}")
app.mount(
    "/dataset",
    StaticFiles(directory=str(dataset_dir), check_dir=False),
    name="dataset",
)

# Serve raw 3D files mounted into the backend at settings.RAW_DATA_ROOT
raw_dir = Path(settings.RAW_DATA_ROOT).resolve()
app.mount(
    "/raw",
    StaticFiles(directory=str(raw_dir), check_dir=False),
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

