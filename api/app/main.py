from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.routers.detect import router as detect_router
from app.routers.search import router as search_router

app = FastAPI(title="ObjectLens API")

origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ THIS is where your indexed dataset images really are:
# /data/imagenet_yolo15/images/train/...  (or val, depending on what you indexed)
app.mount("/dataset", StaticFiles(directory=settings.DATASET_ROOT), name="dataset")

app.include_router(detect_router, prefix="/api")
app.include_router(search_router, prefix="/api")

@app.get("/health")
def health():
    return {"ok": True}
