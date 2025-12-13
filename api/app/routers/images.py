from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from app.core.config import settings

router = APIRouter(prefix="/images", tags=["images"])

@router.get("/{image_path:path}")
def get_image(image_path: str):
    # image_path is like "images/val/xxx.jpg" (stored in Mongo)
    root = Path(settings.DATASET_ROOT)
    full = (root / image_path).resolve()

    # Security: ensure it stays inside dataset root
    if not str(full).startswith(str(root.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not full.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(full))
