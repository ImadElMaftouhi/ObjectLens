from __future__ import annotations

from fastapi import APIRouter, Query, Response
from pathlib import Path
from typing import List
from urllib.parse import quote

from app.core.config import settings

router = APIRouter(prefix="/samples", tags=["samples"])


@router.get("/class")
def list_samples(class_name: str = Query("Amphora"), limit: int = Query(5)) -> dict:
    """Return up to `limit` sample model file URLs for a given class.

    Files are expected under `settings.RAW_DATA_ROOT / '3D Models' / class_name`.
    The backend serves these files from `/raw/...` so the frontend should fetch
    model URLs via the API (example: `/raw/3D Models/Amphora/Amphora_1.glb`).
    """
    root = Path(settings.RAW_DATA_ROOT)
    class_dir = root / "3D Models" / class_name
    if not class_dir.exists() or not class_dir.is_dir():
        return {"ok": False, "error": "class not found", "files": []}

    files: List[str] = []
    for p in sorted(class_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".glb", ".gltf", ".obj", ".ply", ".stl"}:
            # Build URL relative to frontend static mount and URL-encode path
            rel_path = p.relative_to(root).as_posix()
            rel = "/raw/" + quote(rel_path, safe="/")
            files.append(rel)
            if len(files) >= int(limit):
                break

    return {"ok": True, "class": class_name, "files": files}


@router.head("/class")
def list_samples_head(class_name: str = Query("Amphora"), limit: int = Query(5)) -> Response:
    """Return headers for the samples list so HEAD requests succeed (e.g. curl -I).

    The body is intentionally omitted for HEAD; clients can call GET to receive JSON.
    """
    # Reuse the GET logic to validate inputs and build the list, but do not include body
    root = Path(settings.RAW_DATA_ROOT)
    class_dir = root / "3D Models" / class_name
    if not class_dir.exists() or not class_dir.is_dir():
        return Response(status_code=404)

    files: List[str] = []
    for p in sorted(class_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".glb", ".gltf", ".obj", ".ply", ".stl"}:
            rel_path = p.relative_to(root).as_posix()
            rel = "/raw/" + quote(rel_path, safe="/")
            files.append(rel)
            if len(files) >= int(limit):
                break

    # Return OK with no body for HEAD
    return Response(status_code=200)
