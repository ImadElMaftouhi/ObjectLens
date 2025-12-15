from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import cv2
from fastapi import APIRouter, File, UploadFile, Query, Form

from app.db.mongo import get_collection
from app.core.config import settings

from app.services.compute_similarity import (
    extract_query_features,
    search_with_class_filter,
)

router = APIRouter(prefix="/search", tags=["search"])

# -----------------------------------------------------------------------------
# In-memory cache for fast similarity search (Mongo -> dict)
# -----------------------------------------------------------------------------
_BASE_FEATURES: Optional[Dict[str, Dict[str, Any]]] = None


def _to_numpy(obj: Any) -> Any:
    """Recursively convert lists back to numpy arrays where helpful."""
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        try:
            return np.array(obj, dtype=np.float32)
        except Exception:
            return [_to_numpy(v) for v in obj]
    return obj


def _load_cache_from_mongo() -> None:
    """
    Load object-level features from MongoDB into memory in the structure required by
    compute_similarity.search_with_class_filter().
    """
    global _BASE_FEATURES
    col = get_collection()

    base: Dict[str, Dict[str, Any]] = {}

    cursor = col.find(
        {},
        {
            "_id": 0,
            "image_path": 1,
            "objects.bbox": 1,
            "objects.class_id": 1,
            "objects.class_name": 1,
            "objects.confidence": 1,   # optional but recommended
            "objects.features": 1,     # ✅ new
        },
    )

    for doc in cursor:
        image_path = str(doc.get("image_path", "")).strip()
        if not image_path:
            continue

        objects = doc.get("objects", []) or []
        cleaned_objects: List[Dict[str, Any]] = []

        for obj in objects:
            feats = obj.get("features") or {}

            # Require combined vectors (SimilarityComputer expects those)
            f_form = feats.get("form") or {}
            f_tex = feats.get("texture") or {}
            f_col = feats.get("color") or {}

            if "combined" not in f_form or "combined" not in f_tex or "combined" not in f_col:
                continue

            cleaned_objects.append(
                {
                    "bbox": obj.get("bbox", [0, 0, 0, 0]),
                    "class_id": int(obj.get("class_id", -1)),
                    "class_name": str(obj.get("class_name", "unknown")),
                    "confidence": float(obj.get("confidence", 0.0)),
                    "features": _to_numpy(feats),
                }
            )

        if not cleaned_objects:
            continue

        base[image_path] = {
            "num_objects": len(cleaned_objects),
            "objects": cleaned_objects,
        }

    _BASE_FEATURES = base


@router.post("/reload-cache")
def reload_cache():
    """
    Rebuild the in-memory feature cache from MongoDB.
    Call this after running your offline indexing script.
    """
    _load_cache_from_mongo()
    count_imgs = 0 if not _BASE_FEATURES else len(_BASE_FEATURES)
    count_objs = 0 if not _BASE_FEATURES else sum(v["num_objects"] for v in _BASE_FEATURES.values())
    return {"ok": True, "images_indexed": count_imgs, "objects_indexed": count_objs}


@router.post("/select-object")
async def select_object(
    crop: UploadFile = File(...),
    class_name: str | None = Form(None),
    confidence: float | None = Form(None),
    source_detection_id: str | None = Form(None),
    image_id: str | None = Form(None),
):
    """
    Receives the selected crop from frontend (optional debug endpoint).
    """
    data = await crop.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Could not decode crop image."}

    h, w = img.shape[:2]
    return {
        "ok": True,
        "message": "Crop received",
        "shape": [h, w],
        "meta": {
            "class_name": class_name,
            "confidence": confidence,
            "source_detection_id": source_detection_id,
            "image_id": image_id,
        },
    }


@router.post("/topk")
async def topk(
    file: UploadFile = File(...),
    top_k: int = Query(default=settings.TOPK_DEFAULT, ge=1, le=200),
    metric: str = Query(default="cosine"),
    # ✅ class filtering (frontend can send this now or later)
    query_class: str | None = Form(None),
    same_class_only: bool = Query(default=True),
):
    """
    Upload an object crop -> extract query features -> retrieve Top-K objects.
    Then aggregate into Top-K images by best object score.

    Returns:
      - best_images: unique images sorted by best object score
      - best_objects: raw top objects (debug)
    """
    global _BASE_FEATURES

    if _BASE_FEATURES is None:
        _load_cache_from_mongo()

    if not _BASE_FEATURES:
        return {
            "ok": False,
            "error": "No objects indexed in cache. Run indexing script then /api/search/reload-cache.",
        }

    if metric not in ["cosine", "euclidean"]:
        return {"ok": False, "error": "metric must be 'cosine' or 'euclidean'."}

    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Could not decode uploaded image."}

    # Extract query features from crop
    q_feats = extract_query_features(img)

    # Enable class filtering only if we actually have query_class
    effective_same_class = bool(same_class_only and query_class)

    # Get top matched OBJECTS
    best_objects = search_with_class_filter(
        query_features=q_feats,
        query_class=query_class or "unknown",
        base_features=_BASE_FEATURES,
        top_k=int(top_k),
        metric=metric,
        categories=["form", "texture", "color"],
        same_class_only=effective_same_class,
    )

    # Aggregate -> best per image
    best_per_image: Dict[str, Dict[str, Any]] = {}
    for obj in best_objects:
        image_path = obj["image_path"]
        score = float(obj["score"])

        prev = best_per_image.get(image_path)
        if prev is None or score > float(prev["score"]):
            best_per_image[image_path] = {
                "image_path": image_path,
                "score": score,
                "best_bbox": obj.get("bbox", [0, 0, 0, 0]),
                "best_class_id": int(obj.get("class_id", -1)),
                "best_class_name": str(obj.get("class_name", "unknown")),
                "best_object_id": int(obj.get("object_id", -1)),
                "best_confidence": float(obj.get("confidence", 0.0)),
            }

    # Sort unique images by score
    best_images = sorted(best_per_image.values(), key=lambda x: x["score"], reverse=True)
    best_images = best_images[: min(int(top_k), len(best_images))]

    # Add URL usable by frontend
    for item in best_images:
        item["image_url"] = f"/dataset/{item['image_path']}"

    return {
        "ok": True,
        "top_k": int(top_k),
        "metric": metric,
        "same_class_only": effective_same_class,
        "query_class": query_class,
        "best_images": best_images,
        "best_objects": best_objects,  # debug: per-object matches
        "query_feature_categories": list(q_feats.keys()),
    }
