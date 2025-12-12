from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from fastapi import APIRouter, File, UploadFile, Query, Form

from app.db.mongo import get_collection
from app.services.features import build_feature_service, extract_object_features, l2_normalize
from app.core.config import settings

router = APIRouter(prefix="/search", tags=["search"])

_feature_service = build_feature_service()

# In-memory cache for fast similarity search
# vectors: (N, D) float32, unit-normalized
# meta[i] = (image_path, bbox, class_id, class_name)
_VECTORS: Optional[np.ndarray] = None
_META: List[Tuple[str, List[int], int, str]] = []


def _load_cache_from_mongo() -> None:
    """
    Loads all objects' final_vector from MongoDB into memory:
      - _VECTORS: stacked vectors (N, D)
      - _META: metadata for each vector index
    """
    global _VECTORS, _META

    col = get_collection()

    vectors: List[np.ndarray] = []
    meta: List[Tuple[str, List[int], int, str]] = []

    cursor = col.find(
        {},
        {
            "_id": 0,
            "image_path": 1,
            "objects.bbox": 1,
            "objects.class_id": 1,
            "objects.class_name": 1,
            "objects.final_vector": 1,
        },
    )

    for doc in cursor:
        image_path = doc.get("image_path", "")
        for obj in doc.get("objects", []):
            fv = obj.get("final_vector")
            if not fv:
                continue

            v = np.array(fv, dtype=np.float32)
            v = l2_normalize(v)

            bbox = obj.get("bbox", [0, 0, 0, 0])
            class_id = int(obj.get("class_id", -1))
            class_name = str(obj.get("class_name", ""))

            vectors.append(v)
            meta.append((image_path, bbox, class_id, class_name))

    if vectors:
        _VECTORS = np.vstack(vectors).astype(np.float32)
        _META = meta
    else:
        _VECTORS = None
        _META = []


@router.post("/reload-cache")
def reload_cache():
    """
    Rebuilds the in-memory vector index from MongoDB.
    Call this after running your offline indexing script.
    """
    _load_cache_from_mongo()
    count = 0 if _VECTORS is None else int(_VECTORS.shape[0])
    return {"ok": True, "objects_indexed": count}


@router.post("/select-object")
async def select_object(
    crop: UploadFile = File(...),
    class_name: str | None = Form(None),
    confidence: float | None = Form(None),
    source_detection_id: str | None = Form(None),
    image_id: str | None = Form(None),
):
    """
    Receives the selected crop (thumbnail) from frontend.
    This is optional but useful for debugging/confirmation.
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
):
    """
    Upload an object crop -> compute query final_vector -> retrieve Top-K images.

    Returns:
      - best_images: unique images sorted by best object score
      - best_objects: raw top objects (debug)
    """
    global _VECTORS, _META

    if _VECTORS is None:
        _load_cache_from_mongo()

    if _VECTORS is None or _VECTORS.shape[0] == 0:
        return {
            "ok": False,
            "error": "No objects indexed in cache. Run indexing script then /api/search/reload-cache.",
        }

    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Could not decode uploaded image."}

    _, q_vec_list = extract_object_features(_feature_service, img)
    q = np.array(q_vec_list, dtype=np.float32)
    q = l2_normalize(q)

    # cosine similarity since vectors are L2-normalized => dot product
    scores = _VECTORS @ q  # (N,)

    k = min(int(top_k), scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]

    best_objects = []
    best_per_image: Dict[str, Dict[str, Any]] = {}

    for i in idx:
        score = float(scores[i])
        image_path, bbox, class_id, class_name = _META[int(i)]

        best_objects.append(
            {
                "image_path": image_path,
                "bbox": bbox,
                "class_id": class_id,
                "class_name": class_name,
                "score": score,
            }
        )

        # keep best score per image
        prev = best_per_image.get(image_path)
        if prev is None or score > prev["score"]:
            best_per_image[image_path] = {
                "image_path": image_path,
                "score": score,
                "best_bbox": bbox,
                "best_class_id": class_id,
                "best_class_name": class_name,
            }

    best_images = sorted(best_per_image.values(), key=lambda x: x["score"], reverse=True)[:k]

    # ✅ add URL usable by frontend
    for item in best_images:
        # image_path in DB looks like "images/val/xxx.jpg"
        # but the real file is in IMAGE_STORE_DIR mounted as /files/
        item["image_url"] = f"/dataset/{item['image_path']}"



    return {
        "ok": True,
        "top_k": k,
        "best_images": best_images,
        "best_objects": best_objects,  # useful for debugging
        "query_vector_dim": int(q.shape[0]),
    }
