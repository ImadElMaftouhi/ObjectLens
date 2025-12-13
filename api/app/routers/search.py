from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from fastapi import APIRouter, File, UploadFile, Query, Form

from app.db.mongo import get_collection
from app.core.config import settings

# ✅ Use ONLY what exists in feature_extraction.py
from app.services.feature_extraction import (
    FeatureExtractionService,
    SimilarityComputer,
    FourierDescriptorExtractor,
    OrientationHistogramExtractor,
    TamuraExtractor,
    GaborExtractor,
    HSVHistogramExtractor,
    DominantColorsExtractor,
)

router = APIRouter(prefix="/search", tags=["search"])


def _build_feature_service() -> FeatureExtractionService:
    # same extractors you already use in feature_extraction.py
    extractors = [
        FourierDescriptorExtractor(n_coeff=40),
        OrientationHistogramExtractor(bins=36),
        TamuraExtractor(kmax=4, n_bins=16),
        GaborExtractor(n_scales=3, n_orientations=4),
        HSVHistogramExtractor(bins=32, normalize=True),
        DominantColorsExtractor(n_colors=3),
    ]
    return FeatureExtractionService(extractors)


_feature_service = _build_feature_service()
_similarity = SimilarityComputer()  # uses weights from feature_extraction.py as-is


# In-memory cache:
# _FEATURES[i] = features dict for an indexed object (with numpy vectors)
# _META[i] = (image_path, bbox, class_id, class_name)
_FEATURES: List[Dict[str, Any]] = []
_META: List[Tuple[str, List[int], int, str]] = []


def _to_numpy_features(feats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert features loaded from Mongo (lists) back to numpy arrays,
    matching the structure produced by FeatureExtractionService.extract().
    """
    out: Dict[str, Any] = {}

    for cat_name, cat_val in feats.items():
        if not isinstance(cat_val, dict):
            continue

        out_cat: Dict[str, Any] = {}

        for k, v in cat_val.items():
            # combined vector
            if k == "combined":
                out_cat["combined"] = np.array(v, dtype=np.float32)
                continue

            # extractor entry: {'vector': ..., 'metadata': ..., 'name': ...}
            if isinstance(v, dict) and "vector" in v:
                out_cat[k] = {
                    "vector": np.array(v["vector"], dtype=np.float32),
                    "metadata": v.get("metadata", {}),
                    "name": v.get("name", k),
                }
            else:
                out_cat[k] = v

        out[cat_name] = out_cat

    return out


def _load_cache_from_mongo() -> None:
    """
    Loads all objects' stored 'features' from MongoDB into memory:
      - _FEATURES: list of feature dicts (converted to numpy)
      - _META: metadata per object
    """
    global _FEATURES, _META

    col = get_collection("images")

    features_list: List[Dict[str, Any]] = []
    meta_list: List[Tuple[str, List[int], int, str]] = []

    cursor = col.find(
        {},
        {
            "_id": 0,
            "image_path": 1,
            "objects.bbox": 1,
            "objects.class_id": 1,
            "objects.class_name": 1,
            "objects.features": 1,   # ✅ we only use stored features now
        },
    )

    for doc in cursor:
        image_path = str(doc.get("image_path", ""))
        for obj in doc.get("objects", []):
            feats = obj.get("features")
            if not feats:
                continue

            bbox = obj.get("bbox", [0, 0, 0, 0])
            class_id = int(obj.get("class_id", -1))
            class_name = str(obj.get("class_name", ""))

            features_list.append(_to_numpy_features(feats))
            meta_list.append((image_path, bbox, class_id, class_name))

    _FEATURES = features_list
    _META = meta_list


@router.post("/reload-cache")
def reload_cache():
    """
    Rebuilds the in-memory feature cache from MongoDB.
    Call this after running your offline indexing script.
    """
    _load_cache_from_mongo()
    return {"ok": True, "objects_indexed": len(_FEATURES)}


@router.post("/select-object")
async def select_object(
    crop: UploadFile = File(...),
    class_name: str | None = Form(None),
    confidence: float | None = Form(None),
    source_detection_id: str | None = Form(None),
    image_id: str | None = Form(None),
):
    """
    Optional debug endpoint. Keeps it if you still want to test uploads.
    Not required for retrieval.
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
    Upload an object crop -> extract features (feature_extraction.py)
    -> compute similarity vs all cached objects (object-level ranking)
    -> return Top-K objects and Top-K images.

    Ranking is OBJECT-level; images are derived from best object per image.
    """
    global _FEATURES, _META

    if not _FEATURES:
        _load_cache_from_mongo()

    if not _FEATURES:
        return {
            "ok": False,
            "error": "No objects indexed in cache. Run indexing script then /api/search/reload-cache.",
        }

    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Could not decode uploaded image."}

    # ✅ Extract query features using your FeatureExtractionService
    q_feats = _feature_service.extract(img, categories=["form", "texture", "color"])

    # Compute similarity against each cached object
    scores = np.empty((len(_FEATURES),), dtype=np.float32)
    for i, feats in enumerate(_FEATURES):
        scores[i] = float(_similarity.compute(q_feats, feats, selected_categories=["form", "texture", "color"]))

    k = min(int(top_k), scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]  # sorted desc

    best_objects: List[Dict[str, Any]] = []
    best_per_image: Dict[str, Dict[str, Any]] = {}

    for ii in idx:
        score = float(scores[int(ii)])
        image_path, bbox, class_id, class_name = _META[int(ii)]

        best_objects.append(
            {
                "image_path": image_path,
                "bbox": bbox,
                "class_id": class_id,
                "class_name": class_name,
                "score": score,
            }
        )

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

    # ✅ frontend expects a URL under /dataset (mounted in main.py)
    for item in best_images:
        item["image_url"] = f"/dataset/{item['image_path']}"

    return {
        "ok": True,
        "top_k": k,
        "best_images": best_images,
        "best_objects": best_objects,
    }
