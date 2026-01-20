"""
Search for objects in an image and return the bounding boxes.
"""

from __future__ import annotations
import cv2
import tempfile
import numpy as np
from pathlib import Path
from pymongo import MongoClient
from pymongo.collection import Collection
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form

# for 2D similarity
from backend.db.mongo import get_collection
from backend.core.config import settings
from backend.utils.descriptor_viz import build_query_descriptor_viz
from backend.services.compute_similarity import (
    extract_query_features,
    search_with_class_filter,
)

# for 3D similarity
from backend.services.mesh import MeshLoader, Mesh, MeshNormalizer, Renderer
from backend.services.descriptors import (
    LFDDescriptor,
    LFDModelDescriptor,
    LFDMetadata,
    DepthBufferDescriptor,
    DepthModelDescriptor,
    DepthMetadata
)
from backend.services.similarity import SimilarityEngine


router = APIRouter(prefix="/search", tags=["search"])

# -----------------------------------------------------------------------------
# In-memory cache for fast similarity search (Mongo -> dict)
# -----------------------------------------------------------------------------
_BASE_FEATURES: Optional[Dict[str, Dict[str, Any]]] = None


# Configuration
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_DB = "objectlens"
DEFAULT_COLLECTION = "models"
DEFAULT_IMAGE_SIZE = 256
DEFAULT_METHOD = "depth"
DEFAULT_METRIC = "l2"
DEFAULT_AGGREGATION = "mean"
DEFAULT_DEPTH_ROTATION_SET = "grid24"


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
            "objects.confidence": 1,
            "objects.features": 1,
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


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization"""
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return (X / n).astype(np.float32)


def _load_3D_index_descriptors(
    coll: Collection, 
    method: str
) -> tuple[List[str], List[str], List[Any]]:
    """
    Load indexed descriptors from MongoDB.
    Returns: (index_ids, index_classes, index_descriptors)
    """
    if method == "lfd":
        proj = {
            "class": 1,
            "lfd.features": 1,
            "lfd.ring_start": 1,
            "lfd.ring_size": 1,
            "lfd.directions": 1,
            "lfd.image_size": 1,
            "lfd.l2_normalized": 1,
        }
        cursor = coll.find({"lfd.features": {"$exists": True, "$ne": []}}, proj)

        index_ids, index_cls, index_desc = [], [], []

        for doc in cursor:
            fn = doc["_id"]
            cl = doc.get("class", "UNKNOWN")

            feats = np.asarray(doc["lfd"]["features"], dtype=np.float32)
            ring_start = int(doc["lfd"].get("ring_start", 2))
            ring_size = int(doc["lfd"].get("ring_size", 8))
            image_size = int(doc["lfd"].get("image_size", 128))

            directions = doc["lfd"].get("directions", None)
            if directions is None:
                directions_np = np.zeros((int(feats.shape[0]), 3), dtype=np.float32)
            else:
                directions_np = np.asarray(directions, dtype=np.float32)

            meta = LFDMetadata(
                preset="LFD_10",
                image_size=image_size,
                ring_start=ring_start,
                ring_size=ring_size,
                directions=directions_np,
            )
            desc = LFDModelDescriptor(features=feats, meta=meta)

            index_ids.append(fn)
            index_cls.append(cl)
            index_desc.append(desc)

        return index_ids, index_cls, index_desc

    elif method == "depth":
        proj = {
            "class": 1,
            "depth.features": 1,
            "depth.directions": 1,
            "depth.image_size": 1,
            "depth.linearized_depth": 1,
            "depth.l2_normalized": 1,
            "depth.rotation_set": 1,
        }
        cursor = coll.find({"depth.features": {"$exists": True, "$ne": []}}, proj)

        index_ids, index_cls, index_desc = [], [], []

        for doc in cursor:
            fn = doc["_id"]
            cl = doc.get("class", "UNKNOWN")

            feats = np.asarray(doc["depth"]["features"], dtype=np.float32)
            directions = np.asarray(doc["depth"]["directions"], dtype=np.float32)
            image_size = int(doc["depth"].get("image_size", 128))
            linearized_depth = bool(doc["depth"].get("linearized_depth", True))
            rotation_set = str(doc["depth"].get("rotation_set", "UNKNOWN"))
            l2_normalized = bool(doc["depth"].get("l2_normalized", False))

            meta = DepthMetadata(
                preset="DEPTH_42",
                image_size=image_size,
                directions=directions,
                linearized_depth=linearized_depth,
                rotation_set=rotation_set,
                l2_normalized=l2_normalized,
            )
            desc = DepthModelDescriptor(features=feats, meta=meta)

            index_ids.append(fn)
            index_cls.append(cl)
            index_desc.append(desc)

        return index_ids, index_cls, index_desc

    raise ValueError(f"Unknown method: {method}")


def _compute_3D_query_descriptor(
    method: str,
    renderer: Renderer,
    image_size: int,
    mesh: Mesh,
    l2_normalize: bool = False,
) -> Any:
    """Compute descriptor for query model"""
    mesh = MeshNormalizer.normalize(mesh)

    if method == "lfd":
        lfd = LFDDescriptor(renderer=renderer, image_size=image_size)
        desc = lfd.compute(mesh)
        if l2_normalize:
            feats = _l2_normalize_rows(np.asarray(desc.features, dtype=np.float32))
            return LFDModelDescriptor(features=feats, meta=desc.meta)
        return desc

    elif method == "depth":
        depth = DepthBufferDescriptor(renderer=renderer, image_size=image_size)
        desc = depth.compute(mesh)
        if l2_normalize:
            feats = _l2_normalize_rows(np.asarray(desc.features, dtype=np.float32))
            return DepthModelDescriptor(features=feats, meta=desc.meta)
        return desc

    raise ValueError(f"Unknown method: {method}")


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
        query_class: str | None = Form(None),
        same_class_only: bool = Query(default=True),
        include_viz: bool = Query(default=False),
    ):
    """
    Upload an object crop -> extract query features -> retrieve Top-K objects.
    Uses FAISS-powered search_with_class_filter for fast similarity search.
    
    Optionally returns meaningful descriptor visualizations for the query object.

    Returns:
      - best_images: unique images sorted by best object score
      - best_objects: raw top objects (debug)
      - query_descriptors (optional): summaries + base64 PNGs
    """
    # Decode image
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Could not decode uploaded image."}
    
    # Extract query features (hierarchical structure)
    q_feats = extract_query_features(img)
    
    # Optional: build visualizations
    query_descriptors = None
    if include_viz:
        try:
            query_descriptors = build_query_descriptor_viz(img, q_feats)
        except Exception as e:
            query_descriptors = {"error": str(e)}
    
    # Use search_with_class_filter (handles FAISS internally)
    effective_same_class = bool(same_class_only and query_class)
    
    try:
        best_objects = search_with_class_filter(
            query_features=q_feats,
            query_class=query_class or "unknown",
            base_features=None,  # Not needed when using FAISS
            top_k=int(top_k),
            metric=metric,
            categories=["form", "texture", "color"],
            same_class_only=effective_same_class,
            use_faiss=True,  # Use FAISS for fast search
        )
    except Exception as e:
        return {
            "ok": False,
            "error": f"Search failed: {str(e)}"
        }
    
    # Group by image (same as before)
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
    
    best_images = sorted(best_per_image.values(), key=lambda x: x["score"], reverse=True)
    best_images = best_images[:min(int(top_k), len(best_images))]
    
    for item in best_images:
        item["image_url"] = f"/dataset/{item['image_path']}"
    
    resp = {
        "ok": True,
        "top_k": int(top_k),
        "metric": metric,  # FAISS uses cosine (IndexFlatIP)
        "same_class_only": effective_same_class,
        "query_class": query_class,
        "best_images": best_images,
        "best_objects": best_objects[:top_k],  # Limit to top_k
        "query_feature_categories": list(q_feats.keys()),
    }
    
    if include_viz:
        resp["query_descriptors"] = query_descriptors
    
    return resp
 

@router.post("/3D-topk")
async def topK_3D(
    file: UploadFile = File(..., description="3D model file (.obj, .stl, .glb, .ply)"),
    top_k: int = Query(10, ge=1, le=100, description="Number of similar models to return"),
    method: str = Query(DEFAULT_METHOD, description="Descriptor method: lfd or depth"),
    metric: str = Query(DEFAULT_METRIC, description="Distance metric: l2, l1, or cosine"),
    aggregation: str = Query(DEFAULT_AGGREGATION, description="Aggregation: mean or sum"),
    same_class_only: bool = Query(False, description="Return only models from same class"),
    l2_normalize: bool = Query(False, description="Apply L2 normalization to query features"),
    image_size: int = Query(DEFAULT_IMAGE_SIZE, ge=64, le=512, description="Rendering resolution"),
    mongo_uri: str = Query(DEFAULT_MONGO_URI, description="MongoDB connection URI"),
    db_name: str = Query(DEFAULT_DB, description="Database name"),
    collection_name: str = Query(DEFAULT_COLLECTION, description="Collection name"),
) -> Dict[str, Any]:
    """
    Search for top-K most similar 3D models.
    
    Upload a 3D model file and get the most similar models from the indexed database.
    Supports both LFD (Light Field Descriptor) and Depth Buffer methods.
    """
    
    # Validate method
    if method not in ["lfd", "depth"]:
        raise HTTPException(status_code=400, detail="Method must be 'lfd' or 'depth'")
    
    # Validate metric
    if metric not in ["l2", "l1", "cosine"]:
        raise HTTPException(status_code=400, detail="Metric must be 'l2', 'l1', or 'cosine'")
    
    # Validate aggregation
    if aggregation not in ["mean", "sum"]:
        raise HTTPException(status_code=400, detail="Aggregation must be 'mean' or 'sum'")
    
    # Validate file extension
    allowed_extensions = {".obj", ".stl", ".glb", ".ply"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    temp_file = None
    renderer = None
    client = None
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
            content = await file.read()
            temp.write(content)
            temp_file = Path(temp.name)
        
        # Load and normalize mesh
        mesh = MeshLoader.load(temp_file)
        
        # Create renderer
        renderer = Renderer(width=image_size, height=image_size)
        
        # Compute query descriptor
        query_desc = _compute_3D_query_descriptor(
            method=method,
            renderer=renderer,
            image_size=image_size,
            mesh=mesh,
            l2_normalize=l2_normalize,
        )
        
        # Connect to MongoDB
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        coll = client[db_name][collection_name]
        
        # Load index descriptors
        index_ids, index_cls, index_desc = _load_3D_index_descriptors(coll, method)
        
        if not index_desc:
            raise HTTPException(
                status_code=404,
                detail=f"No indexed models found for method '{method}'"
            )
        
        # Compute similarities
        engine = SimilarityEngine(
            metric=metric,
            aggregation=aggregation,
            depth_rotation_set=DEFAULT_DEPTH_ROTATION_SET
        )
        
        distances = np.empty(len(index_desc), dtype=np.float32)
        for i, idx_desc in enumerate(index_desc):
            distances[i] = float(engine.compare(query_desc, idx_desc).distance)
        
        # Sort by distance (ascending = more similar)
        order = np.argsort(distances)
        
        # Apply filters and get top-K
        results = []
        for idx in order:
            # Skip if same_class_only filter is active
            if same_class_only and index_cls[idx] != index_cls[idx]:
                # Note: We don't have the query class here, so same_class_only
                # would need the query to come from the indexed dataset
                # For now, we'll skip this filter
                pass
            
            results.append({
                "rank": len(results) + 1,
                "filename": index_ids[idx],
                "class": index_cls[idx],
                "distance": float(distances[idx]),
                "similarity_score": float(1.0 / (1.0 + distances[idx])),  # Normalized score
            })
            
            if len(results) >= top_k:
                break
        
        return {
            "query_filename": file.filename,
            "method": method,
            "metric": metric,
            "aggregation": aggregation,
            "top_k": top_k,
            "num_results": len(results),
            "num_indexed": len(index_desc),
            "results": results,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
    finally:
        # Cleanup
        if temp_file and temp_file.exists():
            temp_file.unlink()
        if renderer:
            try:
                renderer.close()
            except:
                pass
        if client:
            try:
                client.close()
            except:
                pass
