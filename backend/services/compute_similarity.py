from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

from .feature_extraction import (
    FourierDescriptorExtractor,
    OrientationHistogramExtractor,
    TamuraExtractor,
    GaborExtractor,
    HSVHistogramExtractor,
    FeatureExtractionService,
    DominantColorsExtractor,
    SimilarityComputer,
)

# -----------------------------------------------------------------------------
# Feature + similarity services (MUST match indexing config)
# -----------------------------------------------------------------------------

EXTRACTORS = [
    FourierDescriptorExtractor(n_coeff=40),
    OrientationHistogramExtractor(bins=36),
    TamuraExtractor(kmax=4, n_bins=16),
    GaborExtractor(n_scales=3, n_orientations=4),
    HSVHistogramExtractor(h_bins=4, sv_bins=4),
    DominantColorsExtractor(n_colors=3),
]

FEATURE_SERVICE = FeatureExtractionService(EXTRACTORS)
SIMILARITY_SERVICE = SimilarityComputer()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _to_numpy(obj: Any) -> Any:
    """Recursively convert lists to numpy arrays where helpful."""
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        try:
            return np.array(obj, dtype=np.float32)
        except Exception:
            return [_to_numpy(v) for v in obj]
    return obj

# -----------------------------------------------------------------------------
# Query feature extraction (used by API)
# -----------------------------------------------------------------------------
def extract_query_features(
    image: np.ndarray,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extract features from a query object crop.
    Returns dict with per-category features including 'combined' vectors.
    """
    return FEATURE_SERVICE.extract(image, categories=categories)


def extract_query_vector(
    image: np.ndarray,
    categories: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extract single combined feature vector from query image (for FAISS).
    
    This matches the format used during indexing:
    - Combines form, texture, color with weights: 0.5, 0.3, 0.2
    - L2-normalizes the final vector
    
    Args:
        image: Query image (BGR numpy array)
        categories: Feature categories to use (default: all)
        
    Returns:
        Combined normalized vector (shape: [D])
    """
    if categories is None:
        categories = ["form", "texture", "color"]
    
    # Extract features (same as extract_query_features)
    features = FEATURE_SERVICE.extract(image, categories=categories)
    
    # Combine with same weights as indexing
    all_combined = []
    for cat in categories:
        if cat in features and "combined" in features[cat]:
            all_combined.append(features[cat]["combined"])
    
    if not all_combined:
        raise ValueError("No combined vectors found in features")
    
    # Combine with weights: form=0.5, texture=0.3, color=0.2
    weights = {"form": 0.5, "texture": 0.3, "color": 0.2}
    weighted_vectors = []
    
    for cat in categories:
        if cat in features and "combined" in features[cat]:
            weight = weights.get(cat, 1.0 / len(categories))
            weighted_vectors.append(features[cat]["combined"] * weight)
    
    final_vector = np.concatenate(weighted_vectors).astype(np.float32)
    
    # L2 normalize
    norm = np.linalg.norm(final_vector)
    if norm > 0:
        final_vector /= norm
    
    return final_vector


# -----------------------------------------------------------------------------
# Core similarity search (FAISS + MongoDB)
# -----------------------------------------------------------------------------
def _extract_vector_from_features(
        query_features: Dict[str, Any],
        categories: Optional[List[str]] = None
    ) -> np.ndarray:
    """
    Extract combined vector from query_features dict (same format as indexing).
    Used internally by search_with_class_filter to prepare query for FAISS.
    """
    if categories is None:
        categories = ["form", "texture", "color"]
    
    # Combine with weights: form=0.5, texture=0.3, color=0.2 (matches indexing)
    weights = {"form": 0.5, "texture": 0.3, "color": 0.2}
    weighted_vectors = []
    
    for cat in categories:
        if cat in query_features and "combined" in query_features[cat]:
            weight = weights.get(cat, 1.0 / len(categories))
            vec = np.array(query_features[cat]["combined"], dtype=np.float32)
            weighted_vectors.append(vec * weight)
    
    if not weighted_vectors:
        raise ValueError("No combined vectors found in query_features")
    
    final_vector = np.concatenate(weighted_vectors).astype(np.float32)
    
    # L2 normalize
    norm = np.linalg.norm(final_vector)
    if norm > 0:
        final_vector /= norm
    
    return final_vector


def search_with_class_filter(
        query_features: Dict[str, Any],
        query_class: str,
        base_features: Dict[str, Dict] | None = None,  # Optional - kept for backward compatibility
        top_k: int = 10,
        metric: str = "cosine",
        categories: Optional[List[str]] = None,
        same_class_only: bool = True,
        use_faiss: bool = True,  # Use FAISS if available
    ) -> List[Dict[str, Any]]:
    
    """
    Return top-k matching OBJECTS across the dataset.
    
    Uses FAISS for fast similarity search when available, otherwise falls back
    to in-memory search using base_features.

    Args:
        query_features: Feature dict with 'form', 'texture', 'color' categories
        query_class: Class name to filter by (if same_class_only=True)
        base_features: Optional dict of features for fallback (deprecated when using FAISS)
        top_k: Number of results to return
        metric: Similarity metric ("cosine" or "euclidean") - only cosine supported for FAISS
        categories: Feature categories to use
        same_class_only: Filter results to same class as query_class
        use_faiss: Use FAISS if available (default: True)

    Returns:
      [
        {
          "image_path": str,
          "object_id": int,
          "bbox": [x1,y1,x2,y2],
          "class_id": int,
          "class_name": str,
          "confidence": float,
          "score": float
        },
        ...
      ]
    """
    if categories is None:
        categories = ["form", "texture", "color"]

    # Try FAISS first (if enabled and available)
    if use_faiss:
        try:
            from backend.services.faiss_service import get_faiss_service
            from backend.db.mongo import get_collection
            
            # Extract query vector from features
            query_vector = _extract_vector_from_features(query_features, categories)
            
            # Get FAISS service
            faiss_service = get_faiss_service()
            
            # Get MongoDB collection
            objects_col = get_collection("objects")
            
            # FILTER FIRST: Get all FAISS IDs for the class (if same_class_only)
            effective_same_class = bool(same_class_only and query_class)
            
            if effective_same_class:
                # Filter by class FIRST - get all FAISS IDs for this class from MongoDB
                # This ensures similarity is computed ONLY within the same class
                mongo_docs = objects_col.find(
                    {"class_name": query_class},
                    {"faiss_id": 1}  # Only need faiss_id for filtering
                )
                filtered_faiss_ids = [doc["faiss_id"] for doc in mongo_docs if "faiss_id" in doc]
                
                if not filtered_faiss_ids:
                    return []  # No objects in this class
                
                # Search ONLY within the filtered subset using FAISS
                faiss_results = faiss_service.search_filtered(
                    query_vector, 
                    filtered_faiss_ids, 
                    top_k=int(top_k)
                )
            else:
                # No class filtering - search all vectors
                faiss_results = faiss_service.search(query_vector, top_k=int(top_k))
            
            # Get metadata for results
            faiss_ids = [r[0] for r in faiss_results]
            if not faiss_ids:
                return []
            
            mongo_docs = objects_col.find({"faiss_id": {"$in": faiss_ids}})
            id_to_doc = {doc["faiss_id"]: doc for doc in mongo_docs}
            
            # Build results (already filtered by class if same_class_only was True)
            results: List[Dict[str, Any]] = []
            for faiss_id, score, object_id in faiss_results:
                if faiss_id not in id_to_doc:
                    continue
                
                doc = id_to_doc[faiss_id]
                results.append({
                    "image_path": str(doc.get("image_path", "")),
                    "object_id": int(doc.get("object_idx", -1)),
                    "bbox": doc.get("bbox", [0, 0, 0, 0]),
                    "class_id": int(doc.get("class_id", -1)),
                    "class_name": str(doc.get("class_name", "unknown")),
                    "confidence": float(doc.get("confidence", 0.0)),
                    "score": float(score),  # FAISS similarity score
                })
            
            # Results are already sorted by FAISS (descending by score)
            return results[:max(1, int(top_k))]
            
        except Exception as e:
            # Fall back to in-memory search if FAISS fails
            if base_features is None:
                raise RuntimeError(
                    f"FAISS search failed ({str(e)}) and base_features not provided. "
                    "Ensure FAISS index is available or provide base_features for fallback."
                )
            # Continue to fallback implementation below
    
    # Fallback: In-memory search using base_features (original implementation)
    if base_features is None:
        raise ValueError("base_features required when use_faiss=False")
    
    results: List[Dict[str, Any]] = []

    # Iterate all images -> all objects
    for image_path, data in base_features.items():
        objects = data.get("objects", []) or []
        if not objects:
            continue

        for obj_id, obj in enumerate(objects):
            obj_class = str(obj.get("class_name", "unknown"))

            if same_class_only and obj_class != query_class:
                continue

            feats = obj.get("features") or {}
            feats = _to_numpy(feats)

            # SimilarityComputer expects combined vectors to exist
            try:
                score = float(
                    SIMILARITY_SERVICE.compute(
                        query_features, feats, categories=categories, metric=metric
                    )
                )
            except Exception:
                continue

            if score <= 0:
                continue

            results.append(
                {
                    "image_path": image_path,
                    "object_id": int(obj.get("object_id", obj_id)),
                    "bbox": obj.get("bbox", [0, 0, 0, 0]),
                    "class_id": int(obj.get("class_id", -1)),
                    "class_name": obj_class,
                    "confidence": float(obj.get("confidence", 0.0)),
                    "score": score,
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[: max(1, int(top_k))]
