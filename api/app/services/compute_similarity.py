from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np

from app.services.feature_extraction import (
    FourierDescriptorExtractor,
    OrientationHistogramExtractor,
    TamuraExtractor,
    GaborExtractor,
    HSVHistogramExtractor,
    DominantColorsExtractor,
    FeatureExtractionService,
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


# -----------------------------------------------------------------------------
# Core similarity search (Mongo / in-memory features)
# -----------------------------------------------------------------------------
def search_with_class_filter(
    query_features: Dict[str, Any],
    query_class: str,
    base_features: Dict[str, Dict],
    top_k: int = 10,
    metric: str = "cosine",
    categories: Optional[List[str]] = None,
    same_class_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    Return top-k matching OBJECTS across the dataset.

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
