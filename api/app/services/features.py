from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

from app.services.feature_extraction import (
    FeatureExtractionService,
    FourierDescriptorExtractor,
    OrientationHistogramExtractor,
    TamuraExtractor,
    GaborExtractor,
    HSVHistogramExtractor,
    DominantColorsExtractor,
)


def build_feature_service() -> FeatureExtractionService:
    extractors = [
        FourierDescriptorExtractor(n_coeff=40),
        OrientationHistogramExtractor(bins=36),
        TamuraExtractor(kmax=4, n_bins=16),
        GaborExtractor(n_scales=3, n_orientations=4),
        HSVHistogramExtractor(bins=32, normalize=True),
        DominantColorsExtractor(n_colors=3),
    ]
    return FeatureExtractionService(extractors)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        return (v / n).astype(np.float32)
    return v.astype(np.float32)


def extract_object_features(service: FeatureExtractionService, crop_bgr: np.ndarray) -> Tuple[Dict[str, Any], List[float]]:
    """
    Returns:
      - features dict (form/texture/color with extractor vectors + combined vectors)
      - final_vector as python list[float] (L2-normalized)
    """
    feats = service.extract(crop_bgr, categories=["form", "texture", "color"])

    form_comb = feats["form"]["combined"]
    tex_comb = feats["texture"]["combined"]
    col_comb = feats["color"]["combined"]

    final_vec = np.concatenate([form_comb, tex_comb, col_comb]).astype(np.float32)
    final_vec = l2_normalize(final_vec)

    return feats, final_vec.astype(np.float64).tolist()  # store as doubles in Mongo
