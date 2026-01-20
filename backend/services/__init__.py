#! /usr/bin/env python
# Service Layer

from .compute_similarity import (
    extract_query_features,
    search_with_class_filter,
    )
from .feature_extraction import (
    FourierDescriptorExtractor,
    OrientationHistogramExtractor,
    TamuraExtractor,
    GaborExtractor,
    HSVHistogramExtractor,
    DominantColorsExtractor,
    FeatureExtractionService,
    )
from .yolo_service import YoloService

from .descriptors import (
    DepthBufferDescriptor, DepthModelDescriptor, DepthMetadata,
    LFDDescriptor, LFDModelDescriptor, LFDMetadata
)
from .mesh import (
    Mesh, MeshLoader, MeshNormalizer, Renderer, ViewGenerator, CameraPose
)
from .similarity import SimilarityEngine, SimilarityResult

__all__ = [
    # 2D features
    "extract_query_features",
    "search_with_class_filter",
    "FourierDescriptorExtractor",
    "OrientationHistogramExtractor",
    "TamuraExtractor",
    "GaborExtractor",
    "HSVHistogramExtractor",
    "DominantColorsExtractor",
    "FeatureExtractionService",
    "YoloService",
    # 3D features
    ## Descriptors
    "DepthBufferDescriptor", "DepthModelDescriptor", "DepthMetadata",
    "LFDDescriptor", "LFDModelDescriptor", "LFDMetadata",
    ## Mesh operations
    "Mesh", "MeshLoader", "MeshNormalizer", "Renderer", "ViewGenerator", "CameraPose",
    ## Similarity
    "SimilarityEngine", "SimilarityResult",
]