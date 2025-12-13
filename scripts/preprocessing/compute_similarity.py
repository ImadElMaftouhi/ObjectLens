# Similarity functions for feature vectors
# given a query image and a database of images with precomputed features, compute similarity scores and return ranked list of similar images
# logic flow : load image, extract features, compute similarity with database features, rank results


import numpy as np, json, time
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Union
from scipy.spatial.distance import euclidean, cosine
from pathlib import Path
from api.app.services.feature_extraction import (
    FourierDescriptorExtractor,
    OrientationHistogramExtractor,
    TamuraExtractor,
    GaborExtractor,
    HSVHistogramExtractor,
    DominantColorsExtractor,
    FeatureExtractionService,
    SimilarityComputer,
)

# -------- Config ----------
DATA_ROOT = Path("imagenet_yolo15/images")
FEATURES_ROOT = Path("features/all")

# Initialize feature extractors and service
EXTRACTORS = [
    FourierDescriptorExtractor(n_coeff=40),
    OrientationHistogramExtractor(bins=36),
    TamuraExtractor(kmax=4, n_bins=16),
    GaborExtractor(n_scales=3, n_orientations=4),
    HSVHistogramExtractor(bins=256),
    DominantColorsExtractor(n_colors=5),
]

FEATURE_SERVICE = FeatureExtractionService(EXTRACTORS)
SIMILARITY_COMPUTER = SimilarityComputer()


def _to_numpy(obj: Any) -> Any:
    """Recursively convert lists back to numpy arrays where needed."""
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        # Try to convert to numpy array, but keep as list if it's nested differently
        try:
            return np.array(obj)
        except (ValueError, TypeError):
            return [_to_numpy(v) for v in obj]
    return obj


def load_aggregated_features(query_image_path: str | Path) -> Dict[str, Any]:
    """
    Load precomputed per-image Object-level features from JSON files.
    
    Returns:
        Dictionary mapping image stem to their features
    """
    features_dir = FEATURES_ROOT
    
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    # Load all JSON files from the features directory except the query image
    aggregated = {}
    json_files = [file_path for file_path in features_dir.glob("*.json") if file_path != Path(query_image_path).name]
    
    if not json_files:
        raise FileNotFoundError(f"No JSON feature files found in: {features_dir}")
    
    start_time = time.time()
    for json_file in tqdm(json_files, desc="Loading features", unit="file"):
        with open(json_file, "r", encoding="utf-8") as f:
            features = json.load(f)
            aggregated[json_file.stem] = _to_numpy(features)
    end_time = time.time() - start_time
    print(f"Loaded {len(aggregated)} features in {end_time:.2f} seconds.")
    return aggregated


def extract_query_features(query_image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
    """
    Extract features from a query image.
    Args:
        query_image_path: Path to the query image
    Returns:
        Dictionary of extracted features
    """
    return FEATURE_SERVICE.extract(query_image)


def search_similar_images(
    query_image_path: str | Path,
    top_k: int = 10,
    distance_metric: str = "cosine",
    categories: Optional[List[str]] = None
) -> List[Tuple[str, float]]:
    """
    Find similar images from a database given a query image.
    
    Args:
        query_image_path: Path to the query image
        top_k: Number of top results to return
        distance_metric: 'cosine' or 'euclidean'
        categories: Feature categories to use ('form', 'texture', 'color')
    
    Returns:
        List of (image_path, similarity_score) tuples, sorted by similarity (highest first)
    """
    if categories is None:
        categories = ['form', 'texture', 'color']
    
    query_features = extract_query_features(query_image_path)
    if len(query_features) == 0:
        raise ValueError("No features extracted from query image.")

    # Load database features
    database_features = load_aggregated_features(query_image_path)
    # key = next(iter(database_features.keys()), None)
    # if key is not None:
    #     print(f"{key}: {database_features[key]}")
    
    print(f"Computing similarity against {len(database_features)} images...")
    similarities = []
    
    # For each database image, compute similarity
    for image_path, db_features in database_features.items():
        try:
            # print(f"Computing similarity for {image_path}")
            similarity = SIMILARITY_COMPUTER.compute(query_features, db_features)
            similarities.append((image_path, similarity))
        
        except Exception as e:
            # Skip images that fail similarity computation
            continue
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k
    return similarities[:top_k]


def render_image(query_path:str=""):
    # visualise the query image
    import matplotlib.pyplot as plt
    img = plt.imread(str(query_path))
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def render_topk_images(query_path: str, topk_image_names: List[str]):
    """
    Render a figure containing the query image and top-k similar images.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    # Set up figure with better spacing
    num_cols = min(len(topk_image_names) + 1, 6)  # Max 6 images per row
    num_rows = (len(topk_image_names) + 1 + num_cols - 1) // num_cols
    
    fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols, figure=fig, hspace=0.3, wspace=0.2)
    
    # Render query image with highlighted border
    query_image_path = DATA_ROOT / query_path
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(plt.imread(str(query_image_path)))
    ax0.axis('off')
    ax0.set_title('Query Image', fontsize=14, fontweight='bold', color='blue', pad=10)
    for spine in ax0.spines.values():
        spine.set_edgecolor('blue')
        spine.set_linewidth(3)
        spine.set_visible(True)
    
    # Render top-k similar images
    for i, image_name in enumerate(topk_image_names):
        image_path = next((p for p in DATA_ROOT.rglob(f"**/{image_name}.*") if p.is_file()), None)
        if image_path is None:
            print(f"Image not found: {image_name}")
            continue
        
        row = (i + 1) // num_cols
        col = (i + 1) % num_cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(plt.imread(str(image_path)))
        ax.axis('off')
        ax.set_title(f"#{i + 1}: {image_name}", fontsize=12, pad=8)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Example usage: search for similar images given a query image.
    """
    query_image = DATA_ROOT / "train/n02958343_3003.jpeg"
    
    print(f"\n=== Searching similar images for: {Path(query_image).name} ===")
    results = search_similar_images(query_image,top_k=10,distance_metric="cosine")
    print(f"search finished with {len(results)} results")

    print(f"\nTop 10 similar images:")
    for idx, (image_path, similarity) in enumerate(results, 1):
        print(f"{idx:2d}. {image_path:<60} (similarity: {similarity:.4f})")

    topk_image_names = [Path(image_path).name for image_path, _ in results]
    render_topk_images(str(query_image.relative_to(DATA_ROOT)), topk_image_names)


if __name__ == "__main__":
    main()

