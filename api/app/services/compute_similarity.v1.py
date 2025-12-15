
import numpy as np, json, time, os, cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Union
from scipy.spatial.distance import euclidean, cosine
from pathlib import Path
from ultralytics.models.yolo import YOLO
from feature_extraction import (
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
    HSVHistogramExtractor(h_bins=4, sv_bins=4),  # 4×4×4 = 64 dims (was 512!)
    DominantColorsExtractor(n_colors=3),
]

FEATURE_SERVICE = FeatureExtractionService(EXTRACTORS)
SIMILARITY_SERVICE = SimilarityComputer()


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


def load_aggregated_features() -> Dict[str, Any]:
    """
    Load precomputed per-image Object-level features from JSON files.
    
    Returns:
        Dictionary mapping image stem to their features
    """
    
    if not FEATURES_ROOT.exists():
        raise FileNotFoundError(f"Features directory not found: {FEATURES_ROOT}")
    
    # Load all JSON files from the features directory except the query image
    aggregated = {}
    json_files = [file_path for file_path in FEATURES_ROOT.glob("*.json")]
    
    if not json_files:
        raise FileNotFoundError(f"No JSON feature files found in: {FEATURES_ROOT}")
    
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
        query_image: np.ndarray,
        query_metadata: Dict,
        top_k: int = 10,
        metric: str = "cosine",
        categories: Optional[List[str]] = None,
        class_filter:bool=True
    ) -> List[Tuple[str, float]]:
    
    if categories is None:
        categories = ['form', 'texture', 'color']
    
    query_features = extract_query_features(query_image)
    if len(query_features) == 0:
        raise ValueError("No features extracted from query image.")

    # Load database features
    base_features = load_aggregated_features()
    similarities = []

    # Strategy 1 : use class filtering
    similarities = SIMILARITY_SERVICE.compute_with_class_filter(
        query_features=query_features, 
        query_class=query_metadata["class_name"], 
        base_features=base_features, 
        metric=metric,
        same_class_only=class_filter)

    # Strategy 2 : Compute with all objects per image
    # for base_image_path, data in base_features.items():
    #     if data.get("num_objects", 0) == 0:
    #         print(f"    skipping '{base_image_path}', No objects detected.")
    #         continue
    #     objects = data["objects"]
    #     if isinstance(objects, np.ndarray):
    #         objects = objects.tolist() if objects.ndim > 0 else [objects.item()]

    #     # Option 1 : Use the object with the highest confidence score
    #     obj = max(objects, key=lambda x: x["confidence"])
    #     object_features = obj["features"]
    #     object_similarity = SIMILARITY_SERVICE.compute(query_features, object_features, metric="euclidean")
    #     similarities.append((base_image_path, object_similarity))  

    #     # # Option 2 : Compute similarity for each object in the image
    #     # max_similarity = 0.0
    #     # for obj in objects:
    #     #     try:
    #     #         object_features = obj["features"]
    #     #         object_similarity = SIMILARITY_SERVICE.compute(query_features, object_features, metric="cosine")
    #     #         max_similarity = max(max_similarity, object_similarity)
    #     #     except (KeyError, ValueError) as e:
    #     #         # Skip if features are malformed or dimensions don't match
    #     #         print(f"Warning: Skipping object in {base_image_path}: {e}")
    #     #         continue
    #     # if max_similarity > 0:
    #     #     similarities.append((base_image_path, max_similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

    # O(N log k) instead of O(N log N), more efficient for large databases.
    # import heapq 
    # return heapq.nlargest(top_k, similarities, key=lambda x: x[1])


def render_topk_images(
        query_image: np.ndarray,
        topk_image_names: List[str],
        query_title: str = "Query Image"
    ):
    """
    Render a figure containing the query image (as np.ndarray, BGR or RGB) and top-k similar images.
    """

    print(f"Query image shape: {query_image.shape}")

    num_images = len(topk_image_names)
    num_cols = min(num_images + 1, 6)
    num_rows = (num_images + 1 + num_cols - 1) // num_cols
    
    fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols, figure=fig, hspace=0.3, wspace=0.2)
    
    # Query image
    ax0 = fig.add_subplot(gs[0, 0])
    img = query_image
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[..., ::-1]  # BGR to RGB if needed

    ax0.imshow(img)
    ax0.axis('off')
    ax0.set_title(query_title, fontsize=14, fontweight='bold', color='blue', pad=10)
    for spine in ax0.spines.values():
        spine.set_edgecolor('blue')
        spine.set_linewidth(3)
        spine.set_visible(True)
    
    # Top-k images
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

    # the entire image loading and cropping should happens at the user request

    model = YOLO("api/models/yolo/best.pt")

    # query_img_path = DATA_ROOT / "train/n02958343_3003.jpeg"
    # query_img_path = DATA_ROOT / "train/n02084071_1.jpeg"
    # query_img_path = DATA_ROOT / "train/n00007846_6247.JPEG" 
    query_img_path = DATA_ROOT / "train/n02124075_428.JPEG" 
    image = cv2.imread(str(query_img_path))

    if image is None:
        raise ValueError(f"Could not load image: {query_img_path}")

    results = model.predict(image, verbose=False)[0]

    # Extract bounding boxes and crop objects
    print(f"Number of detected objects: {len(results.boxes)}")

    if len(results.boxes) == 0:
        raise ValueError("No objects detected in query image")

    crops = []
    crop_metadata = []  # Store bbox, confidence, class info

    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = results.names[cls]
        
        print(f"Object {idx}: ({x1},{y1}) -> ({x2},{y2}) | "
            f"Class: {class_name} | Confidence: {conf:.3f}")
        
        # Crop object
        crop = image[y1:y2, x1:x2]
        
        # Skip tiny crops
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            print(f"  -> Skipped (too small)")
            continue
        print(f"Crop shape right after slicing: {crop.shape}")  # Should be (H, W, 3)
        
        crops.append(crop)
        crop_metadata.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'class_id': cls,
            'class_name': class_name
        })
    
    if len(crops) == 0:
        raise ValueError("No valid objects after filtering")

    # combine crops and crop metadata into a single list of dictionaries
    query_objects = []
    for obj_id, crop in enumerate(crops):
        try:
            query_objects.append({
                'object_id': obj_id,
                'metadata': crop_metadata[obj_id]
            })
        except Exception as e:
            print(f"  Combining crops and crop_metadata failed - {e}")
            continue

    if not query_objects:
        raise ValueError("Failed to combine crops and crop_metadata!")

    # Select query object (multiple strategies)
    print("\nSelecting query object...")

    # # Strategy 1: Most confident object
    # query_obj = max(query_objects, key=lambda x: x['metadata']['confidence'])
    # print(f"Using most confident object: ID={query_obj['object_id']}, "
    #     f"Class={query_obj['metadata']['class_name']}, "
    #     f"Conf={query_obj['metadata']['confidence']:.3f}")

    # Strategy 2: Largest object
    # query_obj = max(query_objects, key=lambda x: 
    #     (x['metadata']['bbox'][2] - x['metadata']['bbox'][0]) * 
    #     (x['metadata']['bbox'][3] - x['metadata']['bbox'][1]))

    # Strategy 3: User selection
    print("Available objects:")
    for obj in query_objects:
        print(f"  {obj['object_id']}: {obj['metadata']['class_name']} "
              f"(conf: {obj['metadata']['confidence']:.3f})")
    selected_id = int(input("Select object ID: "))
    query_obj = next(obj for obj in query_objects if obj['object_id'] == selected_id)
    selected_crop = crops[query_obj["object_id"]]

    # calculate the query image's features
    query_features = FEATURE_SERVICE.extract(selected_crop)
    query_class = query_obj['metadata']['class_name']
    print(f"\n=== Query Object Selected ===")
    print(f"Class: {query_class}")
    print(f"Confidence: {query_obj['metadata']['confidence']:.3f}")
    print(f"BBox: {query_obj['metadata']['bbox']}")

    # Verify feature structure
    print(f"\nFeature categories: {list(query_features.keys())}")
    for cat in query_features.keys():
        if 'combined' in query_features[cat]:
            dim = len(query_features[cat]['combined'])
            print(f"  {cat}: {dim} dimensions")

    # Now ready for similarity computation
    print("\n   Query features ready for similarity search!")


    print(f"\n=== Searching similar images for: {query_img_path.name} ===")
    results = search_similar_images(
        selected_crop,
        query_obj["metadata"], 
        top_k=10, 
        metric="cosine", 
        class_filter=True
        )

    print(f"search finished with {len(results)} results")
    print(f"\nTop 10 similar images:")
    for idx, (img_name, similarity) in enumerate(results, 1):
        print(f"{idx:2d}. {img_name:<60} (similarity: {similarity:.4f})")

    topk_image_names = [img_name for img_name, _, in results]
    render_topk_images(selected_crop, topk_image_names)


if __name__ == "__main__":
    
    assert DATA_ROOT.exists(), f"DATA_ROOT directory not found"
    assert FEATURES_ROOT.exists(), f"FEATURE_ROOT directory not found"
    assert FEATURE_SERVICE is not None, "FEATURE_SERVICE is not initialized"
    assert SIMILARITY_SERVICE is not None, "SIMILARITY_SERVICE is not initialized"

    main()



