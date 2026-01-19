import os
import sys
import cv2
import json
import time
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from ultralytics.models.yolo import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.services.feature_extraction import (
    FourierDescriptorExtractor, OrientationHistogramExtractor,
    TamuraExtractor, GaborExtractor,
    HSVHistogramExtractor, DominantColorsExtractor,
    FeatureExtractionService,
)

# -------- Config ----------
DATA_ROOT = Path("data/imagenet_4_yolo/images")
OUT_ROOT = Path("data/features")
FAISS_OUTPUT_DIR = Path("data/index/faiss") # FAISS Index and vectors for fast retrieval
METADATA_OUTPUT_DIR = Path("data/index/metadata") # Object metadata for MongoDB
FEATURES_DIR = Path("data/features/all") # All extracted features (*.json)
LABELS_ROOT = Path("data/imagenet_4_yolo/labels")
MODEL_PATH = Path("backend/models/yolo/best.pt")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".JPEG", ".JPG"}

EXTRACTORS = [
    FourierDescriptorExtractor(n_coeff=40),
    OrientationHistogramExtractor(bins=36),
    TamuraExtractor(kmax=4, n_bins=16),
    GaborExtractor(n_scales=3, n_orientations=4),
    HSVHistogramExtractor(h_bins=4, sv_bins=4),  # 4×4×4 = 64 dims
    DominantColorsExtractor(n_colors=3),
]

FEATURE_SERVICE = FeatureExtractionService(EXTRACTORS)

# Global model - will be initialized once
MODEL = None

# Logging setup - ensure all directories exist
OUT_ROOT.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
FAISS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUT_ROOT / 'processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------- Helper Functions ----------

def get_model():
    """Lazy load model once per process."""
    global MODEL
    if MODEL is None:
        MODEL = YOLO(MODEL_PATH)
        MODEL.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)  # Warmup
    return MODEL


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON dumping."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "dtype") and (hasattr(obj, "item") or isinstance(obj, (np.generic,))):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def extract_object_features(image_path: Path) -> List[Dict[str, Any]]:
    """Extract features from all objects detected in an image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    if MODEL is None:
        raise ValueError("Model not initialized. Call get_model() first.")
    
    results = MODEL.predict(img, verbose=False)[0]

    objects = []
    for i, box in enumerate(results.boxes): # type: ignore
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = results.names[cls]

        # Crop object
        crop = img[y1:y2, x1:x2]
        
        # Skip if crop is too small
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        features = FEATURE_SERVICE.extract(crop)

        # Weighted final_vector
        final_vector = np.concatenate([
            0.5 * features['form']['combined'],
            0.3 *features['texture']['combined'],
            0.2 * features['color']['combined'],
        ])
        norm = np.linalg.norm(final_vector)
        if norm > 0:
            final_vector /= norm

        objects.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": cls,
            "class_name": class_name,
            "confidence": conf,
            "features": features,
            "final_vector": final_vector.tolist()
        })

    return objects


def process_single_image(img_path: Path, data_root: Path, out_root: Path) -> Dict[str, Any]:
    """Process a single image and return result statistics."""

    try:
        stem = img_path.stem
        out_file = out_root / f"{stem}.json"
        
        # Skip if already processed
        if out_file.exists():
            return {"status": "skipped", "path": str(img_path)}

        objects = extract_object_features(img_path)

        result = {
            "image_path": str(img_path.relative_to(data_root)),
            "objects": objects,
            "processed_at": datetime.now().isoformat(),
            "num_objects": len(objects)
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(result), f, indent=2)

        return {
            "status": "success",
            "path": str(img_path),
            "num_objects": len(objects)
        }

    except Exception as e:
        logger.error(f"Error processing {img_path}: {str(e)}")
        return {
            "status": "error",
            "path": str(img_path),
            "error": str(e)
        }


def verify_output_sample(out_root: Path) -> bool:
    """Verify a sample output file for correctness."""
    json_files = list(out_root.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found to verify")
        return False
    
    sample = json_files[0]
    logger.info(f"Verifying sample: {sample.name}")
    
    try:
        with open(sample) as f:
            data = json.load(f)
        
        assert "image_path" in data
        assert "objects" in data
        
        if data["objects"]:
            obj = data["objects"][0]
            vec = np.array(obj["final_vector"])
            norm = np.linalg.norm(vec)
            
            logger.info(f"  Vector dimension: {len(vec)}")
            logger.info(f"  Vector norm: {norm:.6f}")
            logger.info(f"  Num objects: {len(data['objects'])}")
            
            assert 0.99 < norm < 1.01, f"Vector not normalized: {norm}"
            assert not np.isnan(vec).any(), "NaN in vector"
            assert not np.isinf(vec).any(), "Inf in vector"
            
        logger.info("Output format validated")
        return True
        
    except Exception as e:
        logger.error(f" Verification failed: {e}")
        return False


def print_summary(results: List[Dict], elapsed: float):
    """Print processing summary with statistics."""
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    total_objects = sum(r.get("num_objects", 0) for r in results if r["status"] == "success")
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"    Total images:      {len(results)}")
    print(f"    Successful:     {success}")
    print(f"    Skipped:        {skipped}")
    print(f"    Errors:         {errors}")
    print(f"    Total objects:  {total_objects}")
    print(f"    Time elapsed:   {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"    Avg time/img:   {elapsed/len(results):.2f}s")
    print("="*60 + "\n")


def build_faiss_index(features_root: Path, faiss_output_dir: Path, metadata_output_dir: Path):
    """
    Collect all vectors from JSON files and build FAISS index.
    
    Creates:
    - vectors.npy: All feature vectors (N x D numpy array)
    - index.faiss: FAISS index for fast similarity search
    - ids.npy: FAISS ID -> MongoDB _id mapping
    - metadata.json: Index metadata (dimension, metric, etc.)
    - object_mapping.json: Full object metadata mapping for MongoDB
    """
    try:
        import faiss
    except ImportError:
        logger.error("FAISS not installed. Install with: pip install faiss-cpu (or faiss-gpu)")
        return False
    
    logger.info("Building FAISS index from processed features...")
    
    # Ensure output directories exist
    faiss_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_vectors = []
    all_ids = []  # MongoDB document IDs (composite: image_path__object_idx)
    object_metadata = []  # Full metadata for each object
    
    # Collect vectors from all JSON files
    json_files = list(features_root.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found to build FAISS index")
        return False
    
    logger.info(f"Collecting vectors from {len(json_files)} JSON files...")
    
    for json_file in tqdm(json_files, desc="Collecting vectors", unit="file"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_path = data.get("image_path", "")
            if not image_path:
                continue
            
            for obj_idx, obj in enumerate(data.get("objects", [])):
                if "final_vector" not in obj:
                    continue
                
                vector = np.array(obj["final_vector"], dtype=np.float32)
                faiss_id = len(all_vectors)  # Sequential FAISS ID
                
                # Create composite ID: image_path__object_idx
                object_id = f"{image_path}__{obj_idx}"
                
                all_vectors.append(vector)
                all_ids.append(object_id)
                
                # Store full metadata for MongoDB
                object_metadata.append({
                    "faiss_id": int(faiss_id),
                    "image_path": image_path,
                    "object_idx": int(obj_idx),
                    "bbox": obj.get("bbox", []),
                    "class_id": int(obj.get("class_id", -1)),
                    "class_name": obj.get("class_name", "unknown"),
                    "confidence": float(obj.get("confidence", 0.0))
                })
                
        except Exception as e:
            logger.warning(f"Error reading {json_file.name}: {e}")
            continue
    
    if not all_vectors:
        logger.error("No vectors collected. Cannot build FAISS index.")
        return False
    
    # Convert to numpy arrays
    logger.info(f"Converting {len(all_vectors)} vectors to numpy array...")
    vectors = np.vstack(all_vectors).astype(np.float32)
    ids_array = np.array(all_ids)
    
    dimension = vectors.shape[1]
    num_vectors = vectors.shape[0]
    
    logger.info(f"Vector shape: {vectors.shape} (N={num_vectors}, D={dimension})")
    
    # Verify vectors are normalized
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=0.01):
        logger.warning(f"Vectors may not be properly normalized. Norm range: [{norms.min():.4f}, {norms.max():.4f}]")
    
    # Save raw vectors
    vectors_path = faiss_output_dir / "vectors.npy"
    logger.info(f"Saving vectors to {vectors_path}")
    np.save(vectors_path, vectors)
    
    # Save ID mapping
    ids_path = faiss_output_dir / "ids.npy"
    logger.info(f"Saving IDs to {ids_path}")
    np.save(ids_path, ids_array)
    
    # Build FAISS index (using Inner Product for cosine similarity since vectors are normalized)
    logger.info("Building FAISS index (IndexFlatIP for cosine similarity)...")
    index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine similarity for normalized vectors
    
    # Add vectors to index
    index.add(vectors)
    
    # Save FAISS index
    index_path = faiss_output_dir / "index.faiss"
    logger.info(f"Saving FAISS index to {index_path}")
    faiss.write_index(index, str(index_path))
    
    # Save index metadata
    metadata_info = {
        "num_vectors": int(num_vectors),
        "dimension": int(dimension),
        "metric": "cosine",  # Using Inner Product for normalized vectors
        "index_type": "IndexFlatIP",
        "created_at": datetime.now().isoformat(),
        "vectors_file": "vectors.npy",
        "ids_file": "ids.npy",
        "index_file": "index.faiss"
    }
    
    metadata_path = faiss_output_dir / "metadata.json"
    logger.info(f"Saving index metadata to {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_info, f, indent=2)
    
    # Save object mapping (for MongoDB lookup)
    object_mapping_path = metadata_output_dir / "object_mapping.json"
    logger.info(f"Saving object mapping to {object_mapping_path}")
    with open(object_mapping_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(object_metadata), f, indent=2)
    
    logger.info("✅ FAISS index built successfully!")
    logger.info(f"   - Vectors: {num_vectors} x {dimension}")
    logger.info(f"   - Index file: {index_path}")
    logger.info(f"   - Vectors file: {vectors_path}")
    logger.info(f"   - IDs file: {ids_path}")
    logger.info(f"   - Metadata file: {metadata_path}")
    logger.info(f"   - Object mapping: {object_mapping_path}")
    
    return True


def main(max_workers: int = 4, verify_first_batch: bool = True):
    """
    Main processing function with parallel execution and progress tracking.
    
    Args:
        max_workers: Number of parallel workers (use 1 for sequential, 2-4 recommended for I/O-bound tasks)
        verify_first_batch: Verify output format after first batch
    
    Note: For YOLO model, use max_workers=1 to avoid model loading issues, or use ProcessPoolExecutor with proper model initialization in each process.
    """
    # Ensure all output directories exist
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Warm up model before parallel processing
    logger.info("Initializing YOLO model...")
    get_model()
    logger.info("Model ready!")
    
    logger.info(f"Starting feature extraction pipeline")
    logger.info(f"Output directory: {FEATURES_DIR}")
    logger.info(f"Workers: {max_workers}")

    for folder in ["train", "val"]:
        folder_path = DATA_ROOT / folder
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            continue
            
        images = [p for p in folder_path.iterdir() if p.suffix in EXTS]
        
        if not images:
            logger.warning(f"No images found in {folder}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing folder: {folder} ({len(images)} images)")
        logger.info(f"{'='*60}")

        start_time = time.time()
        results = []

        # Process with progress bar
        if max_workers == 1:
            # Sequential processing
            for img_path in tqdm(images, desc=f"{folder}", unit="img"):
                result = process_single_image(img_path, DATA_ROOT, FEATURES_DIR)
                results.append(result)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_single_image, img_path, DATA_ROOT, FEATURES_DIR): img_path
                    for img_path in images
                }
                
                for future in tqdm(as_completed(futures), total=len(images), desc=f"{folder}", unit="img"):
                    result = future.result()
                    results.append(result)

        elapsed = time.time() - start_time
        
        # Verify after first batch
        if verify_first_batch and folder == "train":
            if not verify_output_sample(FEATURES_DIR):
                logger.error("Verification failed! Stopping execution.")
                return
            verify_first_batch = False  # Only verify once

        # Print summary
        print_summary(results, elapsed)
        
        # Log errors
        errors = [r for r in results if r["status"] == "error"]
        if errors:
            logger.warning(f"\nErrors occurred in {len(errors)} images:")
            for err in errors[:5]:  # Show first 5
                logger.warning(f"  {err['path']}: {err.get('error', 'Unknown')}")
            if len(errors) > 5:
                logger.warning(f"  ... and {len(errors) - 5} more")

    logger.info("\n" + "="*60)
    logger.info("JSON feature extraction complete!")
    logger.info("="*60)
    
    # Build FAISS index from all processed JSON files
    logger.info("\nBuilding FAISS index...")
    if build_faiss_index(FEATURES_DIR, FAISS_OUTPUT_DIR, METADATA_OUTPUT_DIR):
        logger.info("\n✅ All processing complete! FAISS index is ready for fast retrieval.")
    else:
        logger.warning("\n⚠️  Processing complete, but FAISS index build failed or skipped.")
    
    logger.info("\n" + "="*60)
    logger.info("OUTPUT SUMMARY")
    logger.info("="*60)
    logger.info(f"JSON features:     {FEATURES_DIR}")
    logger.info(f"FAISS index:       {FAISS_OUTPUT_DIR}")
    logger.info(f"Metadata mapping:  {METADATA_OUTPUT_DIR}")
    logger.info("="*60 + "\n")


def test_single_image():
    """Test feature extraction on a single image with detailed output."""
    print("\n" + "="*60)
    print("TESTING SINGLE IMAGE EXTRACTION")
    print("="*60 + "\n")


    # Find first available test image
    test_image = None
    for folder in ["train", "val"]:
        folder_path = DATA_ROOT / folder
        if folder_path.exists():
            images = [p for p in folder_path.iterdir() if p.suffix in EXTS]
            if images:
                test_image = images[0]
                break
    
    if not test_image:
        logger.error("No test image found!")
        return
    
    logger.info(f"Test image: {test_image.name}")
    
    try:
        # Time the extraction
        start = time.time()
        objects = extract_object_features(test_image)
        elapsed = time.time() - start
        
        print(f"\n Extraction successful!")
        print(f" Time: {elapsed:.3f}s")
        print(f" Objects detected: {len(objects)}")
        
        if objects:
            obj = objects[0]
            print(f"\n--- First Object Details ---")
            print(f"Class: {obj['class_name']}")
            print(f"Confidence: {obj['confidence']:.3f}")
            print(f"BBox: {obj['bbox']}")
            
            # Check vector
            vec = np.array(obj['final_vector'])
            print(f"\n--- Feature Vector ---")
            print(f"Dimension: {len(vec)}")
            print(f"Norm: {np.linalg.norm(vec):.6f}")
            print(f"Min/Max: [{vec.min():.4f}, {vec.max():.4f}]")
            print(f"Mean/Std: {vec.mean():.4f} ± {vec.std():.4f}")
            
            # Check individual categories
            features = obj['features']
            print(f"\n--- Category Dimensions ---")
            for cat in ['form', 'texture', 'color']:
                if cat in features and 'combined' in features[cat]:
                    dim = len(features[cat]['combined'])
                    print(f"{cat.capitalize():8s}: {dim:4d} dims")
            
            # Validate
            norm = np.linalg.norm(vec)
            has_nan = np.isnan(vec).any()
            has_inf = np.isinf(vec).any()
            
            print(f"\n--- Validation ---")
            print(f"Normalized (0.99-1.01): {'(+)' if 0.99 < norm < 1.01 else '(-)'} ({norm:.6f})")
            print(f"No NaN values:          {'(+)' if not has_nan else '(-)'}")
            print(f"No Inf values:          {'(+)' if not has_inf else '(-)'}")
            
            # Save test output
            test_output = OUT_ROOT / "test_output.json"
            OUT_ROOT.mkdir(parents=True, exist_ok=True)
            FEATURES_DIR.mkdir(parents=True, exist_ok=True)
            
            result = {
                "image_path": str(test_image.relative_to(DATA_ROOT)),
                "objects": objects,
                "processed_at": datetime.now().isoformat(),
                "num_objects": len(objects)
            }
            
            with open(test_output, "w", encoding="utf-8") as f:
                json.dump(_to_serializable(result), f, indent=2)
            
            print(f"\n  Test output saved: {test_output}")
            
        else:
            logger.warning("No objects detected in test image")
        
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)


def test_batch_sample(n_images: int = 5):
    """Test extraction on a small batch to verify consistency."""
    print("\n" + "="*60)
    print(f"TESTING BATCH EXTRACTION ({n_images} images)")
    print("="*60 + "\n")
    
    # Get sample images
    images = []
    for folder in ["train", "val"]:
        folder_path = DATA_ROOT / folder
        if folder_path.exists():
            folder_images = [p for p in folder_path.iterdir() if p.suffix in EXTS]
            images.extend(folder_images[:n_images])
            if len(images) >= n_images:
                break

    images = images[:n_images]

    if not images:
        logger.error("No test images found!")
        return

    logger.info(f"Testing {len(images)} images...")

    results = []
    total_objects = 0
    start = time.time()

    for img in tqdm(images, desc="Testing", unit="img"):
        try:
            result = process_single_image(img, DATA_ROOT, FEATURES_DIR)
            results.append(result)
            if result["status"] == "success":
                total_objects += result.get("num_objects", 0)
        except Exception as e:
            logger.error(f"Failed on {img.name}: {e}")

    elapsed = time.time() - start

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"\n--- Batch Test Results ---")
    print(f"    Success: {success}/{len(images)}")
    print(f"    Errors:  {errors}/{len(images)}")
    print(f"    Total objects: {total_objects}")
    print(f"    Avg objects/image: {total_objects/success if success > 0 else 0:.1f}")
    print(f"    Time: {elapsed:.2f}s ({elapsed/len(images):.2f}s per image)")

    test_batch_output = OUT_ROOT / "test_batch_output.json"
    assert FEATURES_DIR.exists(), f"Error: FEATURES_DIR does not exist. create it first."
    batch_result = {
        "tested_images": [str(img.relative_to(DATA_ROOT)) for img in images],
        "results": results,
        "processed_at": datetime.now().isoformat(),
        "total_images": len(images),
        "success": success,
        "errors": errors,
        "total_objects": total_objects,
        "avg_objects_per_image": total_objects / success if success > 0 else 0.0,
        "elapsed_time_sec": elapsed
    }
    with open(test_batch_output, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(batch_result), f, indent=2)
    print(f"\n  Batch test output saved: {test_batch_output}")

    # Verify output
    if verify_output_sample(FEATURES_DIR):
        print("\n + All tests passed!")
    else:
        print("\n - Verification failed!")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    
    print(f"Current working directory : {os.getcwd()}")

    assert FourierDescriptorExtractor is not None, f"Error importing FourierDescriptorExtractor"
    assert OrientationHistogramExtractor is not None, f"Error importing OrientationHistogramExtractor"
    assert TamuraExtractor is not None, f"Error importing TamuraExtractor"
    assert GaborExtractor is not None, f"Error importing GaborExtractor"
    assert HSVHistogramExtractor is not None, f"Error importing HSVHistogramExtractor"
    assert DominantColorsExtractor is not None, f"Error importing DominantColorsExtractor"
    assert FeatureExtractionService is not None, f"Error importing FeatureExtractionService"

    assert DATA_ROOT is not None, f"Error importing DATA_ROOT"
    assert OUT_ROOT is not None, f"Error importing OUT_ROOT"
    assert FEATURES_DIR is not None, f"Error importing FEATURES_ROOT"
    assert LABELS_ROOT is not None, f"Error importing LABELS_ROOT"
    assert MODEL_PATH is not None, f"Error importing MODEL_PATH"
    assert EXTS is not None, f"Error importing EXTS"
    assert EXTRACTORS is not None, f"Error importing EXTRACTORS"
    assert FEATURE_SERVICE is not None, f"Error importing FEATURE_SERVICE"

    # Warm up model
    MODEL = get_model()
    logger.info("Model ready!")

    # 1. Test single image first (recommended)
    # test_single_image()
    
    # 2. Test small batch
    # test_batch_sample(n_images=5)
    
    # 3. Process full dataset
    main(max_workers=4, verify_first_batch=True)