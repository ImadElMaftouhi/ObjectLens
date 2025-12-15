import json
import time
import os
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from ultralytics.models.yolo import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from api.app.services.feature_extraction import (
    FourierDescriptorExtractor, OrientationHistogramExtractor,
    TamuraExtractor, GaborExtractor,
    HSVHistogramExtractor, DominantColorsExtractor,
    FeatureExtractionService,
)

# -------- Config ----------
DATA_ROOT = Path("imagenet_yolo15/images")
OUT_ROOT = Path("features")
FEATURES_ROOT = Path("features/all")
LABELS_ROOT = Path("imagenet_yolo15/labels")
MODEL_PATH = Path("api/models/yolo/best.pt")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".JPEG", ".JPG"}

EXTRACTORS = [
    FourierDescriptorExtractor(n_coeff=40),
    OrientationHistogramExtractor(bins=36),
    TamuraExtractor(kmax=4, n_bins=16),
    GaborExtractor(n_scales=3, n_orientations=4),
    HSVHistogramExtractor(h_bins=4, sv_bins=4),  # 4×4×4 = 64 dims (was 512!)
    DominantColorsExtractor(n_colors=3),
]

FEATURE_SERVICE = FeatureExtractionService(EXTRACTORS)

# Global model - will be initialized once
MODEL = None

def get_model():
    """Lazy load model once per process."""
    global MODEL
    if MODEL is None:
        MODEL = YOLO(MODEL_PATH)
        MODEL.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)  # Warmup
    return MODEL

# Logging setup
FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
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


def main(max_workers: int = 4, verify_first: bool = True):
    """
    Main processing function with parallel execution and progress tracking.
    
    Args:
        max_workers: Number of parallel workers (use 1 for sequential, 2-4 recommended for I/O-bound tasks)
        verify_first: Verify output format after first batch
    
    Note: For YOLO model, use max_workers=1 to avoid model loading issues, or use ProcessPoolExecutor with proper model initialization in each process.
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Warm up model before parallel processing
    logger.info("Initializing YOLO model...")
    get_model()
    logger.info("Model ready!")
    
    logger.info(f"Starting feature extraction pipeline")
    logger.info(f"Output directory: {FEATURES_ROOT}")
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
                result = process_single_image(img_path, DATA_ROOT, FEATURES_ROOT)
                results.append(result)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_single_image, img_path, DATA_ROOT, FEATURES_ROOT): img_path
                    for img_path in images
                }
                
                for future in tqdm(as_completed(futures), total=len(images), desc=f"{folder}", unit="img"):
                    result = future.result()
                    results.append(result)

        elapsed = time.time() - start_time
        
        # Verify after first batch
        if verify_first and folder == "train":
            if not verify_output_sample(FEATURES_ROOT):
                logger.error("Verification failed! Stopping execution.")
                return
            verify_first = False  # Only verify once

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

    logger.info("\n All processing complete!")


def test_single_image():
    """Test feature extraction on a single image with detailed output."""
    print("\n" + "="*60)
    print("TESTING SINGLE IMAGE EXTRACTION")
    print("="*60 + "\n")

    model = get_model()

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
            FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
            
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
    
    model = get_model()
    
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
            result = process_single_image(img, DATA_ROOT, FEATURES_ROOT)
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
    
    # Verify output
    if verify_output_sample(FEATURES_ROOT):
        print("\n + All tests passed!")
    else:
        print("\n - Verification failed!")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    
    # 1. Test single image first (recommended)
    # test_single_image()
    
    # 2. Test small batch
    # test_batch_sample(n_images=5)
    
    # 3. Process full dataset
    main(max_workers=6, verify_first=True)