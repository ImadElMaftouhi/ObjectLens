import json
import time
import os
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from ultralytics import YOLO # type: ignore

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
OUT_ROOT = Path("features/all")
LABELS_ROOT = Path("imagenet_yolo15/labels")
# MODEL_PATH = Path("api/models/yolo/yolov8n.pt")
MODEL_PATH = Path("yolo/model/weights/best.pt")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".JPEG", ".JPG"}

EXTRACTORS = [
        FourierDescriptorExtractor(n_coeff=40),
        OrientationHistogramExtractor(bins=36),
        TamuraExtractor(kmax=4, n_bins=16),
        GaborExtractor(n_scales=3, n_orientations=4),
        HSVHistogramExtractor(bins=32),
        DominantColorsExtractor(n_colors=3),
    ]

FEATURE_SERVICE = FeatureExtractionService(EXTRACTORS)

MODEL = YOLO(MODEL_PATH)

# testing the execution time on a single image
def test_run():
    img_path = DATA_ROOT / "train/n00007846_13214.jpeg"
    start_time = time.time()
    result = FEATURE_SERVICE.extract(img_path)
    processing_time = time.time() - start_time
    print(processing_time)
    print(f"result : \n{result}")

def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON dumping."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalar types
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
    img = cv2.imread(str(image_path))
    results = MODEL.predict(img)[0]

    objects = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = results.names[cls]

        # Crop with 12% margin
        h, w = y2 - y1, x2 - x1
        margin_h = int(h * 0.12)
        margin_w = int(w * 0.12)
        crop = img[max(0, y1 - margin_h):min(img.shape[0], y2 + margin_h),
                   max(0, x1 - margin_w):min(img.shape[1], x2 + margin_w)]

        features = FEATURE_SERVICE.extract(crop)

        # Weighted final_vector
        final_vector = np.concatenate([
            0.6 * features['form']['combined'],
            0.3 * features['texture']['combined'],
            0.1 * features['color']['combined'],
        ])
        norm = np.linalg.norm(final_vector)
        if norm > 0:
            final_vector /= norm

        objects.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": cls,
            "class_name": class_name,
            "confidence": conf,
            "features": features,  # Full category/extractor breakdown
            "final_vector": final_vector.tolist()  # L2-normalized
        })

    return objects

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for folder in ["train", "val"]:
        folder_path = DATA_ROOT / folder
        images = [p for p in folder_path.iterdir() if p.suffix in EXTS]

        start_time = time.time()

        for idx, img_path in enumerate(images, 1):
            stem = img_path.stem
            out_file = OUT_ROOT / f"{stem}.json"

            objects = extract_object_features(img_path)

            result = {
                "image_path": str(img_path.relative_to(DATA_ROOT)),
                "objects": objects
            }

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(_to_serializable(result), f, indent=4)

            if idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {idx}/{len(images)} in {folder} ({elapsed:.2f}s)")

        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        print(f"Folder {folder} done. Time: {mins}m {secs}s")


def test():
    img_path = DATA_ROOT / "train" / "n00007846_6247.JPEG"
    # img_path = DATA_ROOT / "train" / "n02124075_428.JPEG"

    start_time = time.time()

    stem = img_path.stem
    out_file = f"{stem}.json"

    objects = extract_object_features(img_path)

    result = {
        "image_path": str(img_path.relative_to(DATA_ROOT)),
        "objects": objects
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(result), f, indent=4)

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)

if __name__=="__main__":
    # test()
    main()