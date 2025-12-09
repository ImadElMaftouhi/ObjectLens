import json
import time
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any


from feature_extraction import (
    FourierDescriptorExtractor, OrientationHistogramExtractor,
    TamuraExtractor, GaborExtractor,
    HSVHistogramExtractor, DominantColorsExtractor,
    FeatureExtractionService,
)

# -------- Config ----------
DATA_ROOT = Path("imagenet_yolo15/images")
OUT_ROOT = Path("features/all")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".JPEG", ".JPG"}

EXTRACTORS = [
        FourierDescriptorExtractor(n_coeff=40),
        OrientationHistogramExtractor(bins=36),
        TamuraExtractor(kmax=4, n_bins=16),
        GaborExtractor(n_scales=3, n_orientations=4),
        HSVHistogramExtractor(bins=256),
        DominantColorsExtractor(n_colors=5),
    ]

FEATURE_SERVICE = FeatureExtractionService(EXTRACTORS)


# testing the execution time on a single image
def test_script():
    """
    there is 5992 images in total between val and train
    0.1365 is the average execution time for a single image
    In theory, 5992 * 0.1365 ~ 818ms is the global execution time for all images
    """
    img_path = DATA_ROOT / "train/n00007846_13214.jpeg"
    start_time = time.time()
    result = FEATURE_SERVICE.extract(img_path)
    processing_time = time.time() - start_time
    print(processing_time)
    print(f"result : \n{result}")

    

def main():
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

    for folder in ["train", "val"]:
        folder_path = DATA_ROOT / folder

        # ensure output dir exists
        if not OUT_ROOT.exists():
            OUT_ROOT.mkdir(parents=True, exist_ok=True)

        images = [p for p in folder_path.rglob("*") if p.suffix in EXTS and p.is_file()]

        start_time = time.time()

        aggregated: Dict[str, Any] = {}

        for idx, img_path in enumerate(images, start=1):
            key = str(img_path.relative_to(DATA_ROOT))

            result = FEATURE_SERVICE.extract(img_path)
            # store per-image file (kept for compatibility)
            stem = img_path.stem
            out_file = OUT_ROOT / f"{stem}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(_to_serializable(result), f, indent=4)

            # add to aggregated dict
            aggregated[key] = _to_serializable(result)

            if idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {idx}/{len(images)} images in {elapsed:.2f} seconds")

        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        print(f"Done processing folder {folder}. Total time: {mins:.2f} mins and {secs:.2f} seconds")


if __name__=="__main__":
    # test_script()
    main()