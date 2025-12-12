from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
from tqdm import tqdm

from app.core.config import settings
from app.db.mongo import get_collection
from app.services.features import build_feature_service, extract_object_features
from app.services.yolo_service import YoloService


SUPPORTED_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".JPG", ".JPEG", ".PNG"
}


def list_images(dataset_root: Path, split: str) -> List[Path]:
    img_dir = dataset_root / "images" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")

    paths = [p for p in img_dir.rglob("*") if p.is_file() and p.suffix in SUPPORTED_EXTS]
    paths.sort()
    return paths


def label_path_for_image(dataset_root: Path, split: str, img_path: Path) -> Path:
    return dataset_root / "labels" / split / (img_path.stem + ".txt")


def yolo_line_to_bbox(line: str, w: int, h: int) -> Optional[Tuple[int, int, int, int, int]]:
    """
    YOLO label: class_id x_center y_center width height
    (usually normalized 0..1)
    Returns: (class_id, x1, y1, x2, y2)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = int(float(parts[0]))
    x = float(parts[1])
    y = float(parts[2])
    bw = float(parts[3])
    bh = float(parts[4])

    normalized = (x <= 1.5 and y <= 1.5 and bw <= 1.5 and bh <= 1.5)
    if normalized:
        x *= w
        y *= h
        bw *= w
        bh *= h

    x1 = int(round(x - bw / 2))
    y1 = int(round(y - bh / 2))
    x2 = int(round(x + bw / 2))
    y2 = int(round(y + bh / 2))

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return class_id, x1, y1, x2, y2


def crop(img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return img_bgr[y1:y2, x1:x2].copy()


def to_jsonable(x: Any) -> Any:
    """
    Recursively convert numpy types to plain Python types for MongoDB.
    """
    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()  # ensure JSON-friendly
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64)):
        return int(x)
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def index_split_to_mongo(dataset_root: str, split: str, limit: Optional[int], drop: bool) -> Dict[str, Any]:
    root = Path(dataset_root)

    # ✅ get class names from YOLO weights
    yolo = YoloService(
        weights_path=settings.YOLO_WEIGHTS,
        conf=settings.YOLO_CONF,
        iou=settings.YOLO_IOU,
        imgsz=settings.YOLO_IMGSZ,
    )
    class_names: Dict[int, str] = yolo.class_names  # {id: name}

    service = build_feature_service()
    col = get_collection("images")

    if drop:
        print("[INFO] Dropping 'images' collection...")
        col.drop()

    images = list_images(root, split)
    if limit is not None:
        images = images[: max(0, int(limit))]

    inserted = 0
    total_objects = 0

    for img_path in tqdm(images, desc=f"Indexing {split}"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        rel_image_path = str(img_path.relative_to(root)).replace("\\", "/")
        lbl_path = label_path_for_image(root, split, img_path)

        objects: List[Dict[str, Any]] = []
        lines = lbl_path.read_text(encoding="utf-8").splitlines() if lbl_path.exists() else []

        for line in lines:
            if not line.strip():
                continue

            parsed = yolo_line_to_bbox(line, w=w, h=h)
            if parsed is None:
                continue

            class_id, x1, y1, x2, y2 = parsed
            obj_crop = crop(img, x1, y1, x2, y2)

            feats, final_vec = extract_object_features(service, obj_crop)

            objects.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), str(class_id)),
                "confidence": 1.0,
                # ✅ FIX: make features JSON/Mongo friendly
                "features": to_jsonable(feats),
                "final_vector": to_jsonable(final_vec),  # list[float]
                "vector_dim": int(len(final_vec)),
            })

        doc = {
            "_id": rel_image_path,
            "image_path": rel_image_path,
            "split": split,
            "width": int(w),
            "height": int(h),
            "objects": objects,
        }

        col.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        inserted += 1
        total_objects += len(objects)

    return {"ok": True, "images_indexed": inserted, "objects_indexed": total_objects, "split": split}


def main():
    parser = argparse.ArgumentParser(description="Index dataset split objects into MongoDB.")
    parser.add_argument("--dataset-root", default=settings.DATASET_ROOT)
    parser.add_argument("--split", default=getattr(settings, "DATASET_SPLIT", "val"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()

    res = index_split_to_mongo(
        dataset_root=args.dataset_root,
        split=args.split,
        limit=args.limit,
        drop=args.drop,
    )
    print(res)


if __name__ == "__main__":
    main()
