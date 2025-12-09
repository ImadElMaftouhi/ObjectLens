import os
import random
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

# Paths from your current script
DATASET_DIR = "dataset"
BBOX_DIR = os.path.join(DATASET_DIR, "bounding_boxes")

# Final YOLO dataset root
YOLO_ROOT = "imagenet_yolo15"
IMAGES_DIR = os.path.join(YOLO_ROOT, "images")
LABELS_DIR = os.path.join(YOLO_ROOT, "labels")

os.makedirs(os.path.join(IMAGES_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(IMAGES_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(LABELS_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(LABELS_DIR, "val"), exist_ok=True)

# ---- Class mapping ----
WNID_TO_NAME = {
    'n00007846': 'person',
    'n02958343': 'car',
    'n02924116': 'bus',
    'n04490091': 'truck',
    'n02834778': 'bicycle',
    'n03790512': 'motorcycle',
    'n02084071': 'dog',
    'n02121620': 'cat',
    'n03001627': 'chair',
    'n03642806': 'laptop',
    'n02992529': 'cell phone',
    'n02823428': 'bottle',
    'n03147509': 'cup',
    'n02769748': 'backpack',
    'n04485082': 'telephone',
}

WNID_TO_ID = {wnid: i for i, wnid in enumerate(WNID_TO_NAME.keys())}


def find_image_for_xml(wnid: str, stem: str):
    """
    Try to find the corresponding image file for a given XML stem.
    e.g. stem = 'n02958343_12345'
    """
    img_dir = Path(DATASET_DIR) / wnid
    if not img_dir.exists():
        return None

    # Try common extensions directly in the wnid folder
    for ext in [".JPEG", ".JPG", ".jpg", ".jpeg", ".png"]:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # Fallback: recursive search (slower, but safer)
    for ext in (".JPEG", ".JPG", ".jpg", ".jpeg", ".png"):
        matches = list(img_dir.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]

    return None


def parse_xml_to_yolo(xml_path: Path, class_id: int):
    """
    Parse one ImageNet XML and return:
    - image stem (filename without extension)
    - list of YOLO bboxes: [ (class_id, x_center, y_center, w, h), ... ]
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return None, []

    # Image size
    size = root.find("size")
    if size is None:
        return None, []

    try:
        w = float(size.find("width").text)
        h = float(size.find("height").text)
    except Exception:
        return None, []

    if w <= 0 or h <= 0:
        return None, []

    bboxes = []

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except Exception:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        # Convert to YOLO format
        x_center = ((xmin + xmax) / 2.0) / w
        y_center = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        # Clip to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        # Skip degenerate boxes
        if bw <= 0 or bh <= 0:
            continue

        bboxes.append((class_id, x_center, y_center, bw, bh))

    # Try to get filename from XML if present
    filename_tag = root.find("filename")
    if filename_tag is not None and filename_tag.text:
        img_stem = Path(filename_tag.text).stem
    else:
        img_stem = xml_path.stem  # e.g. n02958343_12345

    return img_stem, bboxes


def build_image_bbox_index():
    """
    Build a dict: image_path -> list of (class_id, x_center, y_center, w, h)
    If multiple XMLs map to same image, we merge boxes.
    """
    img_to_boxes = {}

    for wnid, class_name in WNID_TO_NAME.items():
        class_id = WNID_TO_ID[wnid]
        bbox_wnid_dir = Path(BBOX_DIR) / wnid
        if not bbox_wnid_dir.exists():
            print(f"[WARN] No bbox dir for {wnid} ({class_name})")
            continue

        xml_files = list(bbox_wnid_dir.glob("*.xml"))
        print(f"{wnid} ({class_name}): processing {len(xml_files)} xml files")

        for xml_path in xml_files:
            img_stem, boxes = parse_xml_to_yolo(xml_path, class_id)
            if not boxes or img_stem is None:
                continue

            img_path = find_image_for_xml(wnid, img_stem)
            if img_path is None:
                # No matching image found for this annotation
                continue

            img_path = img_path.resolve()
            if img_path not in img_to_boxes:
                img_to_boxes[img_path] = []

            img_to_boxes[img_path].extend(boxes)

    return img_to_boxes


def split_train_val(images, train_ratio=0.8):
    images = list(images)
    random.shuffle(images)
    n_train = int(len(images) * train_ratio)
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]
    return train_imgs, val_imgs


def copy_and_write_labels(img_to_boxes):
    """
    Copy images into YOLO structure and write label txt files.
    """
    all_images = list(img_to_boxes.keys())
    train_imgs, val_imgs = split_train_val(all_images, train_ratio=0.8)

    def process_split(img_list, split_name):
        for img_path in img_list:
            rel_name = img_path.name  # keep original filename
            dst_img = Path(IMAGES_DIR) / split_name / rel_name
            dst_lbl = Path(LABELS_DIR) / split_name / (Path(rel_name).stem + ".txt")

            # copy image
            shutil.copy2(img_path, dst_img)

            # write label file
            with open(dst_lbl, "w") as f:
                for (cls, xc, yc, bw, bh) in img_to_boxes[img_path]:
                    line = f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
                    f.write(line)

    print(f"\nTotal images with at least one bbox: {len(all_images)}")
    print(f"Splitting 80/20 -> Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    process_split(train_imgs, "train")
    process_split(val_imgs, "val")


def print_effective_summary(img_to_boxes):
    """
    Print per-class summary of images and boxes actually used
    (only images that have at least one bbox and exist on disk).
    """
    wnid_to_img_count = defaultdict(int)
    wnid_to_box_count = defaultdict(int)

    for img_path, boxes in img_to_boxes.items():
        # parent folder name should be the wnid: dataset/<wnid>/<file>
        wnid = img_path.parent.name
        wnid_to_img_count[wnid] += 1
        wnid_to_box_count[wnid] += len(boxes)

    print("\n============================================")
    print("Effective dataset (only images with ≥1 bbox)")
    print("============================================")

    total_imgs = 0
    total_boxes = 0

    for wnid in WNID_TO_NAME.keys():  # keep same order as mapping
        img_c = wnid_to_img_count.get(wnid, 0)
        box_c = wnid_to_box_count.get(wnid, 0)
        total_imgs += img_c
        total_boxes += box_c
        print(f"  {wnid} ({WNID_TO_NAME[wnid]}): {img_c} images, {box_c} boxes")

    print(f"\nTotal effective: {total_imgs} images, {total_boxes} boxes\n")


def main():
    random.seed(42)

    print("Building image -> bbox index...")
    img_to_boxes = build_image_bbox_index()

    # Print per-class effective stats BEFORE copying
    print_effective_summary(img_to_boxes)

    print("Creating YOLO dataset structure...")
    copy_and_write_labels(img_to_boxes)

    print("Done! YOLO dataset created in:", YOLO_ROOT)


if __name__ == "__main__":
    main()
