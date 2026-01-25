#!/usr/bin/env python3
"""
Copy 20 images from each of the 15 categories from imagenet_yolo15/images/train
to scripts/test/images
"""

import shutil
from pathlib import Path
import random

# 15 synset IDs (categories)
SYNSET_IDS = [
    'n02084071',  # dog
    'n02124075',  # cat
    'n02958343',  # car
    'n02924116',  # bus
    'n04490091',  # truck
    'n03001627',  # chair
    'n02823428',  # bottle
    'n02992529',  # cell_phone
    'n02769748',  # backpack
    'n03642806',  # laptop
    'n02942699',  # camera
    'n04254680',  # soccer_ball
    'n03790512',  # motorcycle
    'n04485082',  # tripod
    'n00007846',  # person
]

# Get project root (assuming script is in scripts/test/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SOURCE_DIR = PROJECT_ROOT / "imagenet_yolo15" / "images" / "train"
TARGET_DIR = PROJECT_ROOT / "scripts" / "test" / "images"

def copy_images():
    """Copy 20 images from each category."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} does not exist")
        return
    
    total_copied = 0
    
    for synset_id in SYNSET_IDS:
        # Find all images with this synset ID prefix
        pattern = f"{synset_id}_*"
        matching_images = list(SOURCE_DIR.glob(pattern))
        
        if not matching_images:
            print(f"Warning: No images found for {synset_id}")
            continue
        
        # Select 20 images (or all if less than 20)
        selected = random.sample(matching_images, min(20, len(matching_images)))
        
        # Copy images
        for img_path in selected:
            target_path = TARGET_DIR / img_path.name
            shutil.copy2(img_path, target_path)
            total_copied += 1
        
        print(f"Copied {len(selected)} images for {synset_id}")
    
    print(f"\nTotal images copied: {total_copied}")

if __name__ == "__main__":
    copy_images()

