"""
ImageNet Dataset Download Script
Downloads synsets from Winter 2021 release and bounding box annotations.
"""

import os
import requests
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

# 15 synsets for the project

WNIDS = [
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
    'n00007846',  # person   <-- replaced cup
]



# Download annotations from: https://www.image-net.org/data/bboxes_annotations.tar.gz
# Download images from Winter 2021 release: https://image-net.org/data/winter21_whole/<WNID>.tar

# Base directories
DATASET_DIR = "dataset"
BBOX_DIR = os.path.join(DATASET_DIR, "bounding_boxes")
WINTER21_BASE_URL = "https://image-net.org/data/winter21_whole"


def download_synset(wnid):
    """
    Download a single synset from Winter 2021 release.
    """
    tar_url = f"{WINTER21_BASE_URL}/{wnid}.tar"
    tar_path = os.path.join(DATASET_DIR, f"{wnid}.tar")
    extract_dir = os.path.join(DATASET_DIR, wnid)
    
    try:
        # Download tar file
        print(f"\nDownloading {wnid}...")
        response = requests.get(tar_url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(tar_path, 'wb') as f, tqdm(
            desc=f"  Downloading {wnid}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Extract tar file
        print(f"  Extracting {wnid}...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_dir)
        
        # Clean up tar file
        os.remove(tar_path)
        print(f"  ✓ Completed {wnid}")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed {wnid}: {e}")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        return False


def download_bounding_boxes():
    """
    Download ImageNet bounding box annotations.
    Structure: bboxes_annotations.tar.gz -> <WNID>.tar.gz -> Annotation/<WNID>/*.xml
    """
    print("\n" + "="*60)
    print("Downloading Bounding Box Annotations")
    print("="*60)
    
    bbox_url = "https://www.image-net.org/data/bboxes_annotations.tar.gz"
    tar_gz_path = os.path.join(DATASET_DIR, "bboxes_annotations.tar.gz")
    temp_extract_dir = os.path.join(DATASET_DIR, "bboxes_temp")
    
    try:
        # Download tar.gz file
        print("\nDownloading bounding box annotations...")
        response = requests.get(bbox_url, stream=True, timeout=600)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(tar_gz_path, 'wb') as f, tqdm(
            desc="  Downloading bboxes.tar.gz",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Extract the parent tar.gz to get individual <WNID>.tar.gz files
        print("  Extracting parent tar.gz...")
        os.makedirs(temp_extract_dir, exist_ok=True)
        with tarfile.open(tar_gz_path, 'r:gz') as tar_gz:
            tar_gz.extractall(temp_extract_dir)
        
        # Find all .tar.gz files in the extracted directory (these are individual synset archives)
        tar_gz_files = list(Path(temp_extract_dir).glob("*.tar.gz"))
        print(f"  Found {len(tar_gz_files)} synset tar.gz files")
        
        # Extract only the .tar.gz files matching our WNIDs
        print("  Extracting needed synset annotations...")
        os.makedirs(BBOX_DIR, exist_ok=True)
        extracted_count = 0
        
        for synset_tar_gz in tqdm(tar_gz_files, desc="  Processing"):
            try:
                # Extract WNID from filename (format: <wnid>.tar.gz)
                wnid = synset_tar_gz.stem.replace('.tar', '')
                
                if wnid in WNIDS:
                    # Create temporary directory for this synset's extraction
                    synset_temp_dir = os.path.join(temp_extract_dir, f"{wnid}_temp")
                    os.makedirs(synset_temp_dir, exist_ok=True)
                    
                    # Extract the synset's tar.gz
                    with tarfile.open(synset_tar_gz, 'r:gz') as synset_tar:
                        synset_tar.extractall(synset_temp_dir)
                    
                    # Find XML files in Annotation/<WNID>/ directory
                    annotation_dir = os.path.join(synset_temp_dir, "Annotation", wnid)
                    
                    if os.path.exists(annotation_dir):
                        # Copy XML files to final destination
                        dest_dir = os.path.join(BBOX_DIR, wnid)
                        os.makedirs(dest_dir, exist_ok=True)
                        
                        xml_files = list(Path(annotation_dir).glob("*.xml"))
                        for xml_file in xml_files:
                            shutil.copy2(xml_file, os.path.join(dest_dir, xml_file.name))
                        
                        xml_count = len(xml_files)
                        extracted_count += xml_count
                        print(f"    ✓ {wnid}: {xml_count} annotations")
                    else:
                        print(f"    ⚠ {wnid}: Annotation directory not found")
                    
                    # Clean up temporary extraction directory
                    shutil.rmtree(synset_temp_dir, ignore_errors=True)
                    
            except Exception as e:
                print(f"    ✗ Failed to extract {synset_tar_gz.name}: {e}")
                continue
        
        # Clean up
        os.remove(tar_gz_path)
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        
        print(f"  ✓ Extracted {extracted_count} annotation files for {len(WNIDS)} synsets")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download bounding boxes: {e}")
        if os.path.exists(tar_gz_path):
            os.remove(tar_gz_path)
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        return False


def verify_downloads():
    """
    Verify downloaded synsets and annotations.
    """
    print("\n" + "="*60)
    print("Verification Report")
    print("="*60)
    
    print("\nSynsets:")
    total_images = 0
    total_bboxes = 0
    
    for wnid in WNIDS:
        img_dir = os.path.join(DATASET_DIR, wnid)
        bbox_dir = os.path.join(BBOX_DIR, wnid)
        
        img_count = 0
        if os.path.exists(img_dir):
            img_count = len([f for f in Path(img_dir).rglob("*") 
                           if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            total_images += img_count
        
        bbox_count = 0
        if os.path.exists(bbox_dir):
            bbox_count = len(list(Path(bbox_dir).glob("*.xml")))
            total_bboxes += bbox_count
        
        status_img = "✓" if img_count > 0 else "✗"
        status_bbox = "✓" if bbox_count > 0 else "✗"
        print(f"  {wnid}: {status_img} {img_count} images, {status_bbox} {bbox_count} bboxes")
    
    print(f"\nSummary: {total_images} images, {total_bboxes} bounding boxes")


def main():
    """
    Main function to download synsets and bounding boxes.
    """
    print("="*60)
    print("ImageNet Dataset Download Script")
    print("="*60)
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Download synsets from Winter 2021 release
    print("\n" + "="*60)
    print("Downloading Synsets (Winter 2021 Release)")
    print("="*60)
    
    success_count = 0
    for wnid in WNIDS:
        if download_synset(wnid):
            success_count += 1
    
    print(f"\n✓ Downloaded {success_count}/{len(WNIDS)} synsets")
    
    # Download bounding boxes
    download_bounding_boxes()
    
    # Verify downloads
    verify_downloads()
    
    print("\n" + "="*60)
    print("Download process completed!")
    print("="*60)


if __name__ == "__main__":
    main()
