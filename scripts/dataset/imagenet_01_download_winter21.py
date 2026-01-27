#!/usr/bin/env python3

# filename: imagenet_01_download_images.py
# purpose:  Download ImageNet Winter 2021 synsets + bbox annotations
# output:   data/raw_imagenet/


import os
import requests
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

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
    'n00007846',  # person
]

# Download annotations from: https://www.image-net.org/data/bboxes_annotations.tar.gz
# Download images from Winter 2021 release: https://image-net.org/data/winter21_whole/<WNID>.tar

# Base directories
DATASET_DIR = "data/raw_imagenet"
BBOX_DIR = os.path.join(DATASET_DIR, "bounding_boxes")
WINTER21_BASE_URL = "https://image-net.org/data/winter21_whole"
BBOX_URL = "https://www.image-net.org/data/bboxes_annotations.tar.gz"


def download_synset(wnid):
    tar_url = f"{WINTER21_BASE_URL}/{wnid}.tar"
    tar_path = os.path.join(DATASET_DIR, f"{wnid}.tar")
    extract_dir = os.path.join(DATASET_DIR, wnid)
    try:
        print(f"\nDownloading {wnid}...")
        resp = requests.get(tar_url, stream=True, timeout=300)
        resp.raise_for_status()

        cl = resp.headers.get("content-length")
        if not cl or not cl.strip().isdigit():
            print(f"    Failed {wnid}: missing or invalid Content-Length")
            return False
        total_size = int(cl)
        if total_size <= 0:
            print(f"    Failed {wnid}: Content-Length is 0")
            return False

        bytes_written = 0
        with open(tar_path, "wb") as f, tqdm(
            desc=f"  Downloading {wnid}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    n = len(chunk)
                    bytes_written += n
                    pbar.update(n)

        if bytes_written != total_size:
            print(f"    Failed {wnid}: incomplete download ({bytes_written}/{total_size} bytes)")
            if os.path.exists(tar_path):
                os.remove(tar_path)
            return False

        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(extract_dir)
        os.remove(tar_path)
        print(f"    Completed {wnid}")
        return True
    except Exception as e:
        print(f"    Failed {wnid}: {e}")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        return False


def download_bounding_boxes():
    """
    Download and extract ImageNet bounding box annotations for WNIDS.
    """
    tar_gz_path = os.path.join(DATASET_DIR, "bboxes_annotations.tar.gz")
    temp_extract_dir = os.path.join(DATASET_DIR, "bboxes_temp")

    try:
        # Download bounding box annotations tar.gz
        resp = requests.get(BBOX_URL, stream=True, timeout=600)
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))
        with open(tar_gz_path, "wb") as f, tqdm(
            desc="Downloading bboxes.tar.gz",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Extract parent tar.gz
        os.makedirs(temp_extract_dir, exist_ok=True)
        with tarfile.open(tar_gz_path, 'r:gz') as tar_gz:
            tar_gz.extractall(temp_extract_dir)

        tar_gz_files = list(Path(temp_extract_dir).glob("*.tar.gz"))
        os.makedirs(BBOX_DIR, exist_ok=True)
        extracted_count = 0

        for synset_tar_gz in tar_gz_files:
            wnid = synset_tar_gz.stem.replace('.tar', '')
            if wnid not in WNIDS:
                continue
            synset_temp_dir = os.path.join(temp_extract_dir, f"{wnid}_temp")
            os.makedirs(synset_temp_dir, exist_ok=True)
            try:
                with tarfile.open(synset_tar_gz, 'r:gz') as st:
                    st.extractall(synset_temp_dir)
                annotation_dir = os.path.join(synset_temp_dir, "Annotation", wnid)
                if os.path.exists(annotation_dir):
                    dest_dir = os.path.join(BBOX_DIR, wnid)
                    os.makedirs(dest_dir, exist_ok=True)
                    xml_files = list(Path(annotation_dir).glob("*.xml"))
                    for xml_file in xml_files:
                        shutil.copy2(xml_file, os.path.join(dest_dir, xml_file.name))
                    extracted_count += len(xml_files)
                shutil.rmtree(synset_temp_dir, ignore_errors=True)
            except Exception:
                shutil.rmtree(synset_temp_dir, ignore_errors=True)
                continue

        os.remove(tar_gz_path)
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        print(f"Extracted {extracted_count} bbox annotations")
        return True

    except Exception as e:
        if os.path.exists(tar_gz_path):
            os.remove(tar_gz_path)
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        print(f"Failed to download bounding boxes: {e}")
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
    
    print("\n" + "="*60)
    print("Downloading 15 Synsets (Winter 2021 Release)")
    print("="*60)
    
    success_count = 0
    for wnid in WNIDS:
        if download_synset(wnid):
            success_count += 1
    
    print(f"\n✓ Downloaded {success_count}/{len(WNIDS)} synsets")
    
    download_bounding_boxes()
    verify_downloads()
    
    print("\n" + "="*60)
    print("Download process completed!")
    print("="*60)


if __name__ == "__main__":
    main()
