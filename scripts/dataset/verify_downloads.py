"""
ImageNet Dataset Verification Script
Verifies downloaded images and bounding box annotations.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import json

# 15 synsets for the project (must match download_synsets.py)
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

# Base directories (must match download_synsets.py)
DATASET_DIR = "raw_data"
BBOX_DIR = os.path.join(DATASET_DIR, "bounding_boxes")

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}


def get_image_files(wnid):
    """Get all image files for a synset."""
    img_dir = os.path.join(DATASET_DIR, wnid)
    if not os.path.exists(img_dir):
        return []
    
    image_files = []
    # return image_files
    for file in Path(img_dir).rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file)
        # also check in subdirectories
    return image_files

def get_bbox_files(wnid):
    """Get all bounding box XML files for a synset."""
    bbox_dir = os.path.join(BBOX_DIR, wnid)
    if not os.path.exists(bbox_dir):
        return []
    
    return list(Path(bbox_dir).glob("*.xml"))


def parse_xml_annotation(xml_path):
    """
    Parse XML annotation file and extract metadata.
    Returns dict with filename, size, object count, and validation status.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract filename (without extension)
        filename_elem = root.find('filename')
        filename = filename_elem.text if filename_elem is not None else None
        
        # Extract size
        size_elem = root.find('size')
        width = int(size_elem.find('width').text) if size_elem is not None and size_elem.find('width') is not None else None
        height = int(size_elem.find('height').text) if size_elem is not None and size_elem.find('height') is not None else None
        
        # Count objects
        objects = root.findall('object')
        object_count = len(objects)
        
        # Extract bounding boxes
        bboxes = []
        for obj in objects:
            bbox_elem = obj.find('bndbox')
            if bbox_elem is not None:
                try:
                    xmin = int(bbox_elem.find('xmin').text)
                    ymin = int(bbox_elem.find('ymin').text)
                    xmax = int(bbox_elem.find('xmax').text)
                    ymax = int(bbox_elem.find('ymax').text)
                    
                    # Validate bbox coordinates
                    if xmin < xmax and ymin < ymax and xmin >= 0 and ymin >= 0:
                        bboxes.append({
                            'xmin': xmin, 'ymin': ymin,
                            'xmax': xmax, 'ymax': ymax
                        })
                except (ValueError, AttributeError):
                    pass
        
        return {
            'valid': True,
            'filename': filename,
            'width': width,
            'height': height,
            'object_count': object_count,
            'bbox_count': len(bboxes),
            'bboxes': bboxes
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def find_matching_image(xml_filename, image_files):
    """
    Find image file that matches the XML annotation filename.
    XML filename is without extension, so we check all extensions.
    """
    for img_file in image_files:
        # Get filename without extension
        img_stem = img_file.stem
        if img_stem == xml_filename:
            return img_file
    return None


def verify_synset(wnid):
    """
    Verify a single synset: images, bounding boxes, and their correspondence.
    """
    print(f"\n{'='*60}")
    print(f"Verifying: {wnid}")
    print(f"{'='*60}")
    
    # Get files
    image_files = get_image_files(wnid)
    bbox_files = get_bbox_files(wnid)
    
    print(f"\nFiles found:")
    print(f"  Images: {len(image_files)}")
    print(f"  Bounding boxes: {len(bbox_files)}")
    
    if len(image_files) == 0 and len(bbox_files) == 0:
        print(f"  ⚠ WARNING: No files found for {wnid}")
        return {
            'wnid': wnid,
            'image_count': 0,
            'bbox_count': 0,
            'valid_xml_count': 0,
            'invalid_xml_count': 0,
            'matched_count': 0,
            'unmatched_images': len(image_files),
            'unmatched_bboxes': len(bbox_files),
            'errors': ['No files found']
        }
    
    # Parse XML files
    xml_data = {}
    valid_xml = 0
    invalid_xml = 0
    
    print(f"\nParsing XML annotations...")
    for bbox_file in bbox_files:
        data = parse_xml_annotation(bbox_file)
        xml_data[bbox_file] = data
        
        if data['valid']:
            valid_xml += 1
        else:
            invalid_xml += 1
            print(f"  ✗ Invalid XML {bbox_file.name}: {data.get('error', 'Unknown error')}")
    
    print(f"  ✓ Valid XML: {valid_xml}")
    if invalid_xml > 0:
        print(f"  ✗ Invalid XML: {invalid_xml}")
    
    # Match images with bounding boxes
    print(f"\nMatching images with annotations...")
    matched_images = set()
    matched_bboxes = set()
    unmatched_images = []
    unmatched_bboxes = []
    
    for bbox_file, data in xml_data.items():
        if not data['valid']:
            unmatched_bboxes.append(bbox_file)
            continue
        
        xml_filename = data['filename']
        if xml_filename:
            matching_image = find_matching_image(xml_filename, image_files)
            if matching_image:
                matched_images.add(matching_image)
                matched_bboxes.add(bbox_file)
            else:
                unmatched_bboxes.append(bbox_file)
        else:
            unmatched_bboxes.append(bbox_file)
    
    # Find images without annotations
    for img_file in image_files:
        if img_file not in matched_images:
            unmatched_images.append(img_file)
    
    matched_count = len(matched_images)
    print(f"  ✓ Matched: {matched_count} image-annotation pairs")
    print(f"  ⚠ Unmatched images: {len(unmatched_images)}")
    print(f"  ⚠ Unmatched annotations: {len(unmatched_bboxes)}")
    
    # Summary statistics
    total_objects = sum(data['object_count'] for data in xml_data.values() if data['valid'])
    total_bboxes = sum(data['bbox_count'] for data in xml_data.values() if data['valid'])
    
    return {
        'wnid': wnid,
        'image_count': len(image_files),
        'bbox_count': len(bbox_files),
        'valid_xml_count': valid_xml,
        'invalid_xml_count': invalid_xml,
        'matched_count': matched_count,
        'unmatched_images': len(unmatched_images),
        'unmatched_bboxes': len(unmatched_bboxes),
        'total_objects': total_objects,
        'total_bboxes': total_bboxes,
        'matched_images': [str(img) for img in matched_images],
        'unmatched_image_list': [str(img) for img in unmatched_images[:10]],  # First 10
        'unmatched_bbox_list': [str(bbox) for bbox in unmatched_bboxes[:10]]  # First 10
    }


def verify_all():
    """
    Verify all synsets and generate comprehensive report.
    """
    print("="*60)
    print("ImageNet Dataset Verification")
    print("="*60)
    
    results = {}
    total_images = 0
    total_bboxes = 0
    total_valid_xml = 0
    total_invalid_xml = 0
    total_matched = 0
    total_unmatched_images = 0
    total_unmatched_bboxes = 0
    
    # Verify each synset
    for wnid in WNIDS:
        result = verify_synset(wnid)
        results[wnid] = result
        
        total_images += result['image_count']
        total_bboxes += result['bbox_count']
        total_valid_xml += result['valid_xml_count']
        total_invalid_xml += result['invalid_xml_count']
        total_matched += result['matched_count']
        total_unmatched_images += result['unmatched_images']
        total_unmatched_bboxes += result['unmatched_bboxes']
    
    # Generate summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    print(f"\nOverall Statistics:")
    print(f"  Total synsets: {len(WNIDS)}")
    print(f"  Total images: {total_images}")
    print(f"  Total bounding boxes: {total_bboxes}")
    print(f"  Valid XML files: {total_valid_xml}")
    print(f"  Invalid XML files: {total_invalid_xml}")
    print(f"  Matched pairs: {total_matched}")
    print(f"  Unmatched images: {total_unmatched_images}")
    print(f"  Unmatched annotations: {total_unmatched_bboxes}")
    
    # Per-synset summary table
    print(f"\n{'Synset':<15} {'Images':<10} {'BBoxes':<10} {'Valid XML':<12} {'Matched':<10} {'Status':<10}")
    print("-" * 80)
    
    for wnid in WNIDS:
        r = results[wnid]
        status = "✓" if r['matched_count'] > 0 and r['valid_xml_count'] > 0 else "✗"
        print(f"{wnid:<15} {r['image_count']:<10} {r['bbox_count']:<10} {r['valid_xml_count']:<12} {r['matched_count']:<10} {status:<10}")
    
    # Detailed issues
    print(f"\n{'='*60}")
    print("DETAILED ISSUES")
    print(f"{'='*60}")
    
    issues_found = False
    for wnid in WNIDS:
        r = results[wnid]
        issues = []
        
        if r['image_count'] == 0:
            issues.append("No images found")
        if r['bbox_count'] == 0:
            issues.append("No bounding boxes found")
        if r['invalid_xml_count'] > 0:
            issues.append(f"{r['invalid_xml_count']} invalid XML files")
        if r['unmatched_images'] > 0:
            issues.append(f"{r['unmatched_images']} images without annotations")
        if r['unmatched_bboxes'] > 0:
            issues.append(f"{r['unmatched_bboxes']} annotations without images")
        
        if issues:
            issues_found = True
            print(f"\n{wnid}:")
            for issue in issues:
                print(f"  ⚠ {issue}")
    
    if not issues_found:
        print("\n✓ No issues found! All synsets are properly downloaded and matched.")
    
    # Save results to JSON
    output_file = os.path.join(DATASET_DIR, "verification_report.json")
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_synsets': len(WNIDS),
                'total_images': total_images,
                'total_bboxes': total_bboxes,
                'total_valid_xml': total_valid_xml,
                'total_invalid_xml': total_invalid_xml,
                'total_matched': total_matched,
                'total_unmatched_images': total_unmatched_images,
                'total_unmatched_bboxes': total_unmatched_bboxes
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Detailed report saved to: {output_file}")
    
    return results


def main():
    """Main function."""
    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: Dataset directory not found: {DATASET_DIR}")
        print("Please run download_synsets.py first.")
        return
    
    results = verify_all()
    
    print("\n" + "="*60)
    print("Verification completed!")
    print("="*60)


if __name__ == "__main__":
    main()

