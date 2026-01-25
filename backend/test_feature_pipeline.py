#!/usr/bin/env python3
"""
Test script to verify feature extraction and similarity computation pipeline.
Tests loading images, extracting features, and computing similarities.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

# Add backend to path
backend_root = Path(__file__).parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from services.feature_extraction import (
    FeatureExtractionService,
    FourierDescriptorExtractor,
    OrientationHistogramExtractor, 
    TamuraExtractor,
    GaborExtractor,
    HSVHistogramExtractor,
    SimilarityComputer
)

def load_test_image(image_path: str, max_size: int = 500) -> np.ndarray:
    """Load and preprocess an image for feature extraction."""
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if too large (maintain aspect ratio)
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img

def test_feature_extraction():
    """Test feature extraction on a single image."""
    print("=== Testing Feature Extraction ===")
    
    # Setup feature extraction service
    extractors = [
        FourierDescriptorExtractor(n_coeff=15, contour_mode="canny"),
        OrientationHistogramExtractor(bins=36, contour_mode="canny"),
        TamuraExtractor(kmax=4, n_bins=16),
        GaborExtractor(n_scales=3, n_orientations=4),
        HSVHistogramExtractor(h_bins=4, sv_bins=4),
    ]
    
    feature_service = FeatureExtractionService(extractors)
    
    # Test with a sample image
    test_images_dir = Path("scripts/test/images/")
    if not test_images_dir.exists():
        print(f"❌ Test images directory not found: {test_images_dir}")
        return False
    
    # Get first available image
    image_files = list(test_images_dir.glob("*.JPEG"))
    if not image_files:
        print("❌ No test images found")
        return False
    
    test_image_path = image_files[0]
    print(f"Testing with image: {test_image_path.name}")
    
    try:
        # Load image
        img = load_test_image(str(test_image_path))
        print(f"✅ Image loaded successfully: shape {img.shape}")
        
        # Extract features
        features = feature_service.extract(img)
        print("✅ Features extracted successfully")
        
        # Check feature structure
        expected_categories = ["form", "texture", "color"]
        for cat in expected_categories:
            if cat in features:
                if "combined" in features[cat]:
                    vec = features[cat]["combined"]
                    print(f"  - {cat}: combined vector shape {vec.shape}, dtype {vec.dtype}")
                else:
                    print(f"\t ❌ {cat}: missing 'combined' vector")
            else:
                print(f"\t ❌ Missing category: {cat}")
        
        return True, features
        
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        return False, None

def test_similarity_computation():
    """Test similarity computation between two feature sets."""
    print("\n=== Testing Similarity Computation ===")
    
    # Setup services
    extractors = [
        FourierDescriptorExtractor(n_coeff=15, contour_mode="canny"),
        OrientationHistogramExtractor(bins=36, contour_mode="canny"), 
        TamuraExtractor(kmax=4, n_bins=16),
        GaborExtractor(n_scales=3, n_orientations=4),
        HSVHistogramExtractor(h_bins=8, sv_bins=8),
    ]
    
    feature_service = FeatureExtractionService(extractors)
    similarity_computer = SimilarityComputer()
    
    # Load two different images
    test_images_dir = Path("scripts/test/images/")
    image_files = list(test_images_dir.glob("*.JPEG"))
    
    if len(image_files) < 2:
        print("❌ Need at least 2 test images for similarity test")
        return False
    
    try:
        # Load two images
        img1 = load_test_image(str(image_files[0]))
        img2 = load_test_image(str(image_files[1]))
        
        print(f"Comparing: {image_files[0].name} vs {image_files[1].name}")
        
        # Extract features
        features1 = feature_service.extract(img1)
        features2 = feature_service.extract(img2)
        
        # Compute similarity
        similarity = similarity_computer.compute(
            features1, 
            features2, 
            categories=["form", "texture", "color"],
            metric="cosine"
        )
        
        print("✅ Similarity computation successful")
        
        # Test self-similarity (should be close to 1.0)
        self_similarity = similarity_computer.compute(
            features1,
            features1,
            categories=["form", "texture", "color"], 
            metric="cosine"
        )
        # Test different categories individually
        for cat in ["form", "texture", "color"]:
            cat_sim = similarity_computer.compute(
                features1, features2, 
                categories=[cat], 
                metric="cosine"
            )
            print(f"  - {cat} similarity: {cat_sim:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during similarity computation: {e}")
        return False

def test_combined_vector_extraction():
    """Test extracting a single combined vector per image (for your new indexing approach)."""
    print("\n=== Testing Combined Vector Extraction ===")
    
    extractors = [
        FourierDescriptorExtractor(n_coeff=15, contour_mode="canny"),
        OrientationHistogramExtractor(bins=36, contour_mode="canny"),
        TamuraExtractor(kmax=4, n_bins=16),
        GaborExtractor(n_scales=3, n_orientations=4),
        HSVHistogramExtractor(h_bins=8, sv_bins=8),
    ]
    
    feature_service = FeatureExtractionService(extractors)
    
    # Load test image
    test_images_dir = Path("scripts/test/images/")
    image_files = list(test_images_dir.glob("*.JPEG"))
    
    if not image_files:
        print("❌ No test images found")
        return False
    
    try:
        img = load_test_image(str(image_files[0]))
        features = feature_service.extract(img)
        
        # Combine all category vectors into one final vector
        all_combined = []
        for cat in ["form", "texture", "color"]:
            if cat in features and "combined" in features[cat]:
                all_combined.append(features[cat]["combined"])
        
        if all_combined:
            final_vector = np.concatenate(all_combined).astype(np.float32)
            
            # L2 normalize
            norm = np.linalg.norm(final_vector)
            if norm > 0:
                final_vector /= norm
            
            print(f"✅ Combined vector created: shape {final_vector.shape}, dtype {final_vector.dtype}")
            print(f"  - Vector norm: {np.linalg.norm(final_vector):.6f} (should be ~1.0)")
            print(f"  - Value range: [{final_vector.min():.4f}, {final_vector.max():.4f}]")
            
            return True, final_vector
        else:
            print("❌ Could not create combined vector")
            return False, None
            
    except Exception as e:
        print(f"❌ Error creating combined vector: {e}")
        return False, None

def main():
    """Run all tests."""
    print("\n", "="*50)
    print("Testing Feature Extraction and Similarity Pipeline...", end='\n')
    
    # print(f"Working directory: {os.getcwd()}") ; exit(0)

    # Test 1: Feature extraction
    success1, features = test_feature_extraction()
    
    # Test 2: Similarity computation
    success2 = test_similarity_computation()
    
    # Test 3: Combined vector extraction
    success3, combined_vector = test_combined_vector_extraction()
    
    # Summary
    print("\n" + "="*50)
    print(" ===== TEST SUMMARY ===== ")
    print("="*50)
    print(f"Feature Extraction: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Similarity Computation: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"Combined Vector Extraction: {'✅ PASS' if success3 else '❌ FAIL'}")
    
    all_passed = success1 and success2 and success3
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🚀 Your feature extraction pipeline is ready for integration!")
        # print("   Next steps:")
        # print("   1. Update your indexing script to use combined vectors")
        # print("   2. Integrate with your API endpoints")
        # print("   3. Test with the frontend")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())