#!/usr/bin/env python3
"""Test the system flow.
1. Query the image 
2. Detect the objects in the image
3. Extract the features of a selected object
4. Get the final vector for the query
5. Search top-k similar vector - get FAISS IDs
6. Use FAISS IDs to fetch full metadata from MongoDB
7. Serve from mounted volume to the frontend the images and the metadata
"""

import requests
from pathlib import Path
import json

# Configuration
API_BASE = "http://localhost:8000"
SEARCH_ENDPOINT = f"{API_BASE}/api/search/topk"
DETECT_ENDPOINT = f"{API_BASE}/api/detect"

def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE}/health")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}\n")

def test_detect_endpoint(test_image: Path):
    """Test detect endpoint with a sample image."""
    print("Testing detect endpoint...")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(DETECT_ENDPOINT, files=files)
        
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            detections = data.get('detections', [])
            print(f"  ✅ Success!")
            print(f"  - Found {len(detections)} objects")
            
            if detections:
                top = detections[0]
                print(f"\n  Top detection:")
                print(f"    Class: {top['class_name']}")
                print(f"    Confidence: {top['confidence']:.4f}")
                print(f"    BBox: {top['bbox_xyxy']}")
            
            return detections, True
        else:
            print(f"  ❌ Error: {response.text}")
            return [], False
            
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return [], False

def test_search_endpoint(test_image: Path, query_class: str | None = None, same_class_only: bool = False):
    """Test search endpoint with a sample image."""
    
    test_name = f"Search (class={query_class}, same_class_only={same_class_only})"
    print(f"Testing {test_name}...")
    
    # Prepare request
    params = {
        "top_k": 10,
        "metric": "cosine",
        "same_class_only": same_class_only
    }
    
    form_data = {}
    if query_class:
        form_data["query_class"] = query_class
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(SEARCH_ENDPOINT, params=params, files=files, data=form_data)
        
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Success!")
            print(f"  - Found {len(data.get('best_images', []))} images")
            print(f"  - Found {len(data.get('best_objects', []))} objects")
            print(f"  - Query feature categories: {data.get('query_feature_categories', 'N/A')}")
            print(f"  - Same class only: {data.get('same_class_only', False)}")
            
            if data.get('best_images'):
                top = data['best_images'][0]
                print(f"\n  Top result:")
                print(f"    Image: {top['image_path']}")
                print(f"    Score: {top['score']:.4f}")
                print(f"    Class: {top['best_class_name']}")
            
            # Show classes of top results
            if data.get('best_objects'):
                classes = [obj.get('class_name', 'unknown') for obj in data['best_objects'][:5]]
                print(f"  - Top 5 classes: {set(classes)}")
            
            return True
        else:
            print(f"  ❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing ObjectLens System Workflow")
    print("=" * 60 + "\n")
    
    test_health()
    
    # Find test image
    test_images_dir = Path("scripts/test/images")
    if not test_images_dir.exists():
        # Try alternative locations
        test_images_dir = Path("data/imagenet_4_yolo/images/val")
        if not test_images_dir.exists():
            print(f"  ❌ Test images directory not found")
            exit(1)
    
    test_images = list(test_images_dir.glob("*.JPEG")) + list(test_images_dir.glob("*.jpg"))
    if not test_images:
        print("  ❌ No test images found")
        exit(1)
    
    test_image = test_images[0]
    print(f"Using test image: {test_image.name}\n")
    
    # Step 2: Detect objects via API
    detections, detect_success = test_detect_endpoint(test_image)
    
    detected_class = None
    if detections:
        detected_class = detections[0]['class_name']  # Top detection
        print(f"Detected class: {detected_class}\n")
    
    # Test 1: Search without class filtering
    success1 = test_search_endpoint(test_image, query_class=None, same_class_only=False)
    
    print()
    
    # Test 2: Search with class filtering using detected class
    success2 = test_search_endpoint(test_image, query_class=detected_class, same_class_only=True) if detected_class else True
    
    print("\n" + "=" * 60)
    if success1 and success2 and detect_success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)