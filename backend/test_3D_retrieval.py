#!/usr/bin/env python3
"""
test_3D_retrieval.py

Test script for the 3D-topk API endpoint at /api/search/3D-topk

Usage:
    python backend/test_3D_retrieval.py
    python backend/test_3D_retrieval.py --model path/to/model.obj
    python backend/test_3D_retrieval.py --method lfd --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests


# Default configuration
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_ENDPOINT = "/api/search/3D-topk"
DEFAULT_TOP_K = 10
DEFAULT_METHOD = "depth"
DEFAULT_METRIC = "l2"
DEFAULT_AGGREGATION = "mean"
DEFAULT_IMAGE_SIZE = 256

# Sample model paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MODELS_DIR = PROJECT_ROOT / "data" / "3D_data" / "raw" / "3D Models"


def find_sample_model() -> Optional[Path]:
    """Find a sample .obj model for testing."""
    # Try specific known models first
    known_models = [
        SAMPLE_MODELS_DIR / "Bowl" / "London B 675.obj",
        SAMPLE_MODELS_DIR / "Bowl" / "m545.obj",
        SAMPLE_MODELS_DIR / "Amphora",
    ]
    
    for model_path in known_models:
        if model_path.is_file():
            return model_path
        elif model_path.is_dir():
            # Find first .obj file in directory
            for obj_file in model_path.glob("*.obj"):
                return obj_file
    
    # Fallback: search for any .obj file
    for obj_file in SAMPLE_MODELS_DIR.rglob("*.obj"):
        return obj_file
    
    return None


def test_3d_topk(
    model_path: Path,
    api_url: str = DEFAULT_API_URL,
    top_k: int = DEFAULT_TOP_K,
    method: str = DEFAULT_METHOD,
    metric: str = DEFAULT_METRIC,
    aggregation: str = DEFAULT_AGGREGATION,
    image_size: int = DEFAULT_IMAGE_SIZE,
    l2_normalize: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Test the 3D-topk API endpoint.
    
    Args:
        model_path: Path to the 3D model file to use as query
        api_url: Base URL of the API server
        top_k: Number of similar models to retrieve
        method: Descriptor method ('lfd' or 'depth')
        metric: Distance metric ('l2', 'l1', or 'cosine')
        aggregation: Aggregation method ('mean' or 'sum')
        image_size: Rendering resolution
        l2_normalize: Whether to apply L2 normalization
        verbose: Print detailed output
    
    Returns:
        API response as dictionary
    """
    endpoint = f"{api_url}{DEFAULT_ENDPOINT}"
    
    if verbose:
        print(f"\n{'='*60}")
        print("3D-TopK API Test")
        print(f"{'='*60}")
        print(f"  Endpoint    : {endpoint}")
        print(f"  Model       : {model_path.name}")
        print(f"  Method      : {method}")
        print(f"  Top-K       : {top_k}")
        print(f"  Metric      : {metric}")
        print(f"  Aggregation : {aggregation}")
        print(f"  Image Size  : {image_size}")
        print(f"  L2 Normalize: {l2_normalize}")
        print(f"{'='*60}\n")
    
    # Prepare request
    params = {
        "top_k": top_k,
        "method": method,
        "metric": metric,
        "aggregation": aggregation,
        "image_size": image_size,
        "l2_normalize": str(l2_normalize).lower(),
    }
    
    # Read model file
    with open(model_path, "rb") as f:
        files = {
            "file": (model_path.name, f, "application/octet-stream")
        }
        
        if verbose:
            print("[INFO] Sending request...")
        
        try:
            response = requests.post(
                endpoint,
                files=files,
                params=params,
                timeout=120,  # 3D processing can be slow
            )
        except requests.exceptions.ConnectionError:
            print(f"\n[ERROR] Could not connect to API server at {api_url}")
            print("        Make sure the FastAPI server is running:")
            print("        uvicorn backend.main:app --reload")
            return {"ok": False, "error": "Connection failed"}
        except requests.exceptions.Timeout:
            print("\n[ERROR] Request timed out (>120s)")
            return {"ok": False, "error": "Timeout"}
    
    # Parse response
    if response.status_code != 200:
        print(f"\n[ERROR] API returned status {response.status_code}")
        print(f"        Response: {response.text[:500]}")
        return {
            "ok": False,
            "error": f"HTTP {response.status_code}",
            "detail": response.text,
        }
    
    result = response.json()
    
    if verbose:
        print(f"[SUCCESS] Retrieved {result.get('num_results', 0)} results")
        print(f"          Indexed models: {result.get('num_indexed', 0)}")
        print(f"\n{'─'*60}")
        print("Top Results:")
        print(f"{'─'*60}")
        
        for item in result.get("results", [])[:10]:
            rank = item.get("rank", "?")
            filename = item.get("filename", "unknown")
            cls = item.get("class", "unknown")
            distance = item.get("distance", 0.0)
            score = item.get("similarity_score", 0.0)
            
            print(f"  #{rank:<3} {filename:<40} [{cls:<15}]  dist={distance:.4f}  score={score:.4f}")
        
        print(f"{'─'*60}\n")
    
    return result


def test_validation_errors(api_url: str = DEFAULT_API_URL, model_path: Optional[Path] = None):
    """Test API validation (invalid method, metric, etc.)"""
    print("\n[TEST] Validation Error Handling")
    print("-" * 40)
    
    if model_path is None:
        model_path = find_sample_model()
        if model_path is None:
            print("[SKIP] No sample model found")
            return
    
    endpoint = f"{api_url}{DEFAULT_ENDPOINT}"
    
    # Test invalid method
    print("  Testing invalid method...")
    with open(model_path, "rb") as f:
        response = requests.post(
            endpoint,
            files={"file": (model_path.name, f)},
            params={"method": "invalid_method"},
            timeout=30,
        )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print("    ✓ Invalid method rejected (400)")
    
    # Test invalid metric
    print("  Testing invalid metric...")
    with open(model_path, "rb") as f:
        response = requests.post(
            endpoint,
            files={"file": (model_path.name, f)},
            params={"metric": "euclidean"},  # Should be 'l2'
            timeout=30,
        )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print("    ✓ Invalid metric rejected (400)")
    
    print("  [PASS] Validation tests passed\n")


def test_both_methods(model_path: Path, api_url: str = DEFAULT_API_URL):
    """Test both LFD and Depth methods."""
    print("\n[TEST] Comparing LFD vs Depth Methods")
    print("-" * 40)
    
    # Test LFD
    print("\n  === LFD Method ===")
    lfd_result = test_3d_topk(
        model_path=model_path,
        api_url=api_url,
        method="lfd",
        top_k=5,
        verbose=False,
    )
    
    if "results" in lfd_result:
        print(f"    Retrieved {len(lfd_result['results'])} results")
        for r in lfd_result["results"][:3]:
            print(f"      - {r['filename']} (dist={r['distance']:.4f})")
    
    # Test Depth
    print("\n  === Depth Method ===")
    depth_result = test_3d_topk(
        model_path=model_path,
        api_url=api_url,
        method="depth",
        top_k=5,
        verbose=False,
    )
    
    if "results" in depth_result:
        print(f"    Retrieved {len(depth_result['results'])} results")
        for r in depth_result["results"][:3]:
            print(f"      - {r['filename']} (dist={r['distance']:.4f})")
    
    print("\n  [PASS] Both methods tested\n")
    
    return lfd_result, depth_result


def main():
    parser = argparse.ArgumentParser(
        description="Test the 3D-topk API endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backend/test_3D_retrieval.py
  python backend/test_3D_retrieval.py --model data/3D_data/raw/3D\ Models/Bowl/m545.obj
  python backend/test_3D_retrieval.py --method lfd --top-k 5
  python backend/test_3D_retrieval.py --all-tests
        """,
    )
    
    parser.add_argument(
        "--model", "-m",
        type=Path,
        default=None,
        help="Path to 3D model file to use as query (default: auto-detect)",
    )
    parser.add_argument(
        "--api-url", "-u",
        type=str,
        default=DEFAULT_API_URL,
        help=f"API base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of similar models to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--method",
        choices=["lfd", "depth"],
        default=DEFAULT_METHOD,
        help=f"Descriptor method (default: {DEFAULT_METHOD})",
    )
    parser.add_argument(
        "--metric",
        choices=["l2", "l1", "cosine"],
        default=DEFAULT_METRIC,
        help=f"Distance metric (default: {DEFAULT_METRIC})",
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "sum"],
        default=DEFAULT_AGGREGATION,
        help=f"Aggregation method (default: {DEFAULT_AGGREGATION})",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Rendering resolution (default: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--l2-normalize",
        action="store_true",
        help="Apply L2 normalization to query features",
    )
    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="Run all tests (validation, both methods, etc.)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    
    args = parser.parse_args()
    
    # Find or validate model path
    if args.model is not None:
        model_path = args.model
        if not model_path.exists():
            print(f"[ERROR] Model file not found: {model_path}")
            sys.exit(1)
    else:
        model_path = find_sample_model()
        if model_path is None:
            print("[ERROR] No sample model found. Please specify --model path/to/model.obj")
            sys.exit(1)
        print(f"[INFO] Using sample model: {model_path}")
    
    # Run tests
    if args.all_tests:
        # Run comprehensive tests
        print("\n" + "=" * 60)
        print(" Running All 3D Retrieval API Tests")
        print("=" * 60)
        
        # Basic test
        print("\n[TEST 1] Basic Endpoint Test")
        test_3d_topk(
            model_path=model_path,
            api_url=args.api_url,
            top_k=5,
            method="depth",
            verbose=True,
        )
        
        # Validation tests
        try:
            test_validation_errors(args.api_url, model_path)
        except Exception as e:
            print(f"  [WARN] Validation tests failed: {e}")
        
        # Compare methods
        try:
            test_both_methods(model_path, args.api_url)
        except Exception as e:
            print(f"  [WARN] Method comparison failed: {e}")
        
        print("\n" + "=" * 60)
        print(" All Tests Complete!")
        print("=" * 60 + "\n")
    
    else:
        # Single test with specified parameters
        result = test_3d_topk(
            model_path=model_path,
            api_url=args.api_url,
            top_k=args.top_k,
            method=args.method,
            metric=args.metric,
            aggregation=args.aggregation,
            image_size=args.image_size,
            l2_normalize=args.l2_normalize,
            verbose=not args.json,
        )
        
        if args.json:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
