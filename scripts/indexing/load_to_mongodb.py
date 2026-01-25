#!/usr/bin/env python3
"""
Load FAISS index metadata into MongoDB.

This script:
1. Loads object metadata from object_mapping.json
2. Groups objects by image_path
3. Stores in MongoDB 'images' collection
4. Optionally stores individual objects in 'objects' collection
5. Updates index_metadata with FAISS index info

Images themselves are NOT stored in MongoDB - only metadata.
Images are accessed via mounted volumes or local file system.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.core.config import settings
from backend.db.mongo import get_collection
from pymongo import MongoClient


def load_faiss_metadata(faiss_metadata_path: Path) -> Dict[str, Any]:
    """Load FAISS index metadata."""
    with open(faiss_metadata_path, 'r') as f:
        return json.load(f)


def load_object_mapping(mapping_path: Path) -> List[Dict[str, Any]]:
    """Load object mapping from JSON file."""
    print(f"Loading object mapping from {mapping_path}...")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    print(f"✅ Loaded {len(mapping)} objects")
    return mapping


def group_objects_by_image(objects: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group objects by image_path."""
    grouped = {}
    for obj in objects:
        image_path = obj["image_path"]
        # Normalize path separators
        image_path = image_path.replace("\\", "/")
        if image_path not in grouped:
            grouped[image_path] = []
        grouped[image_path].append(obj)
    return grouped


def determine_split(image_path: str) -> str:
    """Determine dataset split from image path."""
    if "/train/" in image_path or "train\\" in image_path:
        return "train"
    elif "/val/" in image_path or "val\\" in image_path:
        return "val"
    elif "/test/" in image_path or "test\\" in image_path:
        return "test"
    return "unknown"


def insert_images_collection(
    images_col: Collection,
    grouped_objects: Dict[str, List[Dict[str, Any]]],
    drop_existing: bool = False,
    get_collection_func=None
) -> Dict[str, int]:
    """Insert/update images collection grouped by image_path."""
    if drop_existing:
        print("⚠️  Dropping existing 'images' collection...")
        images_col.drop()
        if get_collection_func:
            images_col = get_collection_func("images")  # Recreate after drop
        else:
            images_col = get_collection("images")
    
    stats = {"inserted": 0, "updated": 0, "errors": 0}
    
    print(f"\nInserting {len(grouped_objects)} images into MongoDB...")
    
    for image_path, objects in tqdm(grouped_objects.items(), desc="Loading images", unit="img"):
        try:
            # Normalize path
            image_path = image_path.replace("\\", "/")
            
            # Prepare objects array
            mongo_objects = []
            for obj in objects:
                mongo_obj = {
                    "object_idx": int(obj["object_idx"]),
                    "faiss_id": int(obj["faiss_id"]),
                    "bbox": [int(x) for x in obj["bbox"]],
                    "class_id": int(obj["class_id"]),
                    "class_name": str(obj["class_name"]),
                    "confidence": float(obj.get("confidence", 0.0))
                }
                mongo_objects.append(mongo_obj)
            
            # Determine split
            split = determine_split(image_path)
            
            # Create document
            doc = {
                "_id": image_path,  # Use image_path as _id for easy lookup
                "image_path": image_path,
                "split": split,
                "objects": mongo_objects,
                "num_objects": len(mongo_objects),
                "indexed_at": datetime.now()
            }
            
            # Insert or update
            result = images_col.replace_one(
                {"_id": image_path},
                doc,
                upsert=True
            )
            
            if result.upserted_id:
                stats["inserted"] += 1
            else:
                stats["updated"] += 1
                
        except Exception as e:
            stats["errors"] += 1
            print(f"\n⚠️  Error processing {image_path}: {e}")
    
    return stats


def insert_objects_collection(
    objects_col: Collection,
    all_objects: List[Dict[str, Any]],
    drop_existing: bool = False,
    get_collection_func=None
) -> Dict[str, int]:
    """Insert individual objects into objects collection."""
    if drop_existing:
        print("⚠️  Dropping existing 'objects' collection...")
        objects_col.drop()
        if get_collection_func:
            objects_col = get_collection_func("objects")
        else:
            objects_col = get_collection("objects")
    
    stats = {"inserted": 0, "updated": 0, "errors": 0}
    
    print(f"\nInserting {len(all_objects)} individual objects into MongoDB...")
    
    for obj in tqdm(all_objects, desc="Loading objects", unit="obj"):
        try:
            image_path = obj["image_path"].replace("\\", "/")
            object_id = f"{image_path}__{obj['object_idx']}"
            
            doc = {
                "_id": object_id,
                "faiss_id": int(obj["faiss_id"]),
                "image_path": image_path,
                "object_idx": int(obj["object_idx"]),
                "bbox": [int(x) for x in obj["bbox"]],
                "class_id": int(obj["class_id"]),
                "class_name": str(obj["class_name"]),
                "confidence": float(obj.get("confidence", 0.0))
            }
            
            result = objects_col.replace_one(
                {"_id": object_id},
                doc,
                upsert=True
            )
            
            if result.upserted_id:
                stats["inserted"] += 1
            else:
                stats["updated"] += 1
                
        except Exception as e:
            stats["errors"] += 1
            print(f"\n⚠️  Error processing object {obj.get('faiss_id', 'unknown')}: {e}")
    
    return stats


def update_index_metadata(
    metadata_col: Collection,
    faiss_metadata: Dict[str, Any]
) -> None:
    """Update index_metadata collection with FAISS index info."""
    print("\nUpdating index metadata...")
    
    metadata_col.update_one(
        {"_id": "faiss_index"},
        {
            "$set": {
                "num_vectors": int(faiss_metadata.get("num_vectors", 0)),
                "dimension": int(faiss_metadata.get("dimension", 0)),
                "metric": str(faiss_metadata.get("metric", "cosine")),
                "index_type": str(faiss_metadata.get("index_type", "IndexFlatIP")),
                "updated_at": datetime.now(),
                "vectors_file": str(faiss_metadata.get("vectors_file", "vectors.npy")),
                "ids_file": str(faiss_metadata.get("ids_file", "ids.npy")),
                "index_file": str(faiss_metadata.get("index_file", "index.faiss"))
            }
        },
        upsert=True
    )
    
    print("✅ Index metadata updated")


def get_collection_with_uri(name: str, mongo_uri: str):
    """Get MongoDB collection using a specific URI."""
    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    if db is None:
        raise ValueError(f"MONGO_URI must include a database name: {mongo_uri}")
    return db[name]


def main(
    mapping_path: Path,
    faiss_metadata_path: Path,
    drop_existing: bool = False,
    include_objects_collection: bool = True,
    mongo_uri: str | None = None
):
    """Main function to load FAISS metadata into MongoDB."""
    # Use provided URI or fall back to settings
    effective_uri = mongo_uri or settings.MONGO_URI
    
    # Helper function to get collection
    def _get_collection(name: str):
        if mongo_uri:
            return get_collection_with_uri(name, mongo_uri)
        else:
            return get_collection(name)
    
    print("="*60)
    print("Loading FAISS Metadata into MongoDB")
    print("="*60)
    print(f"MongoDB URI: {effective_uri}")
    print(f"Mapping file: {mapping_path}")
    print(f"FAISS metadata: {faiss_metadata_path}")
    print("="*60 + "\n")
    
    # Verify files exist
    if not mapping_path.exists():
        print(f"❌ Error: Mapping file not found: {mapping_path}")
        return False
    
    if not faiss_metadata_path.exists():
        print(f"❌ Error: FAISS metadata file not found: {faiss_metadata_path}")
        return False
    
    # Load data
    all_objects = load_object_mapping(mapping_path)
    faiss_metadata = load_faiss_metadata(faiss_metadata_path)
    
    # Group objects by image
    grouped_objects = group_objects_by_image(all_objects)
    
    # Get MongoDB collections (use helper function)
    images_col = _get_collection("images")
    objects_col = _get_collection("objects")
    metadata_col = _get_collection("index_metadata")
    
    # Insert data
    images_stats = insert_images_collection(images_col, grouped_objects, drop_existing, _get_collection)
    
    if include_objects_collection:
        objects_stats = insert_objects_collection(objects_col, all_objects, drop_existing, _get_collection)
    else:
        objects_stats = {"inserted": 0, "updated": 0, "errors": 0}
    
    # Update index metadata
    update_index_metadata(metadata_col, faiss_metadata)
    
    # Print summary
    print("\n" + "="*60)
    print("LOAD SUMMARY")
    print("="*60)
    print(f"Images Collection:")
    print(f"  Inserted:  {images_stats['inserted']}")
    print(f"  Updated:   {images_stats['updated']}")
    print(f"  Errors:    {images_stats['errors']}")
    print(f"\nObjects Collection:")
    print(f"  Inserted:  {objects_stats['inserted']}")
    print(f"  Updated:   {objects_stats['updated']}")
    print(f"  Errors:    {objects_stats['errors']}")
    print(f"\nTotal Objects: {len(all_objects)}")
    print(f"Total Images:  {len(grouped_objects)}")
    print("="*60)
    
    # Verify counts
    db_images = images_col.count_documents({})
    db_objects = objects_col.count_documents({}) if include_objects_collection else 0
    
    print(f"\n✅ MongoDB now contains:")
    print(f"   - {db_images} images")
    print(f"   - {db_objects} objects")
    print("\n✅ Ready for FAISS-powered search!\n")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load FAISS metadata into MongoDB")
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=Path("data/index/metadata/object_mapping.json"),
        help="Path to object_mapping.json file"
    )
    parser.add_argument(
        "--faiss-metadata-path",
        type=Path,
        default=Path("data/index/faiss/metadata.json"),
        help="Path to FAISS metadata.json file"
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing collections before loading"
    )
    parser.add_argument(
        "--skip-objects-collection",
        action="store_true",
        help="Skip loading individual objects collection (faster, images collection only)"
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        default=None,
        help="MongoDB connection URI (defaults to settings.MONGO_URI). Use 'mongodb://localhost:27017/objectlens' for local development"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    mapping_path = PROJECT_ROOT / args.mapping_path if not args.mapping_path.is_absolute() else args.mapping_path
    faiss_metadata_path = PROJECT_ROOT / args.faiss_metadata_path if not args.faiss_metadata_path.is_absolute() else args.faiss_metadata_path
    
    success = main(
        mapping_path=mapping_path,
        faiss_metadata_path=faiss_metadata_path,
        drop_existing=args.drop_existing,
        include_objects_collection=not args.skip_objects_collection,
        mongo_uri=args.mongo_uri
    )
    
    sys.exit(0 if success else 1)
