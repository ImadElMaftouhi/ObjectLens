"""
FAISS service for fast similarity search.
Loads FAISS index from disk and provides search functionality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import faiss

from backend.core.config import settings


class FAISSService:
    """Service for FAISS-based similarity search."""
    
    def __init__(self, index_path: Path, ids_path: Path, metadata_path: Path):
        """
        Initialize FAISS service.
        
        Args:
            index_path: Path to FAISS index file (index.faiss)
            ids_path: Path to IDs file (ids.npy)
            metadata_path: Path to metadata JSON file
        """
        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        self.metadata_path = Path(metadata_path)
        
        self.index: Optional[faiss.Index] = None
        self.ids: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}
        
        self._load()
    
    def _load(self) -> None:
        """Load FAISS index, IDs, and metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.ids_path.exists():
            raise FileNotFoundError(f"IDs file not found: {self.ids_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))
        
        # Load IDs
        self.ids = np.load(self.ids_path)
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✅ FAISS index loaded: {self.metadata.get('num_vectors', 0)} vectors, "
              f"{self.metadata.get('dimension', 0)}D")
    
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10
    ) -> List[Tuple[int, float, str]]:
        """
        Search for similar vectors using FAISS.
        
        Args:
            query_vector: Query vector (normalized, shape: [D])
            top_k: Number of results to return
            
        Returns:
            List of (faiss_id, score, object_id) tuples, sorted by score descending
        """
        if self.index is None or self.ids is None:
            raise RuntimeError("FAISS index not loaded. Call _load() first.")
        
        # Ensure query vector is the right shape and dtype
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        
        # Ensure vector is normalized (required for IndexFlatIP)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector /= norm
        
        # Search (returns distances and indices)
        # For IndexFlatIP, higher score = more similar
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # Convert to list of results
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            faiss_id = int(idx)
            score = float(dist)  # Already similarity score for IndexFlatIP
            object_id = str(self.ids[faiss_id])  # Get object ID from mapping
            
            results.append((faiss_id, score, object_id))
        
        # Results are already sorted by FAISS (descending by score)
        return results
    
    def get_dimension(self) -> int:
        """Get the dimension of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.d
    
    def get_num_vectors(self) -> int:
        """Get the number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def search_filtered(
        self,
        query_vector: np.ndarray,
        filtered_faiss_ids: List[int],
        top_k: int = 10
    ) -> List[Tuple[int, float, str]]:
        """
        Search within a filtered set of FAISS IDs.
        
        Used for class-based filtering: filter FIRST by class (via MongoDB),
        then compute similarity ONLY within that filtered subset using FAISS.
        
        This ensures similarity is computed only between vectors of the same class,
        which improves result quality.
        
        Args:
            query_vector: Query vector (normalized, shape: [D])
            filtered_faiss_ids: List of FAISS IDs to search within (pre-filtered by class)
            top_k: Number of results to return
            
        Returns:
            List of (faiss_id, score, object_id) tuples, sorted by score descending
        """
        if self.index is None or self.ids is None:
            raise RuntimeError("FAISS index not loaded. Call _load() first.")
        
        if not filtered_faiss_ids:
            return []
        
        # Ensure query vector is normalized
        query_vector = query_vector.astype(np.float32).flatten()
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector /= norm
        else:
            return []
        
        # Compute similarity for each filtered vector
        # For IndexFlatIP with normalized vectors, dot product = cosine similarity
        results = []
        
        for faiss_id in filtered_faiss_ids:
            if faiss_id < 0 or faiss_id >= self.index.ntotal:
                continue
            
            try:
                # Reconstruct vector at position faiss_id from FAISS index
                vector = self.index.reconstruct(int(faiss_id))
                vector = vector.astype(np.float32).flatten()
                
                # Ensure vector is normalized (should already be, but safe check)
                vec_norm = np.linalg.norm(vector)
                if vec_norm > 0:
                    vector /= vec_norm
                
                # Compute cosine similarity (dot product for normalized vectors)
                score = float(np.dot(query_vector, vector))
                
                object_id = str(self.ids[faiss_id])
                results.append((faiss_id, score, object_id))
                
            except Exception:
                # Skip if reconstruction fails
                continue
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return results[:top_k]


# Global FAISS service instance
_FAISS_SERVICE: Optional[FAISSService] = None


def get_faiss_service(
    index_path: Path | None = None,
    ids_path: Path | None = None,
    metadata_path: Path | None = None
) -> FAISSService:
    """
    Get or initialize global FAISS service.
    
    Args:
        index_path: Path to FAISS index (defaults to data/index/faiss/index.faiss)
        ids_path: Path to IDs file (defaults to data/index/faiss/ids.npy)
        metadata_path: Path to metadata (defaults to data/index/faiss/metadata.json)
    """
    global _FAISS_SERVICE
    
    if _FAISS_SERVICE is None:
        # Default paths relative to project root or absolute
        if index_path is None:
            index_path = Path("data/index/faiss/index.faiss")
        if ids_path is None:
            ids_path = Path("data/index/faiss/ids.npy")
        if metadata_path is None:
            metadata_path = Path("data/index/faiss/metadata.json")
        
        # Resolve paths (handle both relative and absolute)
        project_root = Path(__file__).resolve().parents[2]
        if not index_path.is_absolute():
            index_path = project_root / index_path
        if not ids_path.is_absolute():
            ids_path = project_root / ids_path
        if not metadata_path.is_absolute():
            metadata_path = project_root / metadata_path
        
        _FAISS_SERVICE = FAISSService(index_path, ids_path, metadata_path)
    
    return _FAISS_SERVICE