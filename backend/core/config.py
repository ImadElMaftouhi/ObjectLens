from pydantic_settings import BaseSettings
from pathlib import Path


def _find_project_root() -> Path:
    """Find the project root by looking for backend/ and other markers."""
    # Start from this file's location: backend/core/config.py
    current = Path(__file__).resolve()
    
    # Go up to find project root (backend/core/config.py -> backend/core -> backend -> project_root)
    while current.parent != current:
        # Check if current directory is the project root
        # It should have both 'backend/' and either 'frontend/', '.git/', or 'data/'
        has_backend = (current / "backend").is_dir()
        has_marker = (
            (current / "frontend").is_dir() or 
            (current / ".git").is_dir() or 
            (current / "data").is_dir() or
            (current / "docker-compose.yml").exists()
        )
        
        if has_backend and has_marker:
            print(f"[DEBUG] Found project root: {current}")
            return current
        
        current = current.parent
    
    # Fallback: if we can't find project root, assume we're 3 levels up from config.py
    # backend/core/config.py -> go up 3 levels
    fallback = Path(__file__).resolve().parent.parent.parent
    print(f"[DEBUG] Using fallback project root: {fallback}")
    return fallback


# Cache project root
_PROJECT_ROOT = _find_project_root()
print(f"[DEBUG] Project root set to: {_PROJECT_ROOT}")


class Settings(BaseSettings):
    # =====================
    # CORS
    # =====================
    CORS_ORIGINS: str = "http://localhost:5173"

    # =====================
    # MongoDB
    # =====================
    # Default to Docker hostname, but can be overridden via environment variable
    # For local development: MONGO_URI=mongodb://localhost:27017/objectlens
    MONGO_URI: str = "mongodb://mongo:27017/objectlens"

    # =====================
    # 2D Dataset (ImageNet)
    # =====================
    # Path to ImageNet YOLO dataset images
    # Local dev: relative to project root (e.g., "data/imagenet_4_yolo/images")
    # Docker: absolute path in container (e.g., "/data/imagenet_4_yolo/images")
    DATASET_ROOT: str = "data/imagenet_4_yolo/images"
    DATASET_SPLIT: str = "val"

    # =====================
    # 3D Dataset (Pottery Models)
    # =====================
    # Path to 3D model files root directory
    # Expected structure: RAW_DATA_ROOT / "3D Models" / <class_name> / *.obj
    # Local dev: relative to project root (e.g., "data/3D_data/raw")
    # Docker: absolute path in container (e.g., "/data/3D_data/raw")
    RAW_DATA_ROOT: str = "data/3D_data/raw"

    # =====================
    # Image storage
    # =====================
    IMAGE_STORE_DIR: str = "/data/images"

    # =====================
    # YOLO
    # =====================
    YOLO_WEIGHTS: str = "backend/models/yolo/best.pt"
    YOLO_CONF: float = 0.25
    YOLO_IOU: float = 0.45
    YOLO_IMGSZ: int = 640

    # =====================
    # Search
    # =====================
    TOPK_DEFAULT: int = 20

    # =====================
    # Optional class names (comma-separated)
    # =====================
    CLASS_NAMES: str = ""  # safe default

    def _normalize_path(self, path_str: str) -> str:
        """Normalize path string to handle Windows/Docker path differences.
        
        On Windows, paths starting with '/' (like '/data/...') are treated as absolute
        and resolve to C:\data\..., but we want them to be relative to project root.
        Docker-style absolute paths (starting with /) should be treated as relative on Windows.
        """
        import os
        # On Windows, if path starts with / but doesn't have a drive letter, 
        # it's a Docker-style path that should be treated as relative
        if os.name == 'nt' and path_str.startswith('/') and not (len(path_str) > 1 and path_str[1] == ':'):
            # Remove leading slash to make it relative
            path_str = path_str.lstrip('/')
        return path_str

    def get_dataset_root(self) -> Path:
        """Get resolved path to 2D dataset root."""
        path_str = self.DATASET_ROOT
        # Normalize to handle Docker-style paths on Windows
        normalized = self._normalize_path(path_str)
        path = Path(normalized)
        
        # Check if it's a true absolute path
        # On Windows: absolute paths have drive letters (C:\)
        # On Unix: absolute paths start with /
        import os
        is_absolute = (
            (os.name == 'nt' and len(str(path)) > 1 and str(path)[1] == ':') or  # Windows: C:\
            (os.name != 'nt' and str(path).startswith('/'))  # Unix: /
        )
        
        if is_absolute:
            return path.resolve()
        
        # Relative path: resolve from project root
        full_path = (_PROJECT_ROOT / normalized).resolve()
        print(f"[DEBUG] Resolved DATASET_ROOT: {path_str} -> {full_path}")
        return full_path

    def get_raw_data_root(self) -> Path:
        """Get resolved path to 3D dataset root."""
        path_str = self.RAW_DATA_ROOT
        # Normalize to handle Docker-style paths on Windows
        normalized = self._normalize_path(path_str)
        path = Path(normalized)
        
        # Check if it's a true absolute path
        import os
        is_absolute = (
            (os.name == 'nt' and len(str(path)) > 1 and str(path)[1] == ':') or  # Windows: C:\
            (os.name != 'nt' and str(path).startswith('/'))  # Unix: /
        )
        
        if is_absolute:
            return path.resolve()
        
        # Relative path: resolve from project root
        full_path = (_PROJECT_ROOT / normalized).resolve()
        print(f"[DEBUG] Resolved RAW_DATA_ROOT: {path_str} -> {full_path}")
        return full_path

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()