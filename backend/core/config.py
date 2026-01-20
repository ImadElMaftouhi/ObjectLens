from pydantic_settings import BaseSettings


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
    # Dataset
    # =====================
    DATASET_ROOT: str = "/data/imagenet_4_yolo/images"
    DATASET_SPLIT: str = "val"
    # Optional raw 3D data root (for development previews)
    RAW_DATA_ROOT: str = "/data/raw_imagenet"

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

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
