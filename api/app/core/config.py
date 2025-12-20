from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # =====================
    # CORS
    # =====================
    CORS_ORIGINS: str = "http://localhost:5173"

    # =====================
    # MongoDB
    # =====================
    MONGO_URI: str = "mongodb://mongo:27017/objectlens"

    # =====================
    # Dataset (served at /dataset)
    # =====================
    DATASET_ROOT: str = "/data/imagenet_yolo15"

    # =====================
    # YOLO
    # =====================
    YOLO_WEIGHTS: str = "/app/models/yolo/best.pt"
    YOLO_CONF: float = 0.25
    YOLO_IOU: float = 0.45
    YOLO_IMGSZ: int = 640

    # =====================
    # Search
    # =====================
    TOPK_DEFAULT: int = 20

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()








# from pydantic_settings import BaseSettings


# class Settings(BaseSettings):
#     # =====================
#     # CORS
#     # =====================
#     CORS_ORIGINS: str = "http://localhost:5173"

#     # =====================
#     # MongoDB
#     # =====================
#     MONGO_URI: str = "mongodb://mongo:27017/objectlens"

#     # =====================
#     # Dataset
#     # =====================
#     DATASET_ROOT: str = "/data/imagenet_yolo15"
#     DATASET_SPLIT: str = "val"

#     # =====================
#     # Image storage
#     # =====================
#     IMAGE_STORE_DIR: str = "/data/images"

#     # =====================
#     # YOLO
#     # =====================
#     YOLO_WEIGHTS: str = "/app/models/yolo/best.pt"
#     YOLO_CONF: float = 0.25
#     YOLO_IOU: float = 0.45
#     YOLO_IMGSZ: int = 640

#     # =====================
#     # Thumbnails
#     # =====================
#     THUMB_SIZE: int = 256

#     # =====================
#     # Search
#     # =====================
#     TOPK_DEFAULT: int = 20

#     # =====================
#     # Optional class names (comma-separated)
#     # =====================
#     CLASS_NAMES: str = ""  # safe default

#     class Config:
#         env_file = ".env"
#         extra = "ignore"


# settings = Settings()
