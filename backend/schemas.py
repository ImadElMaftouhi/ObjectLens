from pydantic import BaseModel
from typing import List


class BBoxXYWH(BaseModel):
    x: int
    y: int
    w: int
    h: int

class DetectionOut(BaseModel):
    id: int
    bbox_xyxy: List[int]          # [x1, y1, x2, y2]
    bbox: BBoxXYWH                # UI-friendly format
    class_id: int
    class_name: str
    confidence: float

class DetectResponse(BaseModel):
    detections: List[DetectionOut]

class SearchResponseImage(BaseModel):
    image_path: str
    image_url: str
    score: float
    best_bbox: List[int]
    best_class_id: int
    best_class_name: str

class SearchResponse(BaseModel):
    top_k: int
    best_images: List[SearchResponseImage]
