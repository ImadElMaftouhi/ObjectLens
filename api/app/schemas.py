from pydantic import BaseModel
from typing import List, Optional


class BBoxXYWH(BaseModel):
    x: int
    y: int
    w: int
    h: int


class DetectionOut(BaseModel):
    id: int
    bbox_xyxy: List[int]          # [x1, y1, x2, y2]
    bbox: BBoxXYWH                # ✅ added (what the UI uses)
    class_id: int
    class_name: str
    confidence: float
    thumbnail: Optional[str] = None  # data URL


class DetectResponse(BaseModel):
    detections: List[DetectionOut]


class SearchRequest(BaseModel):
    thumbnail: str  # data:image/jpeg;base64,...
