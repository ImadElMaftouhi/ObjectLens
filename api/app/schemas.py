from pydantic import BaseModel
from typing import List


class BBoxXYWH(BaseModel):
    x: int
    y: int
    w: int
    h: int


class DetectionOut(BaseModel):
    id: int
    bbox_xyxy: List[int]  # [x1, y1, x2, y2]
    bbox: BBoxXYWH        # what the UI uses (x, y, w, h)
    class_id: int
    class_name: str
    confidence: float


class DetectResponse(BaseModel):
    detections: List[DetectionOut]
