"""
Detect objects in an image and return the bounding boxes.
"""

from fastapi import APIRouter, UploadFile, File
from backend.core.config import settings
from backend.services.yolo_service import YoloService
from backend.utils.images import bytes_to_bgr
from backend.schemas import DetectResponse, DetectionOut, BBoxXYWH

router = APIRouter(tags=["detect"])

yolo = YoloService(
    weights_path=settings.YOLO_WEIGHTS,
    conf=settings.YOLO_CONF,
    iou=settings.YOLO_IOU,
    imgsz=settings.YOLO_IMGSZ,
)

def xyxy_to_xywh(bbox_xyxy: list[int]) -> BBoxXYWH:
    x1, y1, x2, y2 = bbox_xyxy
    x = int(x1)
    y = int(y1)
    w = int(x2 - x1)
    h = int(y2 - y1)
    if w < 0:
        w = 0
    if h < 0:
        h = 0
    return BBoxXYWH(x=x, y=y, w=w, h=h)

@router.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = bytes_to_bgr(image_bytes)

    detections = yolo.detect(img)

    out: list[DetectionOut] = []
    for d in detections:
        bbox_xyxy = [int(v) for v in d["bbox_xyxy"]]  # [x1,y1,x2,y2]

        out.append(
            DetectionOut(
                id=int(d["id"]),
                bbox_xyxy=bbox_xyxy,
                bbox=xyxy_to_xywh(bbox_xyxy),
                class_id=int(d["class_id"]),
                class_name=str(d["class_name"]),
                confidence=float(d["confidence"]),
            )
        )

    return DetectResponse(detections=out)

@router.get("/detect/health")
async def health():
    return {"status": "ok"}