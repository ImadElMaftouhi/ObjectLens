from fastapi import APIRouter, UploadFile, File
from app.core.config import settings
from app.services.yolo_service import YoloService
from app.utils.images import bytes_to_bgr, crop_xyxy, resize_max, bgr_to_data_url
from app.schemas import DetectResponse, DetectionOut, BBoxXYWH

router = APIRouter()

yolo = YoloService(
    weights_path=settings.YOLO_WEIGHTS,
    conf=settings.YOLO_CONF,
    iou=settings.YOLO_IOU,
    imgsz=settings.YOLO_IMGSZ
)

def xyxy_to_xywh(bbox_xyxy: list[int]) -> BBoxXYWH:
    x1, y1, x2, y2 = bbox_xyxy
    x = int(x1)
    y = int(y1)
    w = int(x2 - x1)
    h = int(y2 - y1)
    if w < 0: w = 0
    if h < 0: h = 0
    return BBoxXYWH(x=x, y=y, w=w, h=h)

@router.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = bytes_to_bgr(image_bytes)

    detections = yolo.detect(img)

    out: list[DetectionOut] = []
    for d in detections:
        bbox_xyxy = d["bbox_xyxy"]  # expected [x1,y1,x2,y2]

        crop = crop_xyxy(img, bbox_xyxy)
        crop = resize_max(crop, settings.THUMB_SIZE)
        thumb = bgr_to_data_url(crop)

        out.append(
            DetectionOut(
                id=int(d["id"]),
                bbox_xyxy=[int(v) for v in bbox_xyxy],
                bbox=xyxy_to_xywh([int(v) for v in bbox_xyxy]),  # ✅ add bbox for UI
                class_id=int(d["class_id"]),
                class_name=str(d["class_name"]),
                confidence=float(d["confidence"]),
                thumbnail=thumb,
            )
        )

    return DetectResponse(detections=out)
