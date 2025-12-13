# app/services/yolo_service.py
from ultralytics import YOLO
import numpy as np

class YoloService:
    def __init__(self, weights_path: str, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640):
        self.model = YOLO(weights_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        # ✅ names can be dict or list depending on training
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            self.class_names = {int(k): str(v) for k, v in names.items()}
        else:
            self.class_names = {i: str(n) for i, n in enumerate(names)}

    def detect(self, img_bgr):
        results = self.model.predict(img_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
        r = results[0]
        out = []
        if r.boxes is None:
            return out

        for i, b in enumerate(r.boxes):
            xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            out.append({
                "id": i,
                "bbox_xyxy": xyxy,
                "class_id": cls_id,
                "class_name": self.class_names.get(cls_id, str(cls_id)),
                "confidence": conf,
            })
        return out
