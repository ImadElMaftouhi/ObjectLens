import base64
import numpy as np
import cv2

def bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img

def bgr_to_data_url(img_bgr: np.ndarray, quality: int = 90) -> str:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buf = cv2.imencode(".jpg", img_bgr, encode_params)
    if not ok:
        raise ValueError("Could not encode thumbnail to jpg")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def crop_xyxy(img_bgr: np.ndarray, xyxy: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return img_bgr.copy()
    return img_bgr[y1:y2, x1:x2].copy()

def resize_max(img_bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if max(h, w) <= max_side:
        return img_bgr
    scale = max_side / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
