
## Object Detection Strategy
- Fine-tuning YOLOv8n on 15 classes.
- Use Ultralytics’ latest training API.
- Expected mAP@50 should be >90% on 15 classes.
- After detection, crop detected objects with some margin (10–20%) → extract features only from relevant regions (region-based CBIR). This dramatically improves precision over global features.

## Web Application Features (Must-Have)

- Upload image → see YOLO detections drawn + confidence
- Click on a detected object → search images containing similar objects
- Global search fallback (if no objects detected)
- Relevance feedback (user marks relevant/irrelevant → simple Rocchio update)
- Visualization of why two images are similar (feature-wise breakdown)
- Responsive gallery with infinite scroll

## Performance Optimizations

- Pre-compute all features & Faiss index at dataset import time
- Store cropped object thumbnails for fast display
- Use GPU only for initial indexing and YOLO inference (CPU is fine for feature extraction of EfficientNet on cropped regions)

## Search Flow (UX + Backend)

1. User uploads a query image  
2. Frontend sends image → Flask/FastAPI endpoint  
3. YOLOv8n runs → returns list of detected objects with:
   - bounding box coordinates
   - class name + confidence
   - small thumbnail crop (e.g., 224×224)
   - unique temporary object_id (e.g., UUID or index 0,1,2…)
4. Frontend displays the image with drawable boxes + clickable thumbnails below it  
   → “Click the object you want to search for”
5. User clicks **one** object → frontend sends only that cropped region (or its object_id) to the search endpoint  
6. Backend:
   - Extracts the exact same feature vector(s) from the selected crop (deep + hand-crafted)
   - Queries Faiss index (which contains **pre-extracted object-level features**, not whole-image features)
   - Returns top-k most similar object crops + their parent image IDs
7. Gallery shows full images containing those similar objects (with the matching object highlighted)