
## flow 

1. User uploads query image → YOLO detects objects → user selects one → crop it → extract final_vector (weighted, L2-normalized).

2. FAISS receives that single vector → searches the index → returns top-k nearest neighbor indices.

3. Use those indices to lookup metadata (MongoDB or JSON/pickle) → get parent image path + bbox + class → return top images with highlighted matching object.

Core: FAISS does fast approximate nearest neighbor search on L2 or IP (cosine) distance. You don't compare one-by-one in a loop anymore.