# api/app/main.py
from fastapi import FastAPI

# If you have routers, e.g. in app/routes/image_routes.py:
# from app.routes import image_routes

app = FastAPI(title="ObjectLens API")

# Example health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# If you have routers:
# app.include_router(image_routes.router, prefix="/images", tags=["images"])
