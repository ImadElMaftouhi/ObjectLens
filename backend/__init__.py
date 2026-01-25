# Backend package for ObjectLens API
# 
# This package provides the FastAPI application and related services.
# The frontend interacts with the backend through HTTP API endpoints.
#
# To run the application:
#   uvicorn backend.main:app --reload
#   or
#   python -m backend.main

__version__ = "0.1.0"

# Only import app when explicitly requested to avoid circular imports
# during testing or when importing other modules
def get_app():
    """Lazy import of the FastAPI app to avoid circular imports."""
    from .main import app
    return app

# Export app for convenience, but use lazy loading
__all__ = ["get_app", "__version__"]
