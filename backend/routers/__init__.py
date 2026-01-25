# API Routes

from .detect import router as detect_router
from .search import router as search_router
from .samples import router as samples_router

__all__ = [
    "detect_router",
    "search_router",
    "samples_router",
    ]