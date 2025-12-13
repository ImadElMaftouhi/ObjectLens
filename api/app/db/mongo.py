from pymongo import MongoClient
from app.core.config import settings

_client: MongoClient | None = None

def _get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGO_URI)
    return _client

def get_collection(name: str = "images"):
    """
    Uses the DB name from MONGO_URI:
      mongodb://mongo:27017/objectlens
    """
    client = _get_client()
    db = client.get_default_database()
    if db is None:
        raise ValueError("MONGO_URI must include a database name, e.g. mongodb://mongo:27017/objectlens")
    return db[name]
