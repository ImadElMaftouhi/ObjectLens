import os
import sys
import subprocess
from pymongo import MongoClient

uri = os.environ.get("MONGO_URI")
if not uri:
    print("[indexer] MONGO_URI missing")
    sys.exit(1)

client = MongoClient(uri)
db = client.get_default_database()
col = db["images"]

count = col.count_documents({})
print(f"[indexer] images collection count = {count}")

if count > 0:
    print("[indexer] DB already populated -> skipping indexing")
    sys.exit(0)

print("[indexer] DB empty -> running indexing script...")

subprocess.check_call([
    sys.executable,
    "-m",
    "app.cli.index_split_to_mongo",
    "--dataset-root", os.environ.get("DATASET_ROOT", "/data/imagenet_yolo15"),
    "--split", os.environ.get("DATASET_SPLIT", "train"),
])

print("[indexer] indexing finished successfully")
