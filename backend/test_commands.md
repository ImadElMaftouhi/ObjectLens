# Test Commands for Search Endpoint

## Prerequisites
- Backend server running on `http://localhost:8000`
- Test image available (or use any image from your dataset)

## Python Test Script

```bash
# Run the test script
python backend/test_search_endpoint.py
```

## Curl Commands

### 1. Test Search Without Class Filtering

```bash
# Get a test image path
TEST_IMAGE="scripts/test/images/your_image.JPEG"
# Or from dataset:
# TEST_IMAGE="data/imagenet_4_yolo/images/val/n00007846_104163.JPEG"

# Search without class filtering
curl -X POST "http://localhost:8000/api/search/topk?top_k=10&metric=cosine&same_class_only=false" \
  -F "file=@${TEST_IMAGE}" \
  | python -m json.tool
```

### 2. Test Search With Class Filtering

First, detect the class of your test image:

```bash
# Get class from search result
CLASS_NAME=$(curl -s -X POST "http://localhost:8000/api/search/topk?top_k=1&metric=cosine&same_class_only=false" \
  -F "file=@${TEST_IMAGE}" \
  | python -c "import sys, json; data=json.load(sys.stdin); print(data['best_objects'][0]['class_name'] if data.get('best_objects') else 'unknown')")

echo "Detected class: $CLASS_NAME"
```

Then search with class filtering:

```bash
# Search with same class only
curl -X POST "http://localhost:8000/api/search/topk?top_k=10&metric=cosine&same_class_only=true" \
  -F "file=@${TEST_IMAGE}" \
  -F "query_class=${CLASS_NAME}" \
  | python -m json.tool
```

### 3. PowerShell Commands (Windows)

```powershell
# Set test image path
$testImage = (Get-ChildItem "scripts\test\images\*.JPEG" | Select-Object -First 1).FullName
# Or: $testImage = "data\imagenet_4_yolo\images\val\n00007846_104163.JPEG"

# Test 1: Without class filtering
curl.exe -X POST "http://localhost:8000/api/search/topk?top_k=10&metric=cosine&same_class_only=false" -F "file=@$testImage"

# Test 2: With class filtering
# First get class
$response = curl.exe -s -X POST "http://localhost:8000/api/search/topk?top_k=1&metric=cosine&same_class_only=false" -F "file=@$testImage" | ConvertFrom-Json
$className = $response.best_objects[0].class_name
Write-Host "Detected class: $className"

# Then search with filtering
curl.exe -X POST "http://localhost:8000/api/search/topk?top_k=10&metric=cosine&same_class_only=true" -F "file=@$testImage" -F "query_class=$className"
```

### 4. Python Requests Example

```python
import requests

# Test image path
test_image = "scripts/test/images/your_image.JPEG"

# Test 1: Without class filtering
with open(test_image, 'rb') as f:
    response = requests.post(
        "http://localhost:8000/api/search/topk",
        params={"top_k": 10, "metric": "cosine", "same_class_only": False},
        files={"file": f}
    )
print(response.json())

# Test 2: With class filtering
# First detect class
with open(test_image, 'rb') as f:
    response = requests.post(
        "http://localhost:8000/api/search/topk",
        params={"top_k": 1, "metric": "cosine", "same_class_only": False},
        files={"file": f}
    )
data = response.json()
class_name = data['best_objects'][0]['class_name']

# Then search with filtering
with open(test_image, 'rb') as f:
    response = requests.post(
        "http://localhost:8000/api/search/topk",
        params={"top_k": 10, "metric": "cosine", "same_class_only": True},
        files={"file": f},
        data={"query_class": class_name}
    )
print(response.json())
```

## Expected Behavior

### Without Class Filtering (`same_class_only=false`):
- Returns top-K results from **all classes**
- Results sorted by similarity score
- May include objects from different classes

### With Class Filtering (`same_class_only=true` + `query_class`):
- **Filters FIRST** by class (via MongoDB query)
- **Then** computes similarity **ONLY** within that filtered subset using FAISS
- Returns top-K results from **same class only**
- Results sorted by similarity score
- All results guaranteed to be from the same class as query

## Verify Class Filtering Works

Compare results:
1. Run search without filtering - note the classes in results
2. Run search with filtering - all results should be same class
3. Check that scores may differ (because similarity is computed only within class)
