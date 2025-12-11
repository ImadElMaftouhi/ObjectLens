# api/Dockerfile
FROM python:3.11-slim

# No .pyc and flushed logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (build tools, if you ever need them)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
# Make sure ultralytics / opencv / numpy are also in requirements.txt if you like,
# but we force-install them here too so they are guaranteed present.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        ultralytics \
        opencv-python-headless \
        numpy

# Copy the rest of the backend
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Adjust "app.main:app" if your FastAPI entrypoint is different
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
