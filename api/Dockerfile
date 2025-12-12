FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Needed by opencv (even headless sometimes) + general runtime deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependency layer (cached unless requirements.txt changes) ----
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# ---- App code ----
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
