FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt backend/
RUN pip install -r backend/requirements.txt

COPY . backend/

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
