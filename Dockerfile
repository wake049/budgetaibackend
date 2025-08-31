# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (build tools only if needed for your wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better caching)
# If you use requirements.txt:
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# If you use Poetry instead of requirements.txt, replace the 3 lines above with:
# COPY pyproject.toml poetry.lock* /app/
# RUN pip install --upgrade pip "poetry>=1.7" && poetry config virtualenvs.create false && poetry install --only main --no-interaction

# Copy app code
COPY . /app

# Optional: tiny health endpoint if you don’t already have one
# from fastapi import FastAPI
# app = FastAPI()
# @app.get("/healthz")
# def healthz(): return {"ok": True}

EXPOSE 8000

# Start with Gunicorn + Uvicorn workers (good defaults)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
