# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — LLM Performance Benchmarking CLI Tool
# ─────────────────────────────────────────────────────────────────────────────
# Base image: Python 3.10 slim (minimal footprint, no GPU drivers)
# For GPU support, swap for: pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only requirements first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY src/       ./src/
COPY config/    ./config/
COPY data/      ./data/

# ── Create output directory ───────────────────────────────────────────────────
RUN mkdir -p /app/output

# ── Environment: Hugging Face cache (overridable via .env / docker-compose) ───
# Models are cached here so repeated runs don't re-download them.
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

# ── Default command ───────────────────────────────────────────────────────────
# Override with --config to point at a different config file.
CMD ["python", "src/cli.py", "--config", "config/benchmark_config.yaml"]
