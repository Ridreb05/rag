# syntax=docker/dockerfile:1.7
# --- frontend build stage ---
# Builds the Vite/React frontend to static assets; only the build output
# (frontend/dist) is copied into the final image below — Node never ships.
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# RunPod Pods cannot run Docker Compose. Copy Qdrant into the application
# image so a single container can provide a private localhost vector service.
FROM qdrant/qdrant:v1.13.4 AS qdrant-runtime

# --- application image ---
# CUDA 13.0 runtime base — matches this project's pinned torch build
# (torch==2.13.0+cu130, pyproject.toml). What matters for compatibility is
# the HOST's NVIDIA driver, not this base image's toolkit version: torch
# wheels bundle their own CUDA runtime libs. Verify the target GPU's driver
# supports CUDA 13.x before deploying — an older driver needs the cu124 pin
# instead (see pyproject.toml's pytorch-cu130 index for how to repoint it).
FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Dependency layer first so `uv sync` only re-runs when deps actually change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist
COPY --from=qdrant-runtime /qdrant /opt/qdrant
COPY infrastructure/runpod-entrypoint.sh /usr/local/bin/runpod-entrypoint
RUN chmod +x /usr/local/bin/runpod-entrypoint
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV QDRANT_URL=http://127.0.0.1:6333
ENV VOICE_RAG_MANAGED_QDRANT=1
ENV VOICE_RAG_DATA_ROOT=/workspace/voice-rag
ENV VOICE_RAG_INDEX_SPLIT=validation

# data/ (Qdrant + BM25 indexes) is a volume mount, not baked into the image —
# see docker-compose.yml. A full-corpus index is several GB; baking it in
# would make every rebuild re-push that much data.
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/runpod-entrypoint"]
