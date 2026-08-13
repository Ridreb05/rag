# syntax=docker/dockerfile:1.7

# Build the React frontend; Node is not included in the runtime image.
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# RunPod Pods cannot run Docker Compose. Copy Qdrant into the application
# image so one container provides a private localhost vector service.
FROM qdrant/qdrant:v1.13.4 AS qdrant-runtime

# CUDA 11.8 supports RTX 4090 (Ada) and is compatible with older RunPod
# host drivers that reject CUDA 13 images before a container can start.
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl ca-certificates libunwind8 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Dependency layer first so uv sync only re-runs when dependencies change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist
COPY --from=qdrant-runtime /qdrant /opt/qdrant
COPY infrastructure/runpod-entrypoint.sh /usr/local/bin/runpod-entrypoint
RUN chmod +x /usr/local/bin/runpod-entrypoint
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# Catch Linux-only torch / Transformers import failures in CI before an image
# is published and deployed to a paid GPU Pod.
RUN PYTHONPATH=/app/src /app/.venv/bin/python -c "from voice_rag.apps.api_gateway.main import app; print(app.title)"

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV QDRANT_URL=http://127.0.0.1:6333
ENV VOICE_RAG_MANAGED_QDRANT=1
ENV VOICE_RAG_DATA_ROOT=/workspace/voice-rag
ENV VOICE_RAG_INDEX_SPLIT=validation

# Data is persisted via the /workspace volume, not baked into the image.
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/runpod-entrypoint"]
