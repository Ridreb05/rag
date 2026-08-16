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

# vLLM lives in its own venv, deliberately not `uv add`ed to pyproject.toml.
# vLLM's published wheels pull their own torch build (CUDA 12.x); installed
# into /app/.venv it would upgrade the torch==...+cu118 pin above, which
# exists specifically for this host's driver — see the base-image comment.
# A separate venv keeps the two CUDA runtimes from touching each other.
#
# UNVERIFIED AGAINST THIS HOST: whether a CUDA 12.x *container* actually runs
# under the driver that rejected CUDA 13 (the constraint the cu118 pin above
# was chosen to satisfy). Check before relying on this in a demo:
#   nvidia-smi   # driver version, on the Pod
# vllm's version is intentionally unpinned to a specific patch release here —
# pin it once a version is confirmed to serve the intended model correctly on
# this host, rather than asserting a version this comment can't verify.
RUN /root/.local/bin/uv venv /opt/vllm-venv --python 3.11 && \
    /root/.local/bin/uv pip install --python /opt/vllm-venv/bin/python vllm

# Catch Linux-only torch / Transformers import failures in CI before an image
# is published and deployed to a paid GPU Pod.
RUN PYTHONPATH=/app/src /app/.venv/bin/python -c "from voice_rag.api.main import app; print(app.title)"

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV QDRANT_URL=http://127.0.0.1:6333
ENV VOICE_RAG_MANAGED_QDRANT=1
ENV VOICE_RAG_DATA_ROOT=/workspace/voice-rag
ENV VOICE_RAG_INDEX_SPLIT=validation

# Data is persisted via the /workspace volume, not baked into the image.
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/runpod-entrypoint"]
