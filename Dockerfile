# CUDA 13.0 runtime base — matches this project's pinned torch build
# (torch==2.13.0+cu130, pyproject.toml). What matters for compatibility is
# the HOST's NVIDIA driver, not this base image's toolkit version: torch
# wheels bundle their own CUDA runtime libs. Verify the target GPU's driver
# supports CUDA 13.x before deploying (docs/runpod-deployment.md) — an
# older driver needs the cu124 pin instead (see pyproject.toml's
# pytorch-cu130 index for how to repoint it).
FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Dependency layer first so `uv sync` only re-runs when deps actually change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY scripts/ ./scripts/
RUN uv sync --frozen --no-dev

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# data/ (Qdrant + BM25 indexes) is a volume mount, not baked into the image —
# see docker-compose.yml and docs/runpod-deployment.md. A full-corpus index
# is several GB; baking it in would make every rebuild re-push that much data.
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "voice_rag.apps.api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
