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

# gcc + ninja-build: not needed by this app's own venv, but required by the
# vLLM venv, which JIT-compiles kernels during startup profiling. This base
# is deliberately the slim "runtime" CUDA variant, so it ships no build
# toolchain at all — every startup failure on this Pod so far has been that
# one root cause surfacing through a different library:
#   Triton (multimodal pos-embed kernel) -> needed gcc
#   FlashInfer (sampler kernel)          -> needs ninja *and* nvcc
# nvcc only exists in the far larger -devel base, so FlashInfer's sampler is
# switched off by env var in runpod-entrypoint.sh rather than compiled;
# ninja-build is installed here to cover the JIT paths that need a build tool
# but not a CUDA compiler (torch cpp_extension, Triton caching), so those fail
# now at build time in CI rather than on a paid GPU Pod.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip curl ca-certificates libunwind8 \
    gcc g++ ninja-build \
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
# Verified on the actual host driver (580.173.02 — comfortably supports
# CUDA 13, so CUDA 12.x is not the constraint the base image's cu118 pin was
# chosen against; that limitation was hit on a different, older RunPod host).
#
# Python 3.12, not 3.11, and specifically not the interpreter on PATH:
#
#   3.12 because flashinfer (a vLLM dependency, imported during vLLM's kernel
#   warmup) annotates with `array.array[int]` at module scope. array.array only
#   became subscriptable in 3.12 — verified directly: 3.11 raises
#   "TypeError: type 'array.array' is not subscriptable", 3.12 and 3.13 do not.
#   On 3.11 this killed the engine at the very last startup step, after KV
#   cache had already been allocated successfully.
#
#   --python-preference only-managed because this base image's apt python3.11
#   is a pre-release build reporting `3.11.0rc1` (visible in this app's own
#   outbound request user-agents), missing sys.get_int_max_str_digits, which
#   recent torch releases assume unconditionally. uv's managed builds are
#   complete final releases; the system one here is not.
#
# The application's own venv stays on its pinned Python — the two venvs are
# isolated, which is the same reason vLLM's CUDA 12.x torch can coexist with
# the app's cu118 pin.
#
# vllm's version is intentionally unpinned to a specific patch release here —
# pin it once a version is confirmed to serve the intended model correctly on
# this host, rather than asserting a version this comment can't verify.
RUN /root/.local/bin/uv python install 3.12 && \
    /root/.local/bin/uv venv /opt/vllm-venv --python 3.12 --python-preference only-managed && \
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
