#!/bin/sh
# Cloud Run entrypoint: the index is already baked into the image (see
# Dockerfile.cloudrun), so unlike the RunPod entrypoint there is no
# corpus download, chunking, or index-build step here — just start
# Qdrant against the baked-in storage and then the API.
set -eu

export QDRANT_URL="http://127.0.0.1:6333"
export QDRANT__STORAGE__STORAGE_PATH=/app/cloudrun-data/qdrant
export QDRANT__SERVICE__HOST=127.0.0.1

qdrant_pid=""
cleanup() {
  if [ -n "$qdrant_pid" ]; then
    kill "$qdrant_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

(
  cd /opt/qdrant
  exec ./qdrant
) &
qdrant_pid=$!

attempts=0
until curl -fsS http://127.0.0.1:6333/ >/dev/null 2>&1; do
  if ! kill -0 "$qdrant_pid" 2>/dev/null; then
    echo "Qdrant exited before becoming ready" >&2
    wait "$qdrant_pid" || true
    exit 1
  fi
  attempts=$((attempts + 1))
  if [ "$attempts" -gt 60 ]; then
    echo "Qdrant did not become ready within 60 seconds" >&2
    exit 1
  fi
  sleep 1
done

echo "Starting Voice RAG API on port ${PORT:-8000}."
# Cloud Run injects PORT and requires the container to listen on it.
exec /app/.venv/bin/uvicorn voice_rag.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
