#!/bin/sh
# One-container RunPod entrypoint. Qdrant stays on localhost; only FastAPI's
# port 8000 is exposed through RunPod's HTTPS proxy.
set -eu

data_root="${VOICE_RAG_DATA_ROOT:-/workspace/voice-rag}"
mkdir -p "$data_root/data" "$data_root/qdrant"

# All existing Python tooling uses paths relative to /app. A symlink gives it
# persistent RunPod storage without duplicating path configuration everywhere.
if [ ! -e /app/data ]; then
  ln -s "$data_root/data" /app/data
fi

if [ "${VOICE_RAG_MANAGED_QDRANT:-1}" = "1" ]; then
  export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
  export QDRANT__STORAGE__STORAGE_PATH="$data_root/qdrant"
  export QDRANT__SERVICE__HOST=127.0.0.1
  (
    cd /opt/qdrant
    exec ./qdrant
  ) &
  qdrant_pid=$!
  trap 'kill "$qdrant_pid" 2>/dev/null || true' EXIT INT TERM

  attempts=0
  until curl -fsS http://127.0.0.1:6333/ >/dev/null; do
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
fi

index_language="${VOICE_RAG_LANGUAGE:-hi}"
index_split="${VOICE_RAG_INDEX_SPLIT:-validation}"
: "${BM25_PATH:=/app/data/full_index/bm25/${index_language}_${index_split}}"
export BM25_PATH

# Set this only for the initial Pod start. It downloads, normalizes, chunks,
# and indexes the chosen corpus before loading the interactive API models.
if [ "${VOICE_RAG_BOOTSTRAP_INDEX:-0}" = "1" ]; then
  /app/.venv/bin/python -m voice_rag.ingestion.build_corpus --languages "$index_language" --split "$index_split"
  /app/.venv/bin/python -m voice_rag.chunking.build_chunks --languages "$index_language" --split "$index_split"
  /app/.venv/bin/python scripts/build_full_index.py \
    --language "$index_language" \
    --split "$index_split" \
    --index-version "${VOICE_RAG_INDEX_VERSION:-full1}" \
    --qdrant-url "$QDRANT_URL"
fi

exec /app/.venv/bin/uvicorn voice_rag.apps.api_gateway.main:app --host 0.0.0.0 --port 8000
