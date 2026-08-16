#!/bin/sh
# One-container RunPod entrypoint. Qdrant stays on localhost; only FastAPI's
# port 8000 is exposed through RunPod's HTTPS proxy.
set -eu

data_root="${VOICE_RAG_DATA_ROOT:-/workspace/voice-rag}"
mkdir -p "$data_root/data" "$data_root/qdrant" "$data_root/huggingface"

index_language="${VOICE_RAG_LANGUAGE:-hi}"
index_split="${VOICE_RAG_INDEX_SPLIT:-validation}"
index_version="${VOICE_RAG_INDEX_VERSION:-full1}"
# Isolate Qdrant storage by index version. A stopped legacy collection can
# otherwise resume its expensive optimizer in the background and starve a
# clean/demo bootstrap mounted on the same network volume.
qdrant_storage_path="${VOICE_RAG_QDRANT_STORAGE_PATH:-$data_root/qdrant/$index_version}"
mkdir -p "$qdrant_storage_path"

qdrant_pid=""
vllm_pid=""
bootstrap_lock_dir=""

release_bootstrap_lock() {
  if [ -n "$bootstrap_lock_dir" ]; then
    rm -f "$bootstrap_lock_dir/owner"
    rmdir "$bootstrap_lock_dir" 2>/dev/null || true
    bootstrap_lock_dir=""
  fi
}

cleanup() {
  release_bootstrap_lock
  if [ -n "$vllm_pid" ]; then
    kill "$vllm_pid" 2>/dev/null || true
  fi
  if [ -n "$qdrant_pid" ]; then
    kill "$qdrant_pid" 2>/dev/null || true
  fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Keep dataset and model downloads across Pod restarts.
export HF_HOME="$data_root/huggingface"

# All existing Python tooling uses paths relative to /app. A symlink gives it
# persistent RunPod storage without duplicating path configuration everywhere.
if [ ! -e /app/data ]; then
  ln -s "$data_root/data" /app/data
fi

if [ "${VOICE_RAG_MANAGED_QDRANT:-1}" = "1" ]; then
  export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
  export QDRANT__STORAGE__STORAGE_PATH="$qdrant_storage_path"
  export QDRANT__SERVICE__HOST=127.0.0.1
  (
    cd /opt/qdrant
    exec ./qdrant
  ) &
  qdrant_pid=$!

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

# Local generation backend. Runs in its own venv (/opt/vllm-venv), built
# separately from /app/.venv — vLLM's published wheels pull their own torch
# build (CUDA 12.x), which would otherwise force-upgrade the torch==...+cu118
# this image is pinned to for BGE-M3/reranker/NLI compatibility with this
# host's driver (see Dockerfile). Two venvs, two CUDA runtimes, one GPU: the
# isolation avoids a dependency conflict, not the GPU-memory sharing, which
# --gpu-memory-utilization below still has to budget for by hand.
if [ "${VOICE_RAG_GENERATION_BACKEND:-gemini}" = "vllm" ]; then
  vllm_model="${VOICE_RAG_VLLM_MODEL:-Qwen/Qwen3.5-4B}"
  vllm_port="${VOICE_RAG_VLLM_PORT:-8001}"
  export VOICE_RAG_VLLM_BASE_URL="${VOICE_RAG_VLLM_BASE_URL:-http://127.0.0.1:${vllm_port}/v1}"
  # FlashInfer JIT-compiles its sampling kernel at startup, which needs ninja
  # AND nvcc. This image is nvidia/cuda:*-runtime, which ships neither (only
  # the far larger -devel variant has nvcc), so that build fails and takes the
  # whole engine down with it — confirmed on the Pod: FileNotFoundError:
  # 'ninja', after model weights had already loaded successfully. Installing
  # ninja alone would only move the failure to nvcc one step later.
  #
  # Disabling it costs nothing here: this deployment samples at
  # temperature=0, i.e. greedy decoding, so FlashInfer's optimized top-k/top-p
  # kernel has no work to do that the PyTorch-native path doesn't already do.
  # It is only exercised at all because vLLM's startup profiling runs a dummy
  # *random* sampler pass regardless of how the server is actually used.
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  echo "Starting vLLM ($vllm_model) on port $vllm_port."
  (
    exec /opt/vllm-venv/bin/vllm serve "$vllm_model" \
      --port "$vllm_port" \
      --dtype bfloat16 \
      --max-model-len "${VOICE_RAG_VLLM_MAX_MODEL_LEN:-4096}" \
      --gpu-memory-utilization "${VOICE_RAG_VLLM_GPU_MEM_FRACTION:-0.55}" \
      --enforce-eager
  ) &
  vllm_pid=$!
  vllm_attempts=0
  until curl -fsS "http://127.0.0.1:${vllm_port}/health" >/dev/null 2>&1; do
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      echo "vLLM exited before becoming ready — check GPU memory (BGE-M3/reranker/NLI already hold some) and that $vllm_model is a real, accessible model id." >&2
      wait "$vllm_pid" || true
      exit 1
    fi
    vllm_attempts=$((vllm_attempts + 1))
    if [ "$vllm_attempts" -gt "${VOICE_RAG_VLLM_READY_TIMEOUT_SECONDS:-300}" ]; then
      # 300s covers a warm boot (weights already in $HF_HOME on the volume).
      # A cold first boot downloads ~8GB of weights first — raise this for
      # that boot specifically rather than raising the default for every one.
      echo "vLLM did not become ready within ${VOICE_RAG_VLLM_READY_TIMEOUT_SECONDS:-300} seconds" >&2
      exit 1
    fi
    sleep 1
  done
  echo "vLLM ready."
fi

: "${BM25_PATH:=/app/data/full_index/bm25/${index_language}_${index_split}_${index_version}}"
export BM25_PATH
state_path="$data_root/data/full_index/${index_language}_${index_split}_${index_version}.state.json"
mkdir -p "$data_root/data/full_index"
# The API uses this manifest plus Qdrant's exact point count for /v1/health,
# so an interrupted upload cannot appear ready simply because its collection
# happens to exist.
export VOICE_RAG_INDEX_STATE_PATH="/app/data/full_index/${index_language}_${index_split}_${index_version}.state.json"
export VOICE_RAG_REQUIRE_COMPLETE_INDEX="${VOICE_RAG_REQUIRE_COMPLETE_INDEX:-1}"

# Keep this enabled for the first boot and later restarts. The index builder
# verifies its state and exits quickly when the selected version is complete.
if [ "${VOICE_RAG_BOOTSTRAP_INDEX:-0}" = "1" ]; then
  bootstrap_marker="$data_root/data/full_index/${index_language}_${index_split}_${index_version}_bootstrap_complete"
  bootstrap_lock_dir="$data_root/data/full_index/.${index_language}_${index_split}_${index_version}.bootstrap.lock"
  if ! mkdir "$bootstrap_lock_dir" 2>/dev/null; then
    # A forced Pod stop can leave this empty directory on the persistent
    # volume. Recovery is deliberately opt-in: automatically removing it
    # could allow two Pods to write the same BM25 index concurrently.
    if [ "${VOICE_RAG_RECOVER_STALE_BOOTSTRAP_LOCK:-0}" = "1" ]; then
      echo "Recovering explicitly approved stale bootstrap lock: $bootstrap_lock_dir"
      rm -f "$bootstrap_lock_dir/owner"
      if ! rmdir "$bootstrap_lock_dir" 2>/dev/null; then
        echo "Bootstrap lock is still active or not empty: $bootstrap_lock_dir" >&2
        exit 1
      fi
      if ! mkdir "$bootstrap_lock_dir" 2>/dev/null; then
        echo "Bootstrap lock is still active or not empty: $bootstrap_lock_dir" >&2
        exit 1
      fi
    else
      echo "Another bootstrap may be using $bootstrap_lock_dir; refusing to corrupt the shared index." >&2
      echo "If every other Pod using this volume is stopped, set VOICE_RAG_RECOVER_STALE_BOOTSTRAP_LOCK=1 for one restart." >&2
      exit 1
    fi
  fi
  printf '%s\n' "pid=$$ started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$bootstrap_lock_dir/owner"

  chunks_path="$data_root/data/processed/${index_language}/${index_split}_chunks.parquet"
  passages_path="$data_root/data/processed/${index_language}/${index_split}_passages.parquet"
  if [ ! -f "$chunks_path" ]; then
    if [ ! -f "$passages_path" ]; then
      echo "Downloading and normalizing the corpus."
      /app/.venv/bin/python -m voice_rag.pipeline.ingestion.build_corpus --languages "$index_language" --split "$index_split"
    else
      echo "Reusing persisted normalized corpus."
    fi
    echo "Building chunks from the persisted corpus."
    /app/.venv/bin/python -m voice_rag.pipeline.chunking.build_chunks --languages "$index_language" --split "$index_split"
  else
    echo "Reusing persisted chunks."
  fi

  # An old bootstrap has no state manifest, so its partial Qdrant/BM25 data
  # cannot be trusted. Reset it once. New bootstraps write a manifest after
  # every committed batch and resume without discarding completed work.
  set -- /app/.venv/bin/python scripts/build_full_index.py \
    --language "$index_language" \
    --split "$index_split" \
    --index-version "$index_version" \
    --qdrant-url "$QDRANT_URL" \
    --optimizer-wait-seconds "${VOICE_RAG_OPTIMIZER_WAIT_SECONDS:-900}"
  if [ ! -f "$state_path" ]; then
    set -- "$@" --reset
  fi
  if [ -n "${VOICE_RAG_BOOTSTRAP_LIMIT:-}" ]; then
    case "$VOICE_RAG_BOOTSTRAP_LIMIT" in
      *[!0-9]*|"")
        echo "VOICE_RAG_BOOTSTRAP_LIMIT must be a positive integer" >&2
        exit 1
        ;;
    esac
    if [ "$VOICE_RAG_BOOTSTRAP_LIMIT" -lt 1 ]; then
      echo "VOICE_RAG_BOOTSTRAP_LIMIT must be a positive integer" >&2
      exit 1
    fi
    set -- "$@" --limit "$VOICE_RAG_BOOTSTRAP_LIMIT"
  fi
  echo "Starting resumable index bootstrap. State: $state_path"
  "$@"
  touch "$bootstrap_marker"
  release_bootstrap_lock
  echo "Index bootstrap completed."
fi

echo "Starting Voice RAG API on port 8000."
# ``exec`` replaces this shell, so release the bootstrap lock explicitly
# rather than depending on the shell EXIT trap.
release_bootstrap_lock
exec /app/.venv/bin/uvicorn voice_rag.api.main:app --host 0.0.0.0 --port 8000
