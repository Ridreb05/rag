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

  # CUDA graphs on by default. --enforce-eager was set for the first boots to
  # minimise memory and startup risk while the image was still failing for
  # other reasons; with startup now proven it is the single largest cost in
  # generation. Measured on this Pod with eager: 634ms to generate 24 tokens,
  # i.e. ~40 tok/s against a ~126 tok/s memory-bandwidth ceiling for a 4B bf16
  # model on a 4090 — only ~32% of what the hardware allows. That gap is
  # per-token Python and kernel-launch overhead, which is exactly what graph
  # capture removes. Set VOICE_RAG_VLLM_ENFORCE_EAGER=1 to restore the old
  # behaviour without a rebuild if graph capture ever exhausts GPU memory.
  vllm_eager_flag=""
  if [ "${VOICE_RAG_VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
    vllm_eager_flag="--enforce-eager"
  fi

  # FP8 weights, because bf16 cannot meet the latency target on this model and
  # no amount of flag tuning changes that. Decoding is memory-bandwidth bound:
  # every token reads the entire weight set from HBM, so a 4B bf16 model moves
  # 8GB/token, capping a 4090 (~1008 GB/s) at ~126 tok/s. Against a ~144ms
  # generation budget that is under 11 tokens — too few for a Hindi sentence.
  # FP8 halves the bytes moved per token, roughly doubling the ceiling to
  # ~250 tok/s and making ~20 tokens fit. Ada (SM 8.9) has native FP8 tensor
  # cores, so this is hardware-supported rather than emulated.
  #
  # Quality cost is real but small for W8A8 on a model this size, and is the
  # right trade here: the alternative that also fits the budget is a
  # substantially smaller model, which would cost more quality than 8-bit
  # weights do. Set VOICE_RAG_VLLM_QUANTIZATION="" to serve bf16 instead.
  vllm_quant_flag=""
  vllm_quant="${VOICE_RAG_VLLM_QUANTIZATION-fp8}"
  if [ -n "$vllm_quant" ]; then
    vllm_quant_flag="--quantization $vllm_quant"
  fi

  # --max-num-seqs 8, well below the default: this deployment serves one query
  # at a time, and the number of CUDA graphs captured at startup scales with
  # this. Fewer graphs is less capture time and less memory held for batch
  # sizes that will never occur.
  #
  # --enable-prefix-caching: every request shares an identical system prompt,
  # so its prefill is recomputed on each query for no reason. Caching it is
  # small (the passages and question still change) but free.

  echo "Starting vLLM ($vllm_model) on port $vllm_port."
  (
    exec /opt/vllm-venv/bin/vllm serve "$vllm_model" \
      --port "$vllm_port" \
      --dtype bfloat16 \
      --max-model-len "${VOICE_RAG_VLLM_MAX_MODEL_LEN:-4096}" \
      --gpu-memory-utilization "${VOICE_RAG_VLLM_GPU_MEM_FRACTION:-0.55}" \
      --enable-prefix-caching \
      --max-num-seqs "${VOICE_RAG_VLLM_MAX_NUM_SEQS:-8}" \
      $vllm_quant_flag \
      $vllm_eager_flag
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

# One or more languages to bootstrap, comma-separated (e.g. "hi,bn,en").
# Falls back to the single VOICE_RAG_LANGUAGE value when unset -- today's
# exact single-language behavior, unchanged. Each language bootstraps
# sequentially against the one running Qdrant server: a real server process
# handles concurrent collections fine, but there is only one GPU here, so
# running these one after another (not in parallel) is the actual bottleneck,
# not a safety requirement.
bootstrap_languages="${VOICE_RAG_BOOTSTRAP_LANGUAGES:-$index_language}"

# Keep this enabled for the first boot and later restarts. The index builder
# verifies its state and exits quickly for any language whose version is
# already complete -- so re-running this list on a restart is cheap for
# languages already built and only does real work for the ones that aren't.
if [ "${VOICE_RAG_BOOTSTRAP_INDEX:-0}" = "1" ]; then
  built_languages=""
  old_ifs="$IFS"
  IFS=','
  for lang in $bootstrap_languages; do
    IFS="$old_ifs"
    lang=$(printf '%s' "$lang" | tr -d '[:space:]')
    if [ -z "$lang" ]; then
      continue
    fi

    bootstrap_marker="$data_root/data/full_index/${lang}_${index_split}_${index_version}_bootstrap_complete"
    bootstrap_lock_dir="$data_root/data/full_index/.${lang}_${index_split}_${index_version}.bootstrap.lock"
    lang_state_path="$data_root/data/full_index/${lang}_${index_split}_${index_version}.state.json"

    if ! mkdir "$bootstrap_lock_dir" 2>/dev/null; then
      # A forced Pod stop can leave this empty directory on the persistent
      # volume. Recovery is deliberately opt-in: automatically removing it
      # could allow two Pods to write the same BM25 index concurrently.
      if [ "${VOICE_RAG_RECOVER_STALE_BOOTSTRAP_LOCK:-0}" = "1" ]; then
        echo "[$lang] Recovering explicitly approved stale bootstrap lock: $bootstrap_lock_dir"
        rm -f "$bootstrap_lock_dir/owner"
        if ! rmdir "$bootstrap_lock_dir" 2>/dev/null; then
          echo "[$lang] Bootstrap lock is still active or not empty: $bootstrap_lock_dir" >&2
          exit 1
        fi
        if ! mkdir "$bootstrap_lock_dir" 2>/dev/null; then
          echo "[$lang] Bootstrap lock is still active or not empty: $bootstrap_lock_dir" >&2
          exit 1
        fi
      else
        echo "[$lang] Another bootstrap may be using $bootstrap_lock_dir; refusing to corrupt the shared index." >&2
        echo "If every other Pod using this volume is stopped, set VOICE_RAG_RECOVER_STALE_BOOTSTRAP_LOCK=1 for one restart." >&2
        exit 1
      fi
    fi
    printf '%s\n' "pid=$$ started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$bootstrap_lock_dir/owner"

    chunks_path="$data_root/data/processed/${lang}/${index_split}_chunks.parquet"
    passages_path="$data_root/data/processed/${lang}/${index_split}_passages.parquet"
    if [ ! -f "$chunks_path" ]; then
      if [ ! -f "$passages_path" ]; then
        echo "[$lang] Downloading and normalizing the corpus."
        /app/.venv/bin/python -m voice_rag.pipeline.ingestion.build_corpus --languages "$lang" --split "$index_split"
      else
        echo "[$lang] Reusing persisted normalized corpus."
      fi
      echo "[$lang] Building chunks from the persisted corpus."
      /app/.venv/bin/python -m voice_rag.pipeline.chunking.build_chunks --languages "$lang" --split "$index_split"
    else
      echo "[$lang] Reusing persisted chunks."
    fi

    # An old bootstrap has no state manifest, so its partial Qdrant/BM25 data
    # cannot be trusted. Reset it once. New bootstraps write a manifest after
    # every committed batch and resume without discarding completed work.
    set -- /app/.venv/bin/python scripts/build_full_index.py \
      --language "$lang" \
      --split "$index_split" \
      --index-version "$index_version" \
      --qdrant-url "$QDRANT_URL" \
      --optimizer-wait-seconds "${VOICE_RAG_OPTIMIZER_WAIT_SECONDS:-900}"
    if [ ! -f "$lang_state_path" ]; then
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
    echo "[$lang] Starting resumable index bootstrap. State: $lang_state_path"
    "$@"
    touch "$bootstrap_marker"
    release_bootstrap_lock
    echo "[$lang] Index bootstrap completed."

    if [ -z "$built_languages" ]; then
      built_languages="$lang"
    else
      built_languages="$built_languages,$lang"
    fi
  done
  IFS="$old_ifs"

  # Serve every language just bootstrapped, unless the operator already
  # pinned VOICE_RAG_LANGUAGES explicitly -- an explicit value always wins,
  # e.g. to bootstrap a language without yet exposing it to traffic. When
  # only one language was bootstrapped this stays unset, so main.py takes
  # its existing single-language path (VOICE_RAG_LANGUAGE alone) -- today's
  # exact behavior, unchanged.
  case "$built_languages" in
    *,*)
      : "${VOICE_RAG_LANGUAGES:=$built_languages}"
      export VOICE_RAG_LANGUAGES
      echo "Serving languages: $VOICE_RAG_LANGUAGES"
      ;;
  esac
fi

echo "Starting Voice RAG API on port 8000."
# ``exec`` replaces this shell, so release the bootstrap lock explicitly
# rather than depending on the shell EXIT trap.
release_bootstrap_lock
exec /app/.venv/bin/uvicorn voice_rag.api.main:app --host 0.0.0.0 --port 8000
