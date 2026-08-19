#!/bin/bash
# Build indices for multiple languages sequentially.
# Run on a machine with a GPU (RTX 4090, 5070 Ti, etc.)
#
# Usage:
#   bash scripts/build_multi_language_indices.sh en bn
#   # or with defaults:
#   bash scripts/build_multi_language_indices.sh
#
# Default languages: hi bn en

set -eu

LANGUAGES="${@:-hi bn en}"
INDEX_VERSION="${INDEX_VERSION:-full1}"
INDEX_SPLIT="${INDEX_SPLIT:-validation}"

echo "Building indices for languages: $LANGUAGES"
echo "Index version: $INDEX_VERSION"
echo "Index split: $INDEX_SPLIT"
echo ""

for lang in $LANGUAGES; do
  echo "=========================================="
  echo "Building index for: $lang"
  echo "=========================================="
  echo ""

  echo "[1/3] Downloading and processing corpus..."
  uv run python -m voice_rag.pipeline.ingestion.build_corpus \
    --languages "$lang" \
    --split "$INDEX_SPLIT"

  echo ""
  echo "[2/3] Building chunks..."
  uv run python -m voice_rag.pipeline.chunking.build_chunks \
    --languages "$lang" \
    --split "$INDEX_SPLIT"

  echo ""
  echo "[3/3] Building full index (dense + sparse + BM25)..."
  echo "This will take 2-4 hours on a GPU. Get coffee ☕"
  echo ""
  uv run python scripts/build_full_index.py \
    --language "$lang" \
    --split "$INDEX_SPLIT" \
    --index-version "$INDEX_VERSION"

  echo ""
  echo "✅ Index built for $lang"
  echo "📍 Location: data/full_index/${lang}_${INDEX_SPLIT}_${INDEX_VERSION}*"
  echo ""
done

echo "=========================================="
echo "✅ All indices built successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload to RunPod network volume:"
echo "   rsync -avz data/full_index/ /path/to/runpod/volume/data/full_index/"
echo ""
echo "2. Upload processed data:"
echo "   rsync -avz data/processed/ /path/to/runpod/volume/data/processed/"
echo ""
echo "3. Launch RunPod pods for each language with VOICE_RAG_LANGUAGE env var"
echo ""
