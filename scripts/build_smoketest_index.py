"""Builds a small, persistent, isolated index for testing the API gateway
without touching the live full-corpus index being built by
scripts/build_full_index.py — see docs/evaluation-results.md's third bug
(concurrent access to a local-mode Qdrant path corrupts the writer).

Usage:
    uv run python scripts/build_smoketest_index.py --language hi --n-queries 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_rag.pipeline.embeddings.service import EmbeddingService
from voice_rag.pipeline.retrieval.dense.index import IndexedChunk, collection_name, ensure_collection, get_client, upsert_chunks
from voice_rag.pipeline.retrieval.sparse.bm25_index import Bm25Index

PROCESSED_DIR = Path("data/processed")
OUT_DIR = Path("data/api_smoketest")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="hi")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n-rows", type=int, default=2000, help="chunks to index")
    parser.add_argument("--index-version", default="smoketest")
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Point at a real Qdrant server instead of embedded local mode. Verified directly "
        "(docs/evaluation-results.md): local mode's sparse-vector search is far slower than the "
        "real server at even moderate scale (~600ms vs low-single-digit ms at 40K points) — use "
        "this for any latency-sensitive measurement.",
    )
    args = parser.parse_args()

    chunks_df = pd.read_parquet(PROCESSED_DIR / args.language / f"{args.split}_chunks.parquet")
    chunks_df = chunks_df.sample(n=min(args.n_rows, len(chunks_df)), random_state=7).reset_index(drop=True)
    log.info("Indexing %d chunks for smoketest", len(chunks_df))

    svc = EmbeddingService()
    result = svc.embed(chunks_df["text"].tolist(), batch_size=64)

    qdrant_path = OUT_DIR / "qdrant"
    client = get_client(path=None if args.qdrant_url else str(qdrant_path), url=args.qdrant_url)
    coll = collection_name(args.language, args.index_version)
    ensure_collection(client, coll, recreate=True)

    indexed = [
        IndexedChunk(
            chunk_id=row.chunk_id,
            passage_id=row.passage_id,
            language=row.language,
            text=row.text,
            dense_vector=result.dense[i].tolist(),
            sparse_vector=result.sparse[i],
        )
        for i, row in enumerate(chunks_df.itertuples())
    ]
    upsert_chunks(client, coll, indexed)
    log.info("Upserted %d points into %s at %s", len(indexed), coll, qdrant_path)

    bm25_path = OUT_DIR / "bm25" / f"{args.language}_{args.split}"
    bm25 = Bm25Index(bm25_path)
    bm25.build(chunks_df["chunk_id"].tolist(), chunks_df["text"].tolist())
    log.info("Built BM25 index at %s", bm25_path)


if __name__ == "__main__":
    main()
