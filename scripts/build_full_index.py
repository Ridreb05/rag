"""Full-corpus embedding + indexing (background batch job, not interactive).

Measured throughput: 72 chunks/sec on this machine's GPU (RTX 4060 Laptop,
8GB VRAM) at batch_size=64. Embedding the full ~965K-chunk Hindi validation
corpus takes ~3.7 hours — this script is designed to run unattended:
persistent (not temp-dir) Qdrant + BM25 indexes, periodic progress logging,
and incremental Qdrant upserts so a crash partway through doesn't lose all
prior work (re-running is also safe — chunk_id-derived point IDs make
upserts idempotent, see voice_rag/retrieval/dense/index.py).

Usage:
    uv run python scripts/build_full_index.py --language hi --split validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_rag.embeddings.service import EmbeddingService
from voice_rag.retrieval.dense.index import IndexedChunk, collection_name, ensure_collection, get_client, upsert_chunks
from voice_rag.retrieval.sparse.bm25_index import Bm25Index

logger = logging.getLogger(__name__)
PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/full_index")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("data/full_index_build.log", encoding="utf-8")],
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embed-chunk-size", type=int, default=5000, help="rows embedded+upserted per progress step")
    parser.add_argument("--index-version", default="full1")
    parser.add_argument("--limit", type=int, default=None, help="cap total rows processed (smoke-testing only)")
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Point at a real Qdrant server instead of embedded local mode — required for deployment "
        "(docs/runpod-deployment.md): local mode's SQLite-backed storage is not the same on-disk "
        "format as the server and cannot be copied into one.",
    )
    args = parser.parse_args()

    chunks_path = PROCESSED_DIR / args.language / f"{args.split}_chunks.parquet"
    chunks_df = pd.read_parquet(chunks_path)
    if args.limit:
        chunks_df = chunks_df.iloc[: args.limit].reset_index(drop=True)
    total = len(chunks_df)
    logger.info("Loaded %d chunks from %s", total, chunks_path)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = INDEX_DIR / f"{args.language}_{args.split}_progress.json"

    logger.info("Loading embedding service...")
    svc = EmbeddingService()

    qdrant_path = str(INDEX_DIR / "qdrant")
    client = get_client(path=None if args.qdrant_url else qdrant_path, url=args.qdrant_url)
    coll = collection_name(args.language, args.index_version)
    ensure_collection(client, coll, recreate=False)

    bm25_path = INDEX_DIR / "bm25" / f"{args.language}_{args.split}"
    bm25 = Bm25Index(bm25_path)
    bm25_writer_chunk_ids: list[str] = []
    bm25_writer_texts: list[str] = []

    start_time = time.time()
    n_done = 0
    for start in range(0, total, args.embed_chunk_size):
        batch_df = chunks_df.iloc[start : start + args.embed_chunk_size]
        texts = batch_df["text"].tolist()

        result = svc.embed(texts, batch_size=args.batch_size)

        indexed_chunks = [
            IndexedChunk(
                chunk_id=row.chunk_id,
                passage_id=row.passage_id,
                language=row.language,
                text=row.text,
                dense_vector=result.dense[i].tolist(),
                sparse_vector=result.sparse[i],
            )
            for i, row in enumerate(batch_df.itertuples())
        ]
        upsert_chunks(client, coll, indexed_chunks)

        bm25_writer_chunk_ids.extend(batch_df["chunk_id"].tolist())
        bm25_writer_texts.extend(texts)

        n_done += len(batch_df)
        elapsed = time.time() - start_time
        rate = n_done / elapsed if elapsed > 0 else 0
        eta_seconds = (total - n_done) / rate if rate > 0 else float("inf")
        logger.info(
            "Progress: %d/%d (%.1f%%) | rate=%.1f chunks/sec | elapsed=%.0fs | eta=%.0fs (%.1f min)",
            n_done,
            total,
            100 * n_done / total,
            rate,
            elapsed,
            eta_seconds,
            eta_seconds / 60,
        )
        progress_path.write_text(
            json.dumps(
                {
                    "language": args.language,
                    "split": args.split,
                    "n_done": n_done,
                    "total": total,
                    "rate_chunks_per_sec": rate,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta_seconds,
                    "done": n_done >= total,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    logger.info("Building BM25 index over all %d chunks...", len(bm25_writer_chunk_ids))
    bm25.build(bm25_writer_chunk_ids, bm25_writer_texts)
    logger.info("Done. Qdrant collection=%s at %s, BM25 index at %s", coll, qdrant_path, bm25_path)


if __name__ == "__main__":
    main()
