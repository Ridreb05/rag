"""End-to-end retrieval evaluation on a closed-world subset of a language's
validation corpus: samples N queries, indexes exactly the passages those
queries reference (dense + BGE-sparse via Qdrant, BM25 via Tantivy), runs
real hybrid search + RRF fusion for each query, and scores against the
dataset's own ``is_selected`` qrels with Recall@K / MRR@K / NDCG@K / Hit
Rate@K (evaluation/retrieval_metrics.py).

This is NOT a full-corpus evaluation — embedding the full ~965K-chunk Hindi
corpus takes ~3.7 hours on this machine's GPU (measured: 72 chunks/sec).
This script validates the whole retrieval pipeline for real, on real
labels, at a scale that fits in an interactive run; scaling to the full
corpus is the same code path pointed at more data (see
docs/09-roadmap-and-summary.md Phase 11).

Usage:
    uv run python -m evaluation.run_subset_eval --language hi --n-queries 300
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from voice_rag.embeddings.service import EmbeddingService
from voice_rag.reranking.service import RerankerService
from voice_rag.retrieval.dense.index import IndexedChunk, collection_name, ensure_collection, get_client, upsert_chunks
from voice_rag.retrieval.fusion.rrf import reciprocal_rank_fusion
from voice_rag.retrieval.sparse.bm25_index import Bm25Index
from evaluation.retrieval_metrics import aggregate, hit_rate_at_k, ndcg_at_k, reciprocal_rank, recall_at_k

logger = logging.getLogger(__name__)
PROCESSED_DIR = Path("data/processed")


def build_eval_subset(lang: str, split: str, n_queries: int, seed: int = 42):
    queries_df = pd.read_parquet(PROCESSED_DIR / lang / f"{split}_queries.parquet")
    passages_df = pd.read_parquet(PROCESSED_DIR / lang / f"{split}_passages.parquet")
    qrels_df = pd.read_parquet(PROCESSED_DIR / lang / f"{split}_qrels.parquet")
    chunks_df = pd.read_parquet(PROCESSED_DIR / lang / f"{split}_chunks.parquet")

    rng = np.random.default_rng(seed)
    sampled_qids = rng.choice(queries_df["query_id"].unique(), size=min(n_queries, len(queries_df)), replace=False)
    sampled_queries = queries_df[queries_df["query_id"].isin(sampled_qids)].reset_index(drop=True)
    subset_qrels = qrels_df[qrels_df["query_id"].isin(sampled_qids)].reset_index(drop=True)

    subset_passage_ids = set(subset_qrels["passage_id"])
    subset_passages = passages_df[passages_df["passage_id"].isin(subset_passage_ids)].reset_index(drop=True)
    subset_chunks = chunks_df[chunks_df["passage_id"].isin(subset_passage_ids)].reset_index(drop=True)

    logger.info(
        "Subset: %d queries, %d qrel rows, %d unique passages, %d chunks",
        len(sampled_queries),
        len(subset_qrels),
        len(subset_passages),
        len(subset_chunks),
    )
    return sampled_queries, subset_passages, subset_chunks, subset_qrels


def chunk_id_to_passage_id(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0]


def dedup_ranking_to_passages(ranked_chunk_ids: list[str]) -> list[str]:
    seen: list[str] = []
    seen_set: set[str] = set()
    for cid in ranked_chunk_ids:
        pid = chunk_id_to_passage_id(cid)
        if pid not in seen_set:
            seen.append(pid)
            seen_set.add(pid)
    return seen


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n-queries", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=20, help="candidates retrieved per signal before fusion")
    parser.add_argument("--eval-k", type=int, default=10, help="K for Recall@K/NDCG@K after fusion+dedup")
    parser.add_argument(
        "--rerank", action="store_true", help="also score a reranked ordering for direct before/after comparison"
    )
    parser.add_argument("--rerank-candidates", type=int, default=20, help="how many fused candidates to rerank")
    args = parser.parse_args()

    queries_df, passages_df, chunks_df, qrels_df = build_eval_subset(args.language, args.split, args.n_queries)
    passage_text_by_id = dict(zip(passages_df["passage_id"], passages_df["text_translated"], strict=True))

    logger.info("Loading embedding service...")
    svc = EmbeddingService()
    reranker = RerankerService() if args.rerank else None

    logger.info("Embedding %d chunks...", len(chunks_df))
    t0 = time.time()
    chunk_texts = chunks_df["text"].tolist()
    result = svc.embed(chunk_texts, batch_size=64)
    logger.info("Embedded in %.1fs", time.time() - t0)

    tmp_dir = tempfile.mkdtemp(prefix="voice_rag_subset_eval_")
    try:
        client = get_client(path=str(Path(tmp_dir) / "qdrant"))
        coll = collection_name(args.language, "subset_eval")
        ensure_collection(client, coll, recreate=True)

        indexed_chunks = [
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
        n = upsert_chunks(client, coll, indexed_chunks)
        logger.info("Upserted %d points into Qdrant collection %s", n, coll)

        bm25 = Bm25Index(Path(tmp_dir) / "bm25")
        bm25.build(chunks_df["chunk_id"].tolist(), chunks_df["text"].tolist())
        logger.info("Built BM25 index")

        qrels_by_query = qrels_df.groupby("query_id")
        relevant_by_query: dict[int, set[str]] = {}
        for qid, group in qrels_by_query:
            relevant_by_query[qid] = set(group[group["is_selected"] == 1]["passage_id"])

        recalls, mrrs, ndcgs, hits = [], [], [], []
        rerank_recalls, rerank_mrrs, rerank_ndcgs, rerank_hits = [], [], [], []
        for row in queries_df.itertuples():
            qid = row.query_id
            query_text = row.query_text
            relevant = relevant_by_query.get(qid, set())

            q_embed = svc.embed_query(query_text)
            dense_vec = q_embed.dense[0].tolist()
            sparse_vec = q_embed.sparse[0]

            dense_hits = client.query_points(
                collection_name=coll, query=dense_vec, using="dense", limit=args.top_k
            ).points
            sparse_hits = client.query_points(
                collection_name=coll,
                query=models_sparse_vector(sparse_vec),
                using="sparse",
                limit=args.top_k,
            ).points
            bm25_hits = bm25.search(query_text, top_k=args.top_k)

            dense_ranked = [h.payload["chunk_id"] for h in dense_hits]
            sparse_ranked = [h.payload["chunk_id"] for h in sparse_hits]
            bm25_ranked = [cid for cid, _ in bm25_hits]

            fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked, bm25_ranked])
            fused_chunk_ids = [cid for cid, _ in fused]
            fused_passage_ids = dedup_ranking_to_passages(fused_chunk_ids)

            recalls.append(recall_at_k(fused_passage_ids, relevant, args.eval_k))
            mrrs.append(reciprocal_rank(fused_passage_ids, relevant))
            ndcgs.append(ndcg_at_k(fused_passage_ids, relevant, args.eval_k))
            hits.append(hit_rate_at_k(fused_passage_ids, relevant, 5))

            if reranker is not None:
                top_n = fused_passage_ids[: args.rerank_candidates]
                candidate_texts = [passage_text_by_id[pid] for pid in top_n]
                scores = reranker.rerank(query_text, candidate_texts)
                reranked_order = [pid for pid, _ in sorted(zip(top_n, scores, strict=True), key=lambda x: x[1], reverse=True)]
                # anything beyond rerank_candidates keeps its original fused order, appended after
                reranked_passage_ids = reranked_order + fused_passage_ids[args.rerank_candidates :]

                rerank_recalls.append(recall_at_k(reranked_passage_ids, relevant, args.eval_k))
                rerank_mrrs.append(reciprocal_rank(reranked_passage_ids, relevant))
                rerank_ndcgs.append(ndcg_at_k(reranked_passage_ids, relevant, args.eval_k))
                rerank_hits.append(hit_rate_at_k(reranked_passage_ids, relevant, 5))

        n_zero_relevant = sum(1 for v in recalls if v != v)
        print()
        print(f"=== Subset retrieval evaluation: {args.language} ({len(queries_df)} queries) ===")
        print(f"Queries with zero relevant passages (excluded from scoring): {n_zero_relevant}")
        print()
        print("--- Fused (dense + BGE-sparse + BM25 via RRF), before reranking ---")
        print(f"Recall@{args.eval_k}: {aggregate(recalls):.4f}")
        print(f"MRR:        {aggregate(mrrs):.4f}")
        print(f"NDCG@{args.eval_k}:   {aggregate(ndcgs):.4f}")
        print(f"HitRate@5:  {aggregate(hits):.4f}")
        if reranker is not None:
            print()
            print(f"--- After bge-reranker-v2-m3 reranking (top {args.rerank_candidates} candidates) ---")
            print(f"Recall@{args.eval_k}: {aggregate(rerank_recalls):.4f}")
            print(f"MRR:        {aggregate(rerank_mrrs):.4f}")
            print(f"NDCG@{args.eval_k}:   {aggregate(rerank_ndcgs):.4f}")
            print(f"HitRate@5:  {aggregate(rerank_hits):.4f}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def models_sparse_vector(sparse: dict[int, float]):
    from qdrant_client import models

    return models.SparseVector(indices=list(sparse.keys()), values=list(sparse.values()))


if __name__ == "__main__":
    main()
