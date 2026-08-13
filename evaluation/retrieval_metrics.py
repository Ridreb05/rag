"""Retrieval evaluation metrics (docs/07-observability-and-evaluation.md).

Pure functions operating on qrels (query_id -> set of relevant ids) and a
predicted ranking (list of ids, best first) — no dependency on the index or
embedding service, so these are cheap to unit-test in isolation and reused
identically by both the offline evaluation harness and the latency
benchmark's quality checks.
"""

from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return float("nan")  # query has no relevant items — exclude from aggregation, don't score as 0
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def hit_rate_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    return 1.0 if set(ranked_ids[:k]) & relevant_ids else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return float("nan")
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    dcg = 0.0
    for i, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(values: list[float]) -> float:
    """Mean over non-NaN values (NaN marks zero-relevant queries, excluded per docs/01)."""
    clean = [v for v in values if v == v]  # filters NaN
    return sum(clean) / len(clean) if clean else float("nan")
