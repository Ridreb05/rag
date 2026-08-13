"""Reciprocal Rank Fusion — docs/02-architecture-and-retrieval.md's chosen
fusion strategy. Chosen over weighted score fusion because dense cosine,
BGE-M3 sparse scores, and BM25 scores live on incompatible scales; RRF only
needs rank order, so no per-language score calibration is required.
"""

from __future__ import annotations


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """``ranked_lists`` is a list of result lists (each best-first, containing
    doc/chunk IDs). Returns a single fused ranking as [(id, score), ...]
    sorted best-first."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
