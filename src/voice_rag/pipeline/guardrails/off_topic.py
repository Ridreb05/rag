"""Off-topic / low-confidence gate.

Two independent, cheap signals — no model call beyond the query embedding
the retrieval stage computes anyway:
1. Cosine distance from the query embedding to the indexed corpus's
   centroid (catches queries entirely outside the knowledge base's topic).
2. Top retrieval/rerank confidence score (catches queries that are
   topically plausible but not actually answered by anything indexed).

Either signal failing routes to the guardrail's fixed refusal rather than
invoking the generator.
"""

from __future__ import annotations

import numpy as np


def compute_corpus_centroid(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalized mean of a sample of indexed dense embeddings.

    A single centroid is a coarse approximation (a multi-topic corpus would
    be better served by k-means over several cluster centroids, taking the
    max similarity across clusters) — adequate as a first-pass "is this
    query anywhere near the corpus at all" check; upgrading to multi-cluster
    is a drop-in change to ``OffTopicGate.similarity``.
    """
    if embeddings.shape[0] == 0:
        raise ValueError("cannot compute a centroid from zero embeddings")
    mean = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


class OffTopicGate:
    def __init__(self, centroid: np.ndarray, similarity_threshold: float = 0.35):
        self.centroid = centroid
        self.similarity_threshold = similarity_threshold

    def similarity(self, query_embedding: np.ndarray) -> float:
        norm = np.linalg.norm(query_embedding)
        if norm == 0:
            return 0.0
        return float(np.dot(query_embedding, self.centroid) / norm)

    def is_off_topic(self, query_embedding: np.ndarray) -> bool:
        return self.similarity(query_embedding) < self.similarity_threshold


REFUSAL_MESSAGE = "I couldn't find enough information in the available knowledge base to answer that reliably."


def should_refuse(
    off_topic_gate: OffTopicGate,
    query_embedding: np.ndarray,
    retrieval_confidence: float,
    confidence_threshold: float = 0.2,
) -> bool:
    """Combines both signals: either one failing triggers refusal."""
    return off_topic_gate.is_off_topic(query_embedding) or retrieval_confidence < confidence_threshold
