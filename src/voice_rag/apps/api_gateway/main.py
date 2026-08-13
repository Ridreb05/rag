"""API gateway (docs/06-data-and-api.md) — the deployable FastAPI app.

Wires the already-built retrieval pipeline (embeddings, Qdrant hybrid
search, BM25, reranker) and the generation harness together behind
`POST /v1/query`. All model services load once at startup (warm worker
pattern, docs/04-latency-and-caching.md) rather than per-request.

Deployment: see docs/runpod-deployment.md. Points at a real Qdrant server
via QDRANT_URL when set (the containerized/production path); falls back to
local embedded mode via QDRANT_PATH for local dev.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import models as qdrant_models

from voice_rag.apps.api_gateway.rate_limit import RateLimitMiddleware
from voice_rag.embeddings.service import EmbeddingService
from voice_rag.generation.anthropic_service import AnthropicGenerationService
from voice_rag.generation.gemini_service import GeminiGenerationService
from voice_rag.generation.harness import GenerationHarness
from voice_rag.generation.schemas import RetrievalCandidate
from voice_rag.guardrails.grounding import GroundingValidator
from voice_rag.guardrails.off_topic import OffTopicGate, compute_corpus_centroid
from voice_rag.reranking.service import RerankerService
from voice_rag.retrieval.dense.index import collection_name, get_client
from voice_rag.retrieval.fusion.rrf import reciprocal_rank_fusion
from voice_rag.retrieval.sparse.bm25_index import Bm25Index

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = os.environ.get("VOICE_RAG_LANGUAGE", "hi")
INDEX_VERSION = os.environ.get("VOICE_RAG_INDEX_VERSION", "full1")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "data/full_index/qdrant")
BM25_PATH = os.environ.get("BM25_PATH", f"data/full_index/bm25/{DEFAULT_LANGUAGE}_validation")
LOAD_GROUNDING_VALIDATOR = os.environ.get("VOICE_RAG_LOAD_GROUNDING_VALIDATOR", "1") == "1"
# "gemini" is the working default — the Anthropic integration is code-verified
# correct (docs/evaluation-results.md) but blocked on account credits as of
# this writing. Flip via env once that's resolved; the harness doesn't care
# which backend it gets, both implement the same generate() interface.
GENERATION_BACKEND = os.environ.get("VOICE_RAG_GENERATION_BACKEND", "gemini")
TOP_K_PER_SIGNAL = 20
RERANK_CANDIDATES = 20


class Services:
    embedding: EmbeddingService
    reranker: RerankerService
    harness: GenerationHarness
    bm25: Bm25Index
    qdrant_client: object
    collection: str
    off_topic_gate: OffTopicGate | None


services = Services()


def _try_build_off_topic_gate(sample_size: int = 2000) -> OffTopicGate | None:
    """Samples dense vectors already in the index to compute a corpus
    centroid (docs/11's off-topic gate) — no separate offline job needed
    since the index itself is the source of truth for "what's in scope."
    Returns None (gate disabled) if the collection doesn't exist yet or is
    empty, e.g. before the background indexing job has produced any points."""
    import numpy as np

    try:
        points, _ = services.qdrant_client.scroll(
            collection_name=services.collection, limit=sample_size, with_vectors=["dense"]
        )
    except Exception:
        logger.warning("Off-topic gate disabled: collection %s not ready", services.collection)
        return None
    if not points:
        logger.warning("Off-topic gate disabled: collection %s is empty", services.collection)
        return None
    vectors = np.array([p.vector["dense"] for p in points if p.vector and "dense" in p.vector])
    if vectors.shape[0] == 0:
        return None
    centroid = compute_corpus_centroid(vectors)
    logger.info("Off-topic gate built from %d sampled points", vectors.shape[0])
    return OffTopicGate(centroid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading warm model pool...")
    services.embedding = EmbeddingService()
    services.reranker = RerankerService()
    grounding_validator = GroundingValidator() if LOAD_GROUNDING_VALIDATOR else None
    generator = GeminiGenerationService() if GENERATION_BACKEND == "gemini" else AnthropicGenerationService()
    services.harness = GenerationHarness(generator=generator, grounding_validator=grounding_validator)
    services.qdrant_client = get_client(path=None if QDRANT_URL else QDRANT_PATH, url=QDRANT_URL)
    services.collection = collection_name(DEFAULT_LANGUAGE, INDEX_VERSION)
    services.bm25 = Bm25Index(BM25_PATH)
    services.off_topic_gate = _try_build_off_topic_gate()
    logger.info("Warm pool ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Voice RAG API", lifespan=lifespan)
app.add_middleware(RateLimitMiddleware, max_requests=20, window_seconds=60.0)


class QueryRequest(BaseModel):
    query: str
    language: str = DEFAULT_LANGUAGE
    top_k: int = 10


class EvidenceItem(BaseModel):
    chunk_id: str
    text: str
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    trace_id: str
    answer_text: str
    mode: str
    confidence: float
    guardrail_flags: list[str]
    evidence: list[EvidenceItem]
    latency_ms: dict[str, float]


def _sparse_vector(sparse: dict[int, float]) -> qdrant_models.SparseVector:
    return qdrant_models.SparseVector(indices=list(sparse.keys()), values=list(sparse.values()))


@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    trace_id = str(uuid.uuid4())
    t_start = time.time()
    timings: dict[str, float] = {}

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if len(req.query) > 2000:
        raise HTTPException(status_code=400, detail="query too long (max 2000 chars)")

    t0 = time.time()
    q_embed = services.embedding.embed_query(req.query)
    timings["embedding_ms"] = (time.time() - t0) * 1000

    t0 = time.time()
    dense_hits = services.qdrant_client.query_points(
        collection_name=services.collection, query=q_embed.dense[0].tolist(), using="dense", limit=TOP_K_PER_SIGNAL
    ).points
    sparse_hits = services.qdrant_client.query_points(
        collection_name=services.collection,
        query=_sparse_vector(q_embed.sparse[0]),
        using="sparse",
        limit=TOP_K_PER_SIGNAL,
    ).points
    bm25_hits = services.bm25.search(req.query, top_k=TOP_K_PER_SIGNAL)
    timings["retrieval_ms"] = (time.time() - t0) * 1000

    dense_ranked = [h.payload["chunk_id"] for h in dense_hits]
    sparse_ranked = [h.payload["chunk_id"] for h in sparse_hits]
    bm25_ranked = [cid for cid, _ in bm25_hits]
    payload_by_chunk_id = {h.payload["chunk_id"]: h.payload for h in [*dense_hits, *sparse_hits]}

    t0 = time.time()
    fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked, bm25_ranked])
    fused_chunk_ids = [cid for cid, _ in fused][:RERANK_CANDIDATES]
    timings["fusion_ms"] = (time.time() - t0) * 1000

    rerank_score_by_chunk_id: dict[str, float] = {}
    if not fused_chunk_ids:
        resp = services.harness.answer(trace_id, req.query, req.language, [])
    else:
        t0 = time.time()
        texts = [payload_by_chunk_id[cid]["text"] for cid in fused_chunk_ids if cid in payload_by_chunk_id]
        valid_chunk_ids = [cid for cid in fused_chunk_ids if cid in payload_by_chunk_id]
        scores = services.reranker.rerank(req.query, texts)
        timings["rerank_ms"] = (time.time() - t0) * 1000

        ranked = sorted(zip(valid_chunk_ids, texts, scores, strict=True), key=lambda x: x[2], reverse=True)
        candidates = [
            RetrievalCandidate(
                chunk_id=cid,
                doc_id=payload_by_chunk_id[cid].get("passage_id", cid),
                language=req.language,
                text=text,
                rerank_score=float(score),
            )
            for cid, text, score in ranked[: req.top_k]
        ]
        rerank_score_by_chunk_id = {c.chunk_id: c.rerank_score for c in candidates if c.rerank_score is not None}

        t0 = time.time()
        resp = services.harness.answer(trace_id, req.query, req.language, candidates, query_embedding=q_embed.dense[0])
        timings["generation_ms"] = (time.time() - t0) * 1000

    timings["total_ms"] = (time.time() - t_start) * 1000

    # Evidence resolves chunk_id -> the actual cited *passage* text, not the
    # claim text (docs/03's citation-carry-through design) — payload_by_chunk_id
    # holds the real passage text; Claim.text is the LLM's own claim wording,
    # which only happens to equal the passage on the extractive fast-path.
    seen_chunk_ids: set[str] = set()
    evidence: list[EvidenceItem] = []
    for c in resp.claims:
        for cid in c.cited_chunk_ids:
            if cid in seen_chunk_ids:
                continue
            seen_chunk_ids.add(cid)
            payload = payload_by_chunk_id.get(cid)
            evidence.append(
                EvidenceItem(
                    chunk_id=cid,
                    text=payload["text"] if payload else c.text,
                    rerank_score=rerank_score_by_chunk_id.get(cid),
                )
            )

    return QueryResponse(
        trace_id=trace_id,
        answer_text=resp.answer_text,
        mode=resp.mode,
        confidence=resp.confidence,
        guardrail_flags=resp.guardrail_flags,
        evidence=evidence,
        latency_ms=timings,
    )


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok"}
