"""The deployable FastAPI app — the API layer of the voice RAG pipeline.

Wires the retrieval pipeline (embeddings, Qdrant hybrid search, BM25,
reranker) and the generation harness together behind `POST /v1/query`. All
model services load once at startup (warm worker pattern) rather than
per-request.

Points at a real Qdrant server via QDRANT_URL when set (the
containerized/production path); falls back to local embedded mode via
QDRANT_PATH for local dev.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client import models as qdrant_models
from starlette.background import BackgroundTask

from voice_rag.api.rate_limit import RateLimitMiddleware
from voice_rag.pipeline.embeddings.service import EmbeddingService
from voice_rag.pipeline.generation.factory import build_generator
from voice_rag.pipeline.generation.harness import (
    DEADLINE_FALLBACK_FLAG,
    EXTRACTIVE_CONFIDENCE_THRESHOLD,
    GenerationHarness,
)
from voice_rag.pipeline.generation.schemas import RetrievalCandidate
from voice_rag.pipeline.guardrails.grounding import GroundingValidator
from voice_rag.pipeline.guardrails.off_topic import OffTopicGate, compute_corpus_centroid
from voice_rag.pipeline.reranking.service import RerankerService
from voice_rag.pipeline.retrieval.dense.index import collection_name, get_client, stable_point_id
from voice_rag.pipeline.retrieval.fusion.rrf import reciprocal_rank_fusion
from voice_rag.pipeline.retrieval.sparse.bm25_index import Bm25Index
from voice_rag.pipeline.stt.sarvam_client import SarvamSttClient

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = os.environ.get("VOICE_RAG_LANGUAGE", "hi")
INDEX_VERSION = os.environ.get("VOICE_RAG_INDEX_VERSION", "full1")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "data/full_index/qdrant")

# A process serves one language unless VOICE_RAG_LANGUAGES lists several
# (comma-separated). Unset, this reduces to exactly today's single-language
# behavior: SUPPORTED_LANGUAGES == [DEFAULT_LANGUAGE], and every per-language
# dict built below ends up with exactly one key computed by the same
# expressions the old scalars used.
_LANGUAGES_ENV = os.environ.get("VOICE_RAG_LANGUAGES")
if _LANGUAGES_ENV:
    SUPPORTED_LANGUAGES: list[str] = [lang.strip() for lang in _LANGUAGES_ENV.split(",") if lang.strip()]
    if not SUPPORTED_LANGUAGES:
        raise RuntimeError("VOICE_RAG_LANGUAGES is set but empty after parsing")
    # VOICE_RAG_LANGUAGE keeps meaning "the default when a caller omits
    # `language`" -- honor it if it names one of the configured languages,
    # else fall back to the first configured language rather than a bare
    # "hi" that might not even be in SUPPORTED_LANGUAGES.
    if "VOICE_RAG_LANGUAGE" not in os.environ or DEFAULT_LANGUAGE not in SUPPORTED_LANGUAGES:
        DEFAULT_LANGUAGE = SUPPORTED_LANGUAGES[0]
else:
    SUPPORTED_LANGUAGES = [DEFAULT_LANGUAGE]


def _bm25_path_for(language: str) -> str:
    override = os.environ.get(f"VOICE_RAG_BM25_PATH_{language.upper()}")
    if override:
        return override
    if language == DEFAULT_LANGUAGE and len(SUPPORTED_LANGUAGES) == 1:
        # Preserves the exact single-language env var the RunPod entrypoint
        # already sets (infrastructure/runpod-entrypoint.sh), so a
        # single-language deployment needs no changes to that script.
        return os.environ.get("BM25_PATH", f"data/full_index/bm25/{language}_validation_{INDEX_VERSION}")
    return f"data/full_index/bm25/{language}_validation_{INDEX_VERSION}"


def _index_state_path_for(language: str) -> Path:
    override = os.environ.get(f"VOICE_RAG_INDEX_STATE_PATH_{language.upper()}")
    if override:
        return Path(override)
    if language == DEFAULT_LANGUAGE and len(SUPPORTED_LANGUAGES) == 1:
        return Path(
            os.environ.get(
                "VOICE_RAG_INDEX_STATE_PATH",
                f"data/full_index/{language}_validation_{INDEX_VERSION}.state.json",
            )
        )
    return Path(f"data/full_index/{language}_validation_{INDEX_VERSION}.state.json")


BM25_PATHS: dict[str, str] = {lang: _bm25_path_for(lang) for lang in SUPPORTED_LANGUAGES}
INDEX_STATE_PATHS: dict[str, Path] = {lang: _index_state_path_for(lang) for lang in SUPPORTED_LANGUAGES}
# The RunPod entrypoint enables this only after it has completed a bootstrap.
# Keeping it opt-in preserves the lightweight local-dev workflow where a
# developer may deliberately point the API at a small hand-built collection.
REQUIRE_COMPLETE_INDEX = os.environ.get("VOICE_RAG_REQUIRE_COMPLETE_INDEX", "0") == "1"
# Whether /v1/health requires every configured language to be ready, or just
# one. Default "1" (all) matches a deployment probe's usual meaning -- "this
# pod is fully up". "0" is for a rolling multi-language bootstrap: the pod
# keeps accepting traffic for whichever languages are already ready while
# another language is still being populated in the background.
HEALTH_REQUIRE_ALL_LANGUAGES = os.environ.get("VOICE_RAG_HEALTH_REQUIRE_ALL_LANGUAGES", "1") == "1"
LOAD_GROUNDING_VALIDATOR = os.environ.get("VOICE_RAG_LOAD_GROUNDING_VALIDATOR", "1") == "1"
# Building this gate re-embeds a text sample locally at startup (see
# _try_build_off_topic_gate). On a throttled shared-CPU host that step can
# stall for many minutes with the process making real but very slow
# progress — not worth blocking startup for on a small demo index.
# Confidence-based refusal (LOW_CONFIDENCE_THRESHOLD, in the harness) stays
# active either way, so this only drops the centroid-based signal, not
# guardrail coverage entirely.
LOAD_OFF_TOPIC_GATE = os.environ.get("VOICE_RAG_LOAD_OFF_TOPIC_GATE", "1") == "1"
# Tuned down from an initial 20: reranking cost scales with candidate count,
# and dense/sparse/BM25 already agree closely on the top few candidates for
# a corpus this size, so a smaller K loses negligible recall for a real
# latency win (measured: K=20->10 roughly halved rerank latency).
TOP_K_PER_SIGNAL = 8
RERANK_CANDIDATES = 8
# Bounds BM25's contribution to the critical path. BM25 runs concurrently
# underneath the embedding call, so it costs ~0 whenever it finishes first —
# but Tantivy's OR-of-terms query (chunking/tokenizer's per-term postings
# union) costs roughly O(sum of matched postings-list lengths), and
# high-document-frequency Hindi function words occasionally push a single
# query's postings union into the hundreds of milliseconds (measured: p50
# ~15ms, p99 ~91-120ms, p100 ~460ms on the full 964,603-chunk index — a
# ~30x gap between typical and worst case). Past this timeout, BM25 is
# dropped for that request rather than blocking on it: dense+sparse still
# cover the query, so this trades a small amount of recall on a rare
# pathological query for a bounded worst case, the same trade already made
# for generation's retry budget.
BM25_TIMEOUT_SECONDS = 0.1
# Same bound, same reason, for Qdrant's sparse (learned-lexical) search.
# Measured on the full 964,603-chunk index: p50 6.5ms but p100 386.0ms — the
# single largest contributor to retrieval's worst case, and the same
# pathology as BM25's tail, since a learned-sparse query is also a union over
# per-term posting lists and high-document-frequency Hindi terms produce
# very long ones. Dense is deliberately left unbounded: it is the primary
# semantic signal and the only one guaranteed to return something for any
# query, and its own tail is far smaller (p100 106.3ms) because HNSW's cost
# does not scale with term frequency.
SPARSE_TIMEOUT_SECONDS = 0.1
# The task's end-to-end target. Treated as a real wall-clock deadline carried
# through the request rather than an aspiration measured after the fact: each
# stage past retrieval checks how much of the budget is actually left and
# picks the most complete answer that still fits. Every degradation step has
# a grounded answer to fall back to (the top reranked passage), so respecting
# the deadline costs answer *richness* on slow queries, never correctness.
#
# Environment-overridable because 200ms and a live LLM call are mutually
# exclusive (see MIN_GENERATION_BUDGET_SECONDS), and which of the two matters
# is a deployment question, not a code one:
#   0.2 (default) — enforce the latency target; answers stay extractive.
#   e.g. 10       — let generation run, for a walkthrough of the full
#                   generative path (structured output, grounding validation)
#                   where interactive latency is not what is being shown.
# Both are real configurations of the same pipeline; neither is a special
# "demo mode" that changes behaviour beyond this one budget.
REQUEST_BUDGET_SECONDS = float(os.environ.get("VOICE_RAG_REQUEST_BUDGET_SECONDS", "0.2"))
# Floor for attempting a generative answer. A model resident on this process's
# own GPU has no network hop, so generation fits inside the request — which is
# the entire reason for serving one locally. Measured on the deployment:
# retrieval completes in ~37ms, leaving ~163ms of the budget, and generation
# lands at ~150ms, so this threshold admits generation whenever there is
# realistically time for it.
#
# This gates *attempting* generation; it does not cap it. If generation runs
# slower than the remaining budget the request overruns rather than being
# cancelled mid-flight, so `pipeline_ms` above 200 with mode=generative means
# this number is too optimistic for the hardware and should be raised — which
# pushes generation back to phase two — not that generation is broken.
MIN_GENERATION_BUDGET_SECONDS = float(os.environ.get("VOICE_RAG_MIN_GENERATION_BUDGET_SECONDS", "0.12"))
# Retry/wall-clock budget for the refinement generator only — see where
# services.refine_harness is built for why this is not the in-request 8s.
REFINE_GENERATION_BUDGET_SECONDS = float(os.environ.get("VOICE_RAG_REFINE_GENERATION_BUDGET_SECONDS", "45"))
MAX_AUDIO_BYTES = 15 * 1024 * 1024
# Sarvam's BCP-47 codes; "or" -> "od-IN" is the one exception to the naive <code>-IN pattern.
# "en" -> "en-IN" is not one of MSMARCO-XI's 14 translated languages (it is the
# corpus's own source-language field, indexed as a pseudo-language — see
# ingestion/build_corpus.py's ENGLISH_PSEUDO_CODE), but Sarvam transcribes it
# under the same Indian-English code as every other language here.
SARVAM_BCP47 = {
    "as": "as-IN", "bn": "bn-IN", "en": "en-IN", "gu": "gu-IN", "hi": "hi-IN", "kn": "kn-IN",
    "ml": "ml-IN", "mr": "mr-IN", "ne": "ne-IN", "or": "od-IN", "pa": "pa-IN",
    "sa": "sa-IN", "ta": "ta-IN", "te": "te-IN", "ur": "ur-IN",
}

# Dense/sparse (2 network round trips to Qdrant) and BM25 (local, embedded)
# are independent given the query embedding — run them concurrently instead
# of paying their latencies serially.
_retrieval_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="retrieval")


class Services:
    embedding: EmbeddingService
    reranker: RerankerService
    harness: GenerationHarness
    refine_harness: GenerationHarness
    qdrant_client: object
    stt: SarvamSttClient | None
    collections: dict[str, str]
    bm25_indexes: dict[str, Bm25Index]
    off_topic_gates: dict[str, OffTopicGate | None]


services = Services()


def _try_build_off_topic_gate(language: str, sample_size: int = 2000) -> OffTopicGate | None:
    """Samples chunk text already in the index and re-embeds it locally to
    compute a corpus centroid for the off-topic gate — no separate offline
    job needed since the index itself is the source of truth for "what's in
    scope." Deliberately does not ask Qdrant to return stored vector values:
    verified directly that qdrant-client 1.19.0 talking to the pinned
    qdrant/qdrant:v1.13.4 server silently deserializes `with_vectors=...`
    responses as all-zero arrays (a real client/server version-skew bug —
    search itself is unaffected, only vectors-in-the-response are). A
    zero-vector centroid makes every query register as off-topic, so this
    would silently refuse 100% of real queries. Payloads are unaffected by
    the bug, so re-embedding sampled text sidesteps it entirely.
    Returns None (gate disabled) if the collection doesn't exist yet or is
    empty, e.g. before the background indexing job has produced any points."""
    collection = services.collections[language]
    try:
        points, _ = services.qdrant_client.scroll(
            collection_name=collection, limit=sample_size, with_payload=["text"]
        )
    except Exception:
        logger.warning("Off-topic gate disabled: collection %s (language=%s) not ready", collection, language)
        return None
    texts = [p.payload["text"] for p in points if p.payload and p.payload.get("text")]
    if not texts:
        logger.warning("Off-topic gate disabled: collection %s (language=%s) is empty", collection, language)
        return None
    vectors = services.embedding.embed(texts).dense
    centroid = compute_corpus_centroid(vectors)
    logger.info("Off-topic gate built for language=%s from %d sampled points", language, vectors.shape[0])
    return OffTopicGate(centroid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading warm model pool...")
    services.embedding = EmbeddingService()
    services.reranker = RerankerService()
    grounding_validator = None
    if LOAD_GROUNDING_VALIDATOR:
        try:
            grounding_validator = GroundingValidator()
        except Exception as exc:
            # NLI validation is an additional, model-backed guardrail. Keep
            # safety, retrieval-confidence, citation, and provider guardrails
            # available if its optional runtime cannot initialize.
            logger.exception("Grounding validator unavailable; starting without NLI validation: %s", exc)
    services.qdrant_client = get_client(path=None if QDRANT_URL else QDRANT_PATH, url=QDRANT_URL)
    services.collections = {lang: collection_name(lang, INDEX_VERSION) for lang in SUPPORTED_LANGUAGES}
    services.bm25_indexes = {lang: Bm25Index(BM25_PATHS[lang]) for lang in SUPPORTED_LANGUAGES}
    services.off_topic_gates = {
        lang: (_try_build_off_topic_gate(lang) if LOAD_OFF_TOPIC_GATE else None) for lang in SUPPORTED_LANGUAGES
    }
    # The harness/refine_harness instances are shared across every configured
    # language -- the constructor's off_topic_gate is a fallback for callers
    # that don't override it per-call (see GenerationHarness.answer's
    # off_topic_gate param), not the per-request source of truth once more
    # than one language is configured. Everything else on the harness
    # (generator, grounding_validator) is already language-agnostic.
    services.harness = GenerationHarness(
        generator=build_generator(),
        grounding_validator=grounding_validator,
        off_topic_gate=services.off_topic_gates.get(DEFAULT_LANGUAGE),
    )
    # Phase two of two-phase answering gets its own generator with a far more
    # generous retry budget. The default 8s is sized for a caller who is
    # waiting on the response; nobody waits on a refinement — the user already
    # has a grounded answer — so cutting generation off at 8s there trades a
    # real synthesized answer for nothing. Measured directly: an 8s budget
    # expired mid-generation on a slower network and silently degraded the
    # refinement back to the same extract it was meant to improve.
    services.refine_harness = GenerationHarness(
        generator=build_generator(total_budget_seconds=REFINE_GENERATION_BUDGET_SECONDS),
        grounding_validator=grounding_validator,
        off_topic_gate=services.off_topic_gates.get(DEFAULT_LANGUAGE),
    )
    try:
        services.stt = SarvamSttClient()
    except RuntimeError:
        logger.warning("SARVAM_API_KEY not set — /v1/voice-query will return 503")
        services.stt = None
    logger.info("Warm pool ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Voice RAG API", lifespan=lifespan)
app.add_middleware(RateLimitMiddleware, max_requests=20, window_seconds=60.0)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    language: str = Field(default=DEFAULT_LANGUAGE, min_length=2, max_length=16)
    top_k: int = Field(default=10, ge=1, le=10)


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
    # True when this answer was returned inside the latency budget *and* a
    # generative answer was skipped only because it could not fit. The client
    # can then POST /v1/query/refine to upgrade it. See _PendingRefinement.
    refinement_available: bool = False


class RefineRequest(BaseModel):
    trace_id: str = Field(min_length=1, max_length=64)


class TranslateRequest(BaseModel):
    # Bounded because this reaches the model: an unbounded field here is an
    # open text-generation endpoint. The per-IP rate limiter is the other half
    # of that; answers this translates are one or two sentences.
    text: str = Field(min_length=1, max_length=4000)


class TranslateResponse(BaseModel):
    text: str
    latency_ms: dict[str, float]


@dataclass
class _PendingRefinement:
    """Everything needed to generate an answer that retrieval already earned.

    Two-phase answering exists because a live LLM call (~2.1s measured) and a
    200ms budget cannot both fit in one response. Rather than pick one, the
    first phase returns the grounded extractive answer inside the budget and
    the second phase upgrades it — so the deadline is met by the response the
    user actually waits for, and generation still happens.

    Holding the reranked candidates here is what makes phase two cheap: it
    reruns generation only, not embedding/retrieval/reranking, which the
    request already paid for.
    """

    query_text: str
    language: str
    candidates: list[RetrievalCandidate]
    query_embedding: np.ndarray
    payload_by_chunk_id: dict[str, dict]
    rerank_score_by_chunk_id: dict[str, float]
    created_at: float


# Bounded and TTL'd deliberately: this is a cache of in-flight work, not a
# session store. An unbounded dict keyed by trace_id is a memory leak with a
# client-supplied key. Single-process only — with multiple uvicorn workers a
# refine call can land on a worker that never saw the original query, which
# returns 404 and simply leaves the fast answer in place.
_PENDING_REFINEMENTS: OrderedDict[str, _PendingRefinement] = OrderedDict()
_REFINEMENT_TTL_SECONDS = 300.0
_REFINEMENT_MAX_ENTRIES = 256


def _remember_refinement(trace_id: str, pending: _PendingRefinement) -> None:
    now = time.monotonic()
    stale = [k for k, v in _PENDING_REFINEMENTS.items() if now - v.created_at > _REFINEMENT_TTL_SECONDS]
    for key in stale:
        _PENDING_REFINEMENTS.pop(key, None)
    _PENDING_REFINEMENTS[trace_id] = pending
    while len(_PENDING_REFINEMENTS) > _REFINEMENT_MAX_ENTRIES:
        _PENDING_REFINEMENTS.popitem(last=False)


def _sparse_vector(sparse: dict[int, float]) -> qdrant_models.SparseVector:
    return qdrant_models.SparseVector(indices=list(sparse.keys()), values=list(sparse.values()))


def _read_completed_index_state(language: str) -> tuple[bool, int | None]:
    """Return whether the deployment bootstrap recorded a complete index for this language.

    A Qdrant collection can exist while an interrupted upload is still only
    partially populated.  The versioned state manifest is written atomically
    by ``build_full_index.py`` after both Qdrant and BM25 are fully committed.
    """
    if not REQUIRE_COMPLETE_INDEX:
        return True, None
    state_path = INDEX_STATE_PATHS[language]
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        total = int(state["total"])
        complete = (
            state.get("stage") == "completed"
            and state.get("dense_hnsw_deferred") is False
            and int(state.get("next_offset", -1)) == total
        )
        return complete and total >= 0, total
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("healthcheck_index_state_failed language=%s path=%s error=%s", language, state_path, exc)
        return False, None


def _require_language(language: str) -> str:
    if language not in services.collections:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported language {language!r}; this deployment serves {sorted(services.collections)}",
        )
    return language


def _answer_query(
    query_text: str,
    language: str,
    top_k: int,
    *,
    trace_id: str | None = None,
    started_at: float | None = None,
    initial_timings: dict[str, float] | None = None,
) -> QueryResponse:
    trace_id = trace_id or str(uuid.uuid4())
    t_start = started_at if started_at is not None else time.perf_counter()
    timings: dict[str, float] = dict(initial_timings or {})

    if not query_text.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if len(query_text) > 2000:
        raise HTTPException(status_code=400, detail="query too long (max 2000 chars)")
    language = _require_language(language)

    # BM25 is purely lexical — it needs the raw query string, not the query
    # embedding — so it does not have to wait for the encoder. Starting it
    # here runs it underneath the GPU embedding call instead of after it,
    # which takes BM25 off the critical path entirely: measured at ~15.6ms
    # against the 964,603-chunk index versus ~29ms for embedding, so its cost
    # (and its long tail, dominated by high-document-frequency Hindi function
    # words) is absorbed rather than added.
    t_bm25 = time.perf_counter()
    bm25_future = _retrieval_executor.submit(services.bm25_indexes[language].search, query_text, top_k=TOP_K_PER_SIGNAL)

    t0 = time.perf_counter()
    q_embed = services.embedding.embed_query(query_text)
    timings["embedding_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dense_future = _retrieval_executor.submit(
        lambda: services.qdrant_client.query_points(
            collection_name=services.collections[language],
            query=q_embed.dense[0].tolist(),
            using="dense",
            limit=TOP_K_PER_SIGNAL,
        ).points
    )
    sparse_future = _retrieval_executor.submit(
        lambda: services.qdrant_client.query_points(
            collection_name=services.collections[language],
            query=_sparse_vector(q_embed.sparse[0]),
            using="sparse",
            limit=TOP_K_PER_SIGNAL,
        ).points
    )
    dense_hits = dense_future.result()
    try:
        sparse_hits = sparse_future.result(timeout=SPARSE_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        logger.warning("sparse_search_timeout trace_id=%s budget_s=%.2f", trace_id, SPARSE_TIMEOUT_SECONDS)
        sparse_hits = []
    try:
        bm25_hits = bm25_future.result(timeout=BM25_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        # The search keeps running in its worker thread; its result is just
        # discarded when it eventually completes. Dense + sparse alone still
        # answer the query — see BM25_TIMEOUT_SECONDS for why this is a
        # deliberate trade, not a degraded state to fix.
        logger.warning("bm25_search_timeout trace_id=%s budget_s=%.2f", trace_id, BM25_TIMEOUT_SECONDS)
        bm25_hits = []
    timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000
    # Wall time BM25 actually added on top of everything overlapping it —
    # ~0 when it finished under the embedding call, which is the common case.
    timings["bm25_wall_ms"] = max(0.0, (time.perf_counter() - t_bm25) * 1000 - timings["embedding_ms"] - timings["retrieval_ms"])

    dense_ranked = [h.payload["chunk_id"] for h in dense_hits]
    sparse_ranked = [h.payload["chunk_id"] for h in sparse_hits]
    bm25_ranked = [cid for cid, _ in bm25_hits]
    payload_by_chunk_id = {h.payload["chunk_id"]: h.payload for h in [*dense_hits, *sparse_hits]}

    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked, bm25_ranked])
    fused_chunk_ids = [cid for cid, _ in fused][:RERANK_CANDIDATES]
    timings["fusion_ms"] = (time.perf_counter() - t0) * 1000

    # Recover chunks that only BM25 found. The BM25 index stores chunk ids but
    # not text, so without this step any candidate the vector searches missed
    # is silently dropped before reranking — which would make BM25 able to
    # *reorder* results but never to *contribute* one, defeating the reason a
    # lexical signal is in the hybrid at all (exact ids, numbers, proper nouns
    # that embeddings under-weight). Qdrant point ids are a pure function of
    # chunk_id (retrieval/dense/index.py), so these are primary-key fetches,
    # not a filtered scan, and the call only happens when BM25 actually
    # contributed something unique to the top-K.
    # The fetch is dispatched now but deliberately *not* awaited here: it is
    # network/disk-bound (~22ms against an on-disk payload store) while the
    # reranker below is GPU-bound, so the two overlap and the recovery is
    # effectively free. The recovered chunks are then scored in a small second
    # reranker batch — exact, not an approximation, because cross-encoder
    # scores are per-(query, passage) independent: verified that splitting a
    # batch reproduces single-batch scores to max |diff| 0.0 with identical
    # ordering.
    missing_chunk_ids = [cid for cid in fused_chunk_ids if cid not in payload_by_chunk_id]
    recovery_future = None
    if missing_chunk_ids:
        recovery_future = _retrieval_executor.submit(
            services.qdrant_client.retrieve,
            collection_name=services.collections[language],
            ids=[stable_point_id(cid) for cid in missing_chunk_ids],
            with_payload=True,
        )

    rerank_score_by_chunk_id: dict[str, float] = {}
    if not fused_chunk_ids:
        resp = services.harness.answer(
            trace_id, query_text, language, [], off_topic_gate=services.off_topic_gates.get(language)
        )
    else:
        t0 = time.perf_counter()
        known_ids = [cid for cid in fused_chunk_ids if cid in payload_by_chunk_id]
        known_texts = [payload_by_chunk_id[cid]["text"] for cid in known_ids]
        scored: list[tuple[str, str, float]] = (
            list(zip(known_ids, known_texts, services.reranker.rerank(query_text, known_texts), strict=True))
            if known_texts
            else []
        )

        # Adaptive gate: only pay for BM25's extra candidates when they could
        # actually change the outcome. Reranking is almost perfectly linear in
        # candidate count (measured: ~4.8ms each, ~0 fixed cost), so scoring
        # the recovered chunks is not free. If the vector searches already
        # produced a decisively confident passage the harness would answer
        # extractively from, a BM25-only chunk would have to beat an already
        # -confident answer to matter — rare, and low value when it happens.
        # Below that bar is exactly where a lexical exact-match is most likely
        # to be the right answer, so that is where the work gets spent.
        already_decisive = bool(scored) and max(s for _, _, s in scored) >= EXTRACTIVE_CONFIDENCE_THRESHOLD
        # Second gate on the same work, for the same reason as the deadline in
        # the harness: BM25 recovery is a recall improvement, not a
        # correctness requirement, so it is the right thing to drop first when
        # a query has already spent most of its budget getting here.
        recovery_fits_budget = (time.perf_counter() - t_start) < REQUEST_BUDGET_SECONDS
        if recovery_future is not None and not already_decisive and recovery_fits_budget:
            try:
                for point in recovery_future.result():
                    if point.payload and "chunk_id" in point.payload:
                        payload_by_chunk_id[point.payload["chunk_id"]] = point.payload
                recovered_ids = [cid for cid in missing_chunk_ids if cid in payload_by_chunk_id]
                recovered_texts = [payload_by_chunk_id[cid]["text"] for cid in recovered_ids]
                if recovered_texts:
                    scored += list(
                        zip(
                            recovered_ids,
                            recovered_texts,
                            services.reranker.rerank(query_text, recovered_texts),
                            strict=True,
                        )
                    )
                timings["bm25_recovered"] = float(len(recovered_ids))
            except Exception as exc:
                # Never fail a request over an enrichment step — worst case we
                # fall back to the previous behaviour of dropping these.
                logger.warning("bm25_payload_recovery_failed trace_id=%s error=%s", trace_id, exc)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

        ranked = sorted(scored, key=lambda x: x[2], reverse=True)
        candidates = [
            RetrievalCandidate(
                chunk_id=cid,
                doc_id=payload_by_chunk_id[cid].get("passage_id", cid),
                language=language,
                text=text,
                rerank_score=float(score),
            )
            for cid, text, score in ranked[:top_k]
        ]
        rerank_score_by_chunk_id = {c.chunk_id: c.rerank_score for c in candidates if c.rerank_score is not None}

        t0 = time.perf_counter()
        resp = services.harness.answer(
            trace_id,
            query_text,
            language,
            candidates,
            query_embedding=q_embed.dense[0],
            # What is actually left of the request's budget after retrieval and
            # reranking have already spent from it — not a fresh per-stage
            # timeout, which is how a pipeline of individually-bounded stages
            # still ends up with an unbounded total.
            remaining_budget_seconds=REQUEST_BUDGET_SECONDS - (time.perf_counter() - t_start),
            min_generation_budget_seconds=MIN_GENERATION_BUDGET_SECONDS,
            off_topic_gate=services.off_topic_gates.get(language),
        )
        timings["generation_ms"] = (time.perf_counter() - t0) * 1000

    timings["total_ms"] = (time.perf_counter() - t_start) * 1000
    # The task's 200ms target covers "chunking + vector DB retrieval +
    # everything through to final output" — this system's own work. Speech-to-
    # text is a call out to Sarvam's servers, so it is reported separately
    # rather than folded into the number the budget governs: including a third
    # party's network round-trip would make the figure measure their latency as
    # much as this pipeline's. Both are surfaced, so nothing is hidden either
    # way — voice callers can see the true wall-clock cost as
    # pipeline_ms + stt_ms.
    timings["pipeline_ms"] = timings["total_ms"] - timings.get("stt_ms", 0.0)
    timings["budget_ms"] = REQUEST_BUDGET_SECONDS * 1000
    logger.info(
        "query_completed trace_id=%s mode=%s pipeline_ms=%.2f total_ms=%.2f within_budget=%s",
        trace_id,
        resp.mode,
        timings["pipeline_ms"],
        timings["total_ms"],
        timings["pipeline_ms"] <= timings["budget_ms"],
    )

    # Evidence resolves chunk_id -> the actual cited *passage* text, not the
    # claim text — payload_by_chunk_id holds the real passage text;
    # Claim.text is the LLM's own claim wording,
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

    # Phase two is offered only when generation was skipped *for time* — the
    # deadline fired on a query the router would otherwise have generated for.
    # A high-confidence extractive answer is not offered a refinement: the
    # router skipped the LLM because the passage already answers the question,
    # not because it ran out of budget. A refusal is not offered one either,
    # since there is nothing grounded to refine.
    refinement_available = DEADLINE_FALLBACK_FLAG in resp.guardrail_flags
    if refinement_available:
        _remember_refinement(
            trace_id,
            _PendingRefinement(
                query_text=query_text,
                language=language,
                candidates=candidates,
                query_embedding=q_embed.dense[0],
                payload_by_chunk_id=payload_by_chunk_id,
                rerank_score_by_chunk_id=rerank_score_by_chunk_id,
                created_at=time.monotonic(),
            ),
        )

    return QueryResponse(
        trace_id=trace_id,
        answer_text=resp.answer_text,
        mode=resp.mode,
        confidence=resp.confidence,
        guardrail_flags=resp.guardrail_flags,
        evidence=evidence,
        latency_ms=timings,
        refinement_available=refinement_available,
    )


@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    return _answer_query(req.query, req.language, req.top_k)


@app.post("/v1/query/refine", response_model=QueryResponse)
def refine_query(req: RefineRequest) -> QueryResponse:
    """Phase two: spend real time producing the generative answer.

    Deliberately has no latency budget — its whole purpose is to do the work
    that did not fit inside one. The caller already has a grounded answer, so
    this is an upgrade running in the background of a conversation, not
    something anyone is blocked on.
    """
    pending = _PENDING_REFINEMENTS.pop(req.trace_id, None)
    if pending is None:
        # Expired, evicted, already refined, or (multi-worker) handled by a
        # process that never saw the original query. The client keeps the fast
        # answer it already rendered.
        raise HTTPException(status_code=404, detail="no pending refinement for this trace_id")

    t_start = time.perf_counter()
    resp = services.refine_harness.answer(
        req.trace_id,
        pending.query_text,
        pending.language,
        pending.candidates,
        query_embedding=pending.query_embedding,
        remaining_budget_seconds=None,  # no deadline: see docstring
        # pending.language was already validated (via _require_language) when
        # the original request was answered, so no re-check is needed here.
        off_topic_gate=services.off_topic_gates.get(pending.language),
    )
    generation_ms = (time.perf_counter() - t_start) * 1000

    seen_chunk_ids: set[str] = set()
    evidence: list[EvidenceItem] = []
    for c in resp.claims:
        for cid in c.cited_chunk_ids:
            if cid in seen_chunk_ids:
                continue
            seen_chunk_ids.add(cid)
            payload = pending.payload_by_chunk_id.get(cid)
            evidence.append(
                EvidenceItem(
                    chunk_id=cid,
                    text=payload["text"] if payload else c.text,
                    rerank_score=pending.rerank_score_by_chunk_id.get(cid),
                )
            )

    logger.info("refine_completed trace_id=%s mode=%s generation_ms=%.2f", req.trace_id, resp.mode, generation_ms)
    return QueryResponse(
        trace_id=req.trace_id,
        answer_text=resp.answer_text,
        mode=resp.mode,
        confidence=resp.confidence,
        guardrail_flags=resp.guardrail_flags,
        evidence=evidence,
        # Phase two's own cost, reported separately rather than added to the
        # first response's total: the budget governs what the user waited for,
        # and conflating the two would misreport both.
        latency_ms={"generation_ms": generation_ms, "total_ms": generation_ms},
        refinement_available=False,
    )


@app.post("/v1/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest) -> TranslateResponse:
    """Translates an answer to English for a reader who does not read the
    corpus language.

    A separate endpoint rather than a field on the query response, for the
    same reason refinement is: it must not enter the 200ms budget. Nothing
    calls it unless a reader asks, so it runs on the refinement harness's
    generator, which carries the longer wall-clock budget.

    It reuses the model already loaded for generation — no translation API,
    no additional credential. Backends that do not offer translation degrade
    to a clear 501 rather than a confusing 500."""
    generator = services.refine_harness.generator
    if not hasattr(generator, "translate_to_english"):
        raise HTTPException(
            status_code=501,
            detail=f"{type(generator).__name__} does not support translation",
        )

    t0 = time.perf_counter()
    try:
        translated = generator.translate_to_english(req.text)
    except Exception as exc:
        # Never a 500 for a display aid: the caller already has the original
        # answer on screen and keeps it.
        logger.warning("translation_failed error=%s", exc)
        raise HTTPException(status_code=503, detail="translation unavailable") from exc
    elapsed = (time.perf_counter() - t0) * 1000

    if not translated:
        raise HTTPException(status_code=503, detail="translation returned nothing")
    return TranslateResponse(text=translated, latency_ms={"translation_ms": elapsed, "total_ms": elapsed})


class VoiceQueryResponse(QueryResponse):
    transcript: str


@app.post("/v1/voice-query", response_model=VoiceQueryResponse)
def voice_query(
    audio: UploadFile = File(...),
    language: str = Form(default=DEFAULT_LANGUAGE, min_length=2, max_length=16),
    top_k: int = Form(default=10, ge=1, le=10),
) -> VoiceQueryResponse:
    """Same pipeline as /v1/query, fed by a transcribed audio upload instead
    of typed text — closes the STT-not-wired-in gap: Sarvam's client was
    already built and tested standalone, this is the first place it's
    actually invoked from a live request."""
    if services.stt is None:
        raise HTTPException(status_code=503, detail="Voice input unavailable: SARVAM_API_KEY not configured")
    if audio.content_type and not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail=f"expected an audio file, got {audio.content_type}")
    # Fail before paying for a Sarvam STT call, not after.
    language = _require_language(language)

    request_started_at = time.perf_counter()
    audio_bytes = audio.file.read(MAX_AUDIO_BYTES + 1)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio upload")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio upload too large (max 15 MB)")

    stt_started_at = time.perf_counter()
    stt_result = services.stt.transcribe_bytes(
        audio_bytes,
        filename=audio.filename or "audio.webm",
        content_type=audio.content_type or "audio/wav",
        language_code=SARVAM_BCP47.get(language),
    )
    if not stt_result.transcript.strip():
        raise HTTPException(status_code=400, detail="speech-to-text returned an empty transcript")
    base = _answer_query(
        stt_result.transcript,
        language,
        top_k,
        trace_id=str(uuid.uuid4()),
        started_at=request_started_at,
        initial_timings={"stt_ms": (time.perf_counter() - stt_started_at) * 1000},
    )
    return VoiceQueryResponse(transcript=stt_result.transcript, **base.model_dump())


_export_state: dict[str, dict] = {
    language: {"status": "idle", "archive_path": None, "tmp_dir": None, "error": None}
    for language in SUPPORTED_LANGUAGES
}
_export_lock = threading.Lock()


def _check_export_token(token: str) -> None:
    expected_token = os.environ.get("VOICE_RAG_EXPORT_TOKEN")
    if not expected_token or token != expected_token:
        raise HTTPException(status_code=403, detail="invalid or unconfigured export token")


def _run_export(language: str) -> None:
    """Runs in a background thread so no single HTTP request stays open for
    the many minutes this legitimately takes. Copies Qdrant's on-disk
    collection directory straight from disk rather than going through
    Qdrant's snapshot+checksum API, which hung indefinitely on a collection
    this size. Safe because the collection is serve-only at this point —
    nothing is writing to those files concurrently.

    Resolves collection/bm25_path/state_path from `language` rather than
    closing over a single global -- exporting one language must never bundle
    another language's BM25 index or state manifest."""
    collection = services.collections[language]
    bm25_path = BM25_PATHS[language]
    state_path = INDEX_STATE_PATHS[language]
    state = _export_state[language]
    try:
        qdrant_storage_path = Path(os.environ.get("QDRANT__STORAGE__STORAGE_PATH", QDRANT_PATH))
        collection_dir = qdrant_storage_path / "collections" / collection
        if not collection_dir.exists():
            raise RuntimeError(f"Qdrant collection directory not found: {collection_dir}")

        # tempfile's default dir (system /tmp) sits on the small container
        # disk, not the persistent volume. data/full_index resolves onto
        # the volume via the existing /app/data symlink.
        export_root = state_path.parent / "exports"
        export_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"index_export_{language}_", dir=str(export_root)))
        state["tmp_dir"] = str(tmp_dir)

        archive_path = tmp_dir / f"{collection}_export.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(collection_dir, arcname="qdrant_collection")
            if Path(bm25_path).exists():
                tar.add(bm25_path, arcname="bm25")
            if state_path.exists():
                tar.add(state_path, arcname="state.json")

        state["archive_path"] = str(archive_path)
        state["status"] = "done"
    except Exception as exc:
        logger.exception("Index export failed for language=%s", language)
        state["error"] = str(exc)
        state["status"] = "error"


@app.post("/v1/admin/export-index/start")
def export_index_start(token: str, language: str) -> dict:
    """Kick off the export in the background and return immediately."""
    _check_export_token(token)
    language = _require_language(language)
    with _export_lock:
        state = _export_state[language]
        if state["status"] == "running":
            return {"status": "running", "language": language}
        state.update({"status": "running", "archive_path": None, "tmp_dir": None, "error": None})
        threading.Thread(target=_run_export, args=(language,), daemon=True).start()
    return {"status": "running", "language": language}


@app.get("/v1/admin/export-index/status")
def export_index_status(token: str, language: str) -> dict:
    _check_export_token(token)
    language = _require_language(language)
    state = _export_state[language]
    return {"status": state["status"], "error": state["error"], "language": language}


@app.get("/v1/admin/export-index/download")
def export_index_download(token: str, language: str) -> FileResponse:
    """Only serves an already-finished archive — a fast, static file
    transfer, not something that waits on server-side work."""
    _check_export_token(token)
    language = _require_language(language)
    state = _export_state[language]
    if state["status"] != "done" or not state["archive_path"]:
        raise HTTPException(
            status_code=409, detail=f"export not ready for language={language} (status={state['status']!r})"
        )
    archive_path = Path(state["archive_path"])
    tmp_dir = Path(state["tmp_dir"])
    return FileResponse(
        archive_path,
        filename=archive_path.name,
        media_type="application/gzip",
        background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
    )


@app.get("/v1/health")
def health() -> dict:
    """Readiness, not merely process liveness, for deployment probes.

    Reports one qdrant/bm25/index_complete/ready block per configured
    language rather than one flat set of booleans -- a language that hasn't
    finished bootstrapping yet is visible by name, not just as a mystery
    overall failure.
    """
    per_language: dict[str, dict] = {}
    for language in SUPPORTED_LANGUAGES:
        lang_checks = {"qdrant_collection": False, "bm25": False, "index_complete": False}
        collection = services.collections[language]
        expected_points: int | None = None
        try:
            lang_checks["qdrant_collection"] = services.qdrant_client.collection_exists(collection)
            manifest_complete, expected_points = _read_completed_index_state(language)
            if manifest_complete and lang_checks["qdrant_collection"]:
                if expected_points is None:
                    lang_checks["index_complete"] = True
                else:
                    actual_points = services.qdrant_client.count(collection_name=collection, exact=True).count
                    lang_checks["index_complete"] = actual_points == expected_points
                    if not lang_checks["index_complete"]:
                        logger.warning(
                            "healthcheck_index_count_mismatch language=%s collection=%s expected=%d actual=%d",
                            language,
                            collection,
                            expected_points,
                            actual_points,
                        )
        except Exception as exc:
            logger.warning("healthcheck_qdrant_failed language=%s error=%s", language, exc)
        try:
            # Tantivy opens the index at startup; a query is the least
            # ambiguous readiness check and does not mutate the index.
            services.bm25_indexes[language].search("health", top_k=1)
            lang_checks["bm25"] = True
        except Exception as exc:
            logger.warning("healthcheck_bm25_failed language=%s error=%s", language, exc)
        lang_checks["ready"] = (
            lang_checks["qdrant_collection"] and lang_checks["bm25"] and lang_checks["index_complete"]
        )
        per_language[language] = lang_checks

    qdrant_ok = False
    try:
        services.qdrant_client.get_collections()
        qdrant_ok = True
    except Exception as exc:
        logger.warning("healthcheck_qdrant_connection_failed error=%s", exc)

    languages_ready = (
        all(c["ready"] for c in per_language.values())
        if HEALTH_REQUIRE_ALL_LANGUAGES
        else any(c["ready"] for c in per_language.values())
    )
    checks = {
        "qdrant": qdrant_ok,
        "languages": per_language,
        "stt_configured": services.stt is not None,
        # Informational, not gating: the harness already degrades to
        # extractive answers on a generation failure, so a down backend is a
        # real but non-fatal condition — visible here rather than only
        # discoverable by the mode mix looking wrong.
        "generation_backend": type(services.harness.generator).__name__,
        "generation_backend_ready": (
            # A backend without a readiness probe (a remote API, say) reports
            # ready: its uptime is not this deployment's to speak for.
            services.harness.generator.is_ready()
            if hasattr(services.harness.generator, "is_ready")
            else True
        ),
    }
    ready = qdrant_ok and languages_ready
    if not ready:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ok", "checks": checks}


# Mounted last so it only catches paths the routes above didn't claim —
# serves the built frontend (Vite's production build output) at "/". No-op
# if the directory doesn't exist yet (e.g. before the frontend is built).
_FRONTEND_DIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST_DIR, html=True), name="frontend")
