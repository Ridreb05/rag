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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client import models as qdrant_models
from starlette.background import BackgroundTask

from voice_rag.api.rate_limit import RateLimitMiddleware
from voice_rag.pipeline.embeddings.service import EmbeddingService
from voice_rag.pipeline.generation.gemini_service import GeminiGenerationService
from voice_rag.pipeline.generation.harness import EXTRACTIVE_CONFIDENCE_THRESHOLD, GenerationHarness
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
BM25_PATH = os.environ.get("BM25_PATH", f"data/full_index/bm25/{DEFAULT_LANGUAGE}_validation_{INDEX_VERSION}")
INDEX_STATE_PATH = Path(
    os.environ.get(
        "VOICE_RAG_INDEX_STATE_PATH",
        f"data/full_index/{DEFAULT_LANGUAGE}_validation_{INDEX_VERSION}.state.json",
    )
)
# The RunPod entrypoint enables this only after it has completed a bootstrap.
# Keeping it opt-in preserves the lightweight local-dev workflow where a
# developer may deliberately point the API at a small hand-built collection.
REQUIRE_COMPLETE_INDEX = os.environ.get("VOICE_RAG_REQUIRE_COMPLETE_INDEX", "0") == "1"
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
MAX_AUDIO_BYTES = 15 * 1024 * 1024
# Sarvam's BCP-47 codes; "or" -> "od-IN" is the one exception to the naive <code>-IN pattern.
SARVAM_BCP47 = {
    "as": "as-IN", "bn": "bn-IN", "gu": "gu-IN", "hi": "hi-IN", "kn": "kn-IN",
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
    bm25: Bm25Index
    qdrant_client: object
    collection: str
    off_topic_gate: OffTopicGate | None
    stt: SarvamSttClient | None


services = Services()


def _try_build_off_topic_gate(sample_size: int = 2000) -> OffTopicGate | None:
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
    try:
        points, _ = services.qdrant_client.scroll(
            collection_name=services.collection, limit=sample_size, with_payload=["text"]
        )
    except Exception:
        logger.warning("Off-topic gate disabled: collection %s not ready", services.collection)
        return None
    texts = [p.payload["text"] for p in points if p.payload and p.payload.get("text")]
    if not texts:
        logger.warning("Off-topic gate disabled: collection %s is empty", services.collection)
        return None
    vectors = services.embedding.embed(texts).dense
    centroid = compute_corpus_centroid(vectors)
    logger.info("Off-topic gate built from %d sampled points", vectors.shape[0])
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
    services.collection = collection_name(DEFAULT_LANGUAGE, INDEX_VERSION)
    services.bm25 = Bm25Index(BM25_PATH)
    services.off_topic_gate = _try_build_off_topic_gate() if LOAD_OFF_TOPIC_GATE else None
    services.harness = GenerationHarness(
        generator=GeminiGenerationService(),
        grounding_validator=grounding_validator,
        off_topic_gate=services.off_topic_gate,
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


def _sparse_vector(sparse: dict[int, float]) -> qdrant_models.SparseVector:
    return qdrant_models.SparseVector(indices=list(sparse.keys()), values=list(sparse.values()))


def _read_completed_index_state() -> tuple[bool, int | None]:
    """Return whether the deployment bootstrap recorded a complete index.

    A Qdrant collection can exist while an interrupted upload is still only
    partially populated.  The versioned state manifest is written atomically
    by ``build_full_index.py`` after both Qdrant and BM25 are fully committed.
    """
    if not REQUIRE_COMPLETE_INDEX:
        return True, None
    try:
        state = json.loads(INDEX_STATE_PATH.read_text(encoding="utf-8"))
        total = int(state["total"])
        complete = (
            state.get("stage") == "completed"
            and state.get("dense_hnsw_deferred") is False
            and int(state.get("next_offset", -1)) == total
        )
        return complete and total >= 0, total
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("healthcheck_index_state_failed path=%s error=%s", INDEX_STATE_PATH, exc)
        return False, None


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

    # BM25 is purely lexical — it needs the raw query string, not the query
    # embedding — so it does not have to wait for the encoder. Starting it
    # here runs it underneath the GPU embedding call instead of after it,
    # which takes BM25 off the critical path entirely: measured at ~15.6ms
    # against the 964,603-chunk index versus ~29ms for embedding, so its cost
    # (and its long tail, dominated by high-document-frequency Hindi function
    # words) is absorbed rather than added.
    t_bm25 = time.perf_counter()
    bm25_future = _retrieval_executor.submit(services.bm25.search, query_text, top_k=TOP_K_PER_SIGNAL)

    t0 = time.perf_counter()
    q_embed = services.embedding.embed_query(query_text)
    timings["embedding_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dense_future = _retrieval_executor.submit(
        lambda: services.qdrant_client.query_points(
            collection_name=services.collection,
            query=q_embed.dense[0].tolist(),
            using="dense",
            limit=TOP_K_PER_SIGNAL,
        ).points
    )
    sparse_future = _retrieval_executor.submit(
        lambda: services.qdrant_client.query_points(
            collection_name=services.collection,
            query=_sparse_vector(q_embed.sparse[0]),
            using="sparse",
            limit=TOP_K_PER_SIGNAL,
        ).points
    )
    dense_hits = dense_future.result()
    sparse_hits = sparse_future.result()
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
            collection_name=services.collection,
            ids=[stable_point_id(cid) for cid in missing_chunk_ids],
            with_payload=True,
        )

    rerank_score_by_chunk_id: dict[str, float] = {}
    if not fused_chunk_ids:
        resp = services.harness.answer(trace_id, query_text, language, [])
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
        if recovery_future is not None and not already_decisive:
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
        resp = services.harness.answer(trace_id, query_text, language, candidates, query_embedding=q_embed.dense[0])
        timings["generation_ms"] = (time.perf_counter() - t0) * 1000

    timings["total_ms"] = (time.perf_counter() - t_start) * 1000
    logger.info("query_completed trace_id=%s mode=%s total_ms=%.2f", trace_id, resp.mode, timings["total_ms"])

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

    return QueryResponse(
        trace_id=trace_id,
        answer_text=resp.answer_text,
        mode=resp.mode,
        confidence=resp.confidence,
        guardrail_flags=resp.guardrail_flags,
        evidence=evidence,
        latency_ms=timings,
    )


@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    return _answer_query(req.query, req.language, req.top_k)


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


_export_state: dict = {"status": "idle", "archive_path": None, "tmp_dir": None, "error": None}
_export_lock = threading.Lock()


def _check_export_token(token: str) -> None:
    expected_token = os.environ.get("VOICE_RAG_EXPORT_TOKEN")
    if not expected_token or token != expected_token:
        raise HTTPException(status_code=403, detail="invalid or unconfigured export token")


def _run_export(collection: str) -> None:
    """Runs in a background thread so no single HTTP request stays open for
    the many minutes this legitimately takes. Copies Qdrant's on-disk
    collection directory straight from disk rather than going through
    Qdrant's snapshot+checksum API, which hung indefinitely on a collection
    this size. Safe because the collection is serve-only at this point —
    nothing is writing to those files concurrently."""
    try:
        qdrant_storage_path = Path(os.environ.get("QDRANT__STORAGE__STORAGE_PATH", QDRANT_PATH))
        collection_dir = qdrant_storage_path / "collections" / collection
        if not collection_dir.exists():
            raise RuntimeError(f"Qdrant collection directory not found: {collection_dir}")

        # tempfile's default dir (system /tmp) sits on the small container
        # disk, not the persistent volume. data/full_index resolves onto
        # the volume via the existing /app/data symlink.
        export_root = INDEX_STATE_PATH.parent / "exports"
        export_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="index_export_", dir=str(export_root)))
        _export_state["tmp_dir"] = str(tmp_dir)

        archive_path = tmp_dir / f"{collection}_export.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(collection_dir, arcname="qdrant_collection")
            if Path(BM25_PATH).exists():
                tar.add(BM25_PATH, arcname="bm25")
            if INDEX_STATE_PATH.exists():
                tar.add(INDEX_STATE_PATH, arcname="state.json")

        _export_state["archive_path"] = str(archive_path)
        _export_state["status"] = "done"
    except Exception as exc:
        logger.exception("Index export failed")
        _export_state["error"] = str(exc)
        _export_state["status"] = "error"


@app.post("/v1/admin/export-index/start")
def export_index_start(token: str) -> dict:
    """Kick off the export in the background and return immediately."""
    _check_export_token(token)
    with _export_lock:
        if _export_state["status"] == "running":
            return {"status": "running"}
        _export_state.update({"status": "running", "archive_path": None, "tmp_dir": None, "error": None})
        threading.Thread(target=_run_export, args=(services.collection,), daemon=True).start()
    return {"status": "running"}


@app.get("/v1/admin/export-index/status")
def export_index_status(token: str) -> dict:
    _check_export_token(token)
    return {"status": _export_state["status"], "error": _export_state["error"]}


@app.get("/v1/admin/export-index/download")
def export_index_download(token: str) -> FileResponse:
    """Only serves an already-finished archive — a fast, static file
    transfer, not something that waits on server-side work."""
    _check_export_token(token)
    if _export_state["status"] != "done" or not _export_state["archive_path"]:
        raise HTTPException(status_code=409, detail=f"export not ready (status={_export_state['status']!r})")
    archive_path = Path(_export_state["archive_path"])
    tmp_dir = Path(_export_state["tmp_dir"])
    return FileResponse(
        archive_path,
        filename=archive_path.name,
        media_type="application/gzip",
        background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
    )


@app.get("/v1/health")
def health() -> dict:
    """Readiness, not merely process liveness, for deployment probes."""
    checks = {
        "qdrant": False,
        "collection": False,
        "bm25": False,
        "index_complete": False,
        "stt_configured": services.stt is not None,
    }
    expected_points: int | None = None
    try:
        services.qdrant_client.get_collections()
        checks["qdrant"] = True
        checks["collection"] = services.qdrant_client.collection_exists(services.collection)
        manifest_complete, expected_points = _read_completed_index_state()
        if manifest_complete and checks["collection"]:
            if expected_points is None:
                checks["index_complete"] = True
            else:
                actual_points = services.qdrant_client.count(
                    collection_name=services.collection, exact=True
                ).count
                checks["index_complete"] = actual_points == expected_points
                if not checks["index_complete"]:
                    logger.warning(
                        "healthcheck_index_count_mismatch collection=%s expected=%d actual=%d",
                        services.collection,
                        expected_points,
                        actual_points,
                    )
    except Exception as exc:
        logger.warning("healthcheck_qdrant_failed error=%s", exc)
    try:
        # Tantivy opens the index at startup; a query is the least ambiguous
        # readiness check and does not mutate the index.
        services.bm25.search("health", top_k=1)
        checks["bm25"] = True
    except Exception as exc:
        logger.warning("healthcheck_bm25_failed error=%s", exc)
    ready = checks["qdrant"] and checks["collection"] and checks["bm25"] and checks["index_complete"]
    if not ready:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ok", "checks": checks}


# Mounted last so it only catches paths the routes above didn't claim —
# serves the built frontend (Vite's production build output) at "/". No-op
# if the directory doesn't exist yet (e.g. before the frontend is built).
_FRONTEND_DIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST_DIR, html=True), name="frontend")
