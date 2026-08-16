# ClearAsk — Voice-Enabled RAG over MSMARCO-XI

Submission for **HH Goa 2026 Shortlisting Task 2: Build a Voice-Enabled RAG Model**.

A user speaks a question in Hindi, the system transcribes it, retrieves grounded evidence from
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI), and answers —
citing exactly which retrieved passages support each claim, and refusing outright when it can't
find enough evidence.

- **GitHub repo:** https://github.com/Ridreb05/rag
- **Live link:** https://clearask-voice-rag.fly.dev/
- **Dataset:** `ai4bharat/MSMARCO-XI`, Hindi (`hi`) validation split, indexed in full — 964,603
  chunks, index version `full1`

## Pipeline shape

```
Voice input → Sarvam speech-to-text → Chunking/Retrieval (Qdrant hybrid dense+sparse
              + Tantivy BM25, fused with RRF, reranked) → Answer generation
              (guardrail-gated harness, Gemini, grounded citations)
```

Both a typed text query (`POST /v1/query`) and a recorded voice query (`POST /v1/voice-query`,
multipart audio) run the same retrieval → guardrail → generation path; voice just adds a Sarvam
transcription step in front.

## Speech-to-text

**Sarvam** (`saarika` STT REST API), chosen per the task's "Sarvam or ElevenLabs, pick one"
requirement. Batch/synchronous endpoint only (files under 30s) — implemented as a direct REST
client (`pipeline/stt/sarvam_client.py`) rather than the streaming WebSocket variant, to get a
real, testable STT round trip first. A browser `MediaRecorder` records, then submits; there's no
live streaming transcription.

## Chunking

Chunking is adaptive, not a single fixed-size pass. `pipeline/chunking/chunker.py` picks per
passage:

1. **Whole-passage (no split).** MSMARCO-XI passages are already short, pre-segmented units —
   dataset analysis of the Hindi validation split puts the median passage at 55 words (p90 91,
   p99 139). For the overwhelming majority of passages, chunking is correctly a no-op: splitting
   an already-short passage would only fragment context for no retrieval benefit. This is the
   dataset-appropriate default, verified against a real sample rather than assumed.
2. **Sentence-aware packing**, for passages that exceed the token budget (default 512 tokens):
   greedily packs whole sentences into ~256-token windows with 64-token overlap, so a chunk
   boundary never lands mid-sentence. Overlap carries trailing sentences into the next window.
3. **Fixed-token-window fallback**, for the rare case sentence-aware packing can't help — either
   no usable sentence boundary was found, or a single sentence alone exceeds the window size.
   Falls back to plain overlapping word windows.
4. **Metadata-aware chunk identity.** Every chunk carries `passage_id`, `language`,
   `chunk_index`, `token_count`, and which strategy produced it, plus a `level`/`parent_id`
   schema hook for hierarchical (document → section → passage) chunking — inert on MSMARCO-XI,
   which has no document/section structure, but the schema doesn't need to change if the system
   is ever pointed at real long-form documents.

Chunk-length accounting uses a whitespace token counter deliberately decoupled from the
embedding model's own tokenizer (to avoid a heavy `transformers` dependency in the chunker
itself), verified against real corpus samples to be an adequate approximation, not presented as
an exact token count.

## Retrieval

Three independent signals run concurrently per query and get fused, not just one:

- **Dense** — BGE-M3 (`BAAI/bge-m3`) 1024-dim embeddings in Qdrant, cosine ANN search.
- **Sparse (learned)** — BGE-M3's own learned lexical-weight output, also in Qdrant, as a named
  sparse vector alongside dense in the same collection.
- **Sparse (lexical)** — Tantivy BM25, embedded (no server), a model-independent third signal
  that keeps working even if the embedding service degrades, and catches exact IDs/numbers/proper
  nouns an embedding-based signal can under-weight.

The three ranked lists are fused with **Reciprocal Rank Fusion** (`k=60`) — chosen over weighted
score fusion because dense cosine, BGE-M3 sparse, and BM25 scores live on incompatible scales;
RRF only needs rank order, so no per-language score calibration is required. The fused top
candidates (8) are reranked with a BGE cross-encoder (`bge-reranker-v2-m3`) before the harness
sees them.

## Generation harness

`pipeline/generation/harness.py` is a typed orchestrator, not a raw prompt-in/text-out call:

1. **Safety pre-filter** — a cheap deterministic regex gate runs before retrieval, so an
   obviously unsafe query never spends an embedding call, a Qdrant round trip, or a reranker
   pass.
2. **Confidence-routed answering** — the reranker's top score decides the path:
   - `< 0.2` → refuse outright (below this, retrieval didn't find real support).
   - `0.2 – 0.85` → generate, via Gemini (`gemini-flash-latest`) with structured output
     (`generationConfig.responseSchema` against a JSON schema: `answer_text` plus per-claim
     `cited_chunk_ids`) — this is what makes citations machine-checkable downstream, not
     free-text parsing.
   - `≥ 0.85` → answer extractively (return the top passage directly), skipping the LLM call
     entirely when retrieval is already confident.
3. **Error recovery** — Gemini's own safety filtering returning zero candidates or a blocked
   `finishReason` (`SAFETY`/`PROHIBITED_CONTENT`/`BLOCKLIST`/`RECITATION`) is treated as a
   guardrail outcome, not a crash. Since Gemini is called via a raw REST API rather than a
   provider SDK, retry/backoff for transient failures (network errors, 429, 5xx) is implemented
   explicitly in `gemini_service.py`, with exponential backoff and immediate failure on
   non-retryable 4xx errors.
4. **Grounding validation** (when loaded) — every generated claim is re-checked against its
   cited evidence chunk(s) with a multilingual NLI cross-encoder
   (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`), scored per claim rather than one opaque
   groundedness number for the whole answer. Claims with entailment `< 0.5`, or no valid
   citation at all, are dropped; the displayed answer is rebuilt only from surviving claims —
   never the model's raw prose once grounding is active.

## Guardrails

The system is built to know when *not* to answer, not just how to answer:

- **Unsafe-input pre-filter** — keyword/regex gate for self-harm, violence-instruction, and
  CSAE-adjacent patterns, ahead of retrieval.
- **Confidence-based refusal** — below the `0.2` rerank-confidence threshold, the harness
  refuses rather than guesses.
- **Off-topic centroid gate** — cosine distance from the query embedding to the indexed
  corpus's centroid, catching queries entirely outside the knowledge base's topic
  (`guardrails/off_topic.py`, covered by tests). Wired into the live API's harness construction
  (`api/main.py` passes the constructed gate into `GenerationHarness`), so both signals —
  confidence-based refusal and centroid-based off-topic rejection — are active on the real
  request path, not just in the benchmark harness.
- **Hallucination / grounding check** — the per-claim NLI entailment filter described above,
  optional at startup (degrades gracefully, not silently, if the NLI model fails to load).
- **Provider-level safety** — Gemini's own trained safety classifier as a second, more capable
  layer behind the local pre-filter.

## Latency

Measured with `benchmark/latency_benchmark.py` against a real Qdrant server running the actual
deployed index — Hindi, index version `full1`, 964,603 chunks, the same artifact baked into
`Dockerfile.cloudrun` — not a smaller stand-in index. 1000 queries for retrieval, 150 for
end-to-end (a full LLM call dominates that path, so 1000 real calls isn't necessary or cheap —
but all 150 are reported, including the slow ones, not a best-case sample).

**Retrieval only** (embed → dense+sparse+BM25 → RRF fuse → rerank), n=1000:

| | P50 | P70 | P100 |
|---|---|---|---|
| **ms** | 72.6 | 81.0 | 302.9 |

Comfortably under the 200ms target at P50/P70; the previous P100 (623.3ms) was BM25's own tail —
fixed by bounding BM25 to a 100ms budget per query (see Engineering notes), which brought P100
down to 302.9ms. Full percentile breakdown including P95/P99 is in
`reports/latency_benchmark/hi_full1.md` if useful, just not the three numbers the task asks for.

**End-to-end** (retrieval + guardrail routing + extractive-or-generative answer), n=150:

| | P50 | P70 | P100 |
|---|---|---|---|
| **ms** | 80.4 | 110.5 | 3434.5 |

Mode mix over those 150 queries: 96 extractive, 39 refused, 15 generative. Against the real,
comprehensive index, most queries retrieve a confident-enough top match to skip the LLM entirely
— **P50 and P70 end-to-end are both under the 200ms target on real production data.** P100 is
driven by the ~10% of queries that route to a real Gemini call — a real network LLM round-trip
cannot fit a 200ms budget on any current serving stack, local or API-based, regardless of how
confident the routing is. Stated plainly: the full pipeline meets <200ms at P50/P70; the worst
case does not, and that remainder is inherent to calling an LLM at all, not an implementation gap.

### Engineering notes

Retrieval P50 went 85.6ms → 76.5ms and end-to-end P50 95.6ms → 77.3ms, while the hybrid retrieval
got strictly *more correct* — driven by per-stage profiling, each change verified to leave results
unchanged (or, where results did change, verified that the change was an improvement):

- **Embedding, 29.4ms → 12.8ms.** `FlagEmbedding.encode()` charges a single query for batch-mode
  overhead it can't use: it re-runs `model.to(device)`/`.eval()` over all 568M params per call and
  runs the model **twice** (once in its adaptive batch-size probe, again in the real encode).
  `embed_query` now does one tokenize + one forward through the model's own heads. Output is
  *bit-identical* to the batch path (max diff `0.0`) — required, since these vectors are matched
  against an index the batch path built.
- **BM25, 22.0ms → 14.1ms, then taken off the critical path entirely.** `search()` reloaded the
  index from disk every call; the searcher is now cached and invalidated only on write. BM25 also
  needs only the raw query string, not the embedding, so it now starts *before* the encoder and
  runs underneath it instead of after — its cost is absorbed, not added.
- **A real correctness bug, not just a latency one: BM25 was hitting the corpus but its unique
  results were silently discarded.** The BM25 index stores chunk ids but not text, and payloads
  were only ever collected from the dense/sparse Qdrant hits — so a BM25-only candidate could
  never reach reranking. Measured over 150 real queries: **93% were discarding a BM25-only
  candidate, and in 9 the discarded chunk was the best available answer.** Fixed by fetching those
  chunks by id (Qdrant point ids are a pure function of `chunk_id`, so this is a primary-key
  fetch), dispatched *before* reranking so it overlaps the GPU work, scored in an exact second
  batch, and skipped entirely once the vector search already returned a decisive answer — the
  gate cut the feature's added cost from +24ms to +12ms at P50.
- **Generation's retry budget was unbounded in wall-clock terms.** 30s timeout × 3 retries + backoff
  meant one query could occupy ~123s inside a sub-second-budget pipeline. Retries now run against
  an 8s deadline (measured real generation: 2.1s median, 3.2s max), and a generation failure now
  degrades to the top reranked passage instead of a 500 — retrieval already succeeded, so that
  passage is still a grounded answer. Both covered by tests in `tests/test_harness.py`.
- **Found while producing this evidence:** the first benchmark run came back 100% refused.
  `qdrant-client==1.19.0` against the pinned `qdrant/qdrant:v1.13.4` server silently returns
  all-zero vectors for any `with_vectors=...` response (search itself is unaffected). The
  off-topic guardrail's centroid computation was the only code path asking for vectors back, so it
  was computing a zero-vector centroid and refusing everything — a live bug, since this exact
  client/server pairing runs in the real deployment. Fixed by re-embedding sampled chunk text
  locally instead of reading stored vectors back (centroid norm verified `0.0` → `1.0`).
- **BM25's own tail was the last thing driving worst-case latency, retrieval P100 623ms → 303ms.**
  BM25 runs concurrently underneath the embedding call, so it's usually free — but Tantivy's
  OR-of-terms query costs roughly O(sum of matched postings-list lengths), and high-frequency Hindi
  function words occasionally pushed a single query past 400ms (measured: p50 ~15ms, p99 ~85ms,
  p100 ~460ms — a ~30x typical-to-worst gap). Bounded to a 100ms budget: past that, BM25 is dropped
  for that request rather than blocking on it (dense+sparse still answer the query), the same
  wall-clock-budget trade already made for generation. Verified end-to-end: BM25 P100 460ms →
  131ms, retrieval P100 623.3ms → 302.9ms, end-to-end P100 4257.6ms → 3434.5ms.

Rejected on evidence: reranker `max_length` tuning (dynamic padding already makes it a no-op) and
DF-filtering BM25's high-frequency terms (faster, but changes the top result on 20% of queries).

## Repository layout

```
src/voice_rag/
  settings.py            # Central env-var settings (SARVAM_API_KEY, GEMINI_API_KEY)
  pipeline/
    ingestion/            # MSMARCO-XI schema, HF source access, corpus dedup, dataset analysis
    chunking/              # Adaptive chunking strategies (above)
    embeddings/            # BGE-M3 dense + learned-sparse embedding service
    retrieval/              # Qdrant dense/sparse index, Tantivy BM25, RRF fusion
    reranking/              # BGE cross-encoder reranker
    guardrails/             # Safety pre-filter, off-topic gate, NLI grounding
    generation/             # Gemini adapter, typed schemas, the harness
    stt/                    # Sarvam speech-to-text client
  api/
    main.py                # FastAPI app: lifespan, routes, readiness, SPA mount
    rate_limit.py           # In-memory per-IP sliding-window rate limit (20 req/60s/worker)
scripts/                  # Persistent resumable full-index builder, small smoke-index helper
evaluation/                # Closed-world retrieval metrics (Recall@K, MRR, NDCG) + subset eval
benchmark/                 # Latency benchmark (above) + live voice HTTP benchmark
frontend/                  # React 18 + TypeScript + Vite SPA ("ClearAsk")
infrastructure/            # RunPod and Cloud Run container entrypoints
```

## Running locally

```powershell
# Backend
uv sync --frozen
cp .env.example .env   # fill in SARVAM_API_KEY, GEMINI_API_KEY
uv run python -m voice_rag.pipeline.ingestion.build_corpus --languages hi --split validation
uv run python -m voice_rag.pipeline.chunking.build_chunks --languages hi --split validation
uv run python scripts/build_full_index.py --language hi --split validation \
    --qdrant-url http://localhost:6333 --index-version full1
uv run uvicorn voice_rag.api.main:app --port 8000

# Frontend
cd frontend
npm ci
npm run dev   # proxies /v1 to http://localhost:8000
```

Or `docker compose up` for the two-service (Qdrant + app) local topology.

## Deployment

**Live deployment** runs on Fly.io (`Dockerfile.fly`), CPU-only. It bakes in a smaller demo index
(`data/api_smoketest/`, embedded Qdrant, ~40k chunks) rather than the full `full1` production
index, so it fits a modest, affordable machine — the Latency section's numbers come from a real
benchmark against the full 964,603-chunk index on GPU, measured separately, not reproduced live.
This deployment exists to prove the system answers real voice and text queries end to end.

`Dockerfile.cloudrun` and `Dockerfile` (RunPod) bake the full production index into a GPU-backed
image (Cloud Run / RunPod) and are the architecture this system is actually designed to run at —
not used for the live link here for reasons unrelated to the code (regional payment-processing
issues with Cloud Billing, not a technical limitation). `infrastructure/runpod-entrypoint.sh`
also doubles as the GPU environment originally used to *build* the full index.

## Tests

```powershell
uv run pytest -q   # 93 passed, 7 deselected (slow/GPU/provider tests) as of 2026-08-16
uv run pytest -m slow tests/test_ml_integration.py tests/test_gemini_integration.py tests/test_sarvam_integration.py
```

Default suite is fast unit/component coverage; slow tests load real ML models or call paid
providers and are opt-in only.

## Known limitations

Stated plainly rather than glossed over:

- MSMARCO-XI covers 14 Indic languages; one running API process serves one configured language
  collection at a time. Request-level `language` labels the response but doesn't switch indexes.
- Guardrails run after retrieval and reranking, so they can prevent an unsafe *answer* but don't
  avoid retrieval cost for an unsafe query that passes the pre-filter.
- There's no full `TestClient` coverage of `/v1/query` or `/v1/voice-query` — passing the
  default fast unit-test suite alone is not live end-to-end proof.
- End-to-end latency meets the <200ms target at P50/P70 on the real deployed index, but the worst
  case does not (see Latency above) — that's queries routed to a real Gemini call, and no code
  change makes a real network LLM round-trip fit under 200ms.
