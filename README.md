# ClearAsk — Voice-Enabled RAG over MSMARCO-XI

Submission for **HH Goa 2026 Shortlisting Task 2: Build a Voice-Enabled RAG Model**.

A user speaks a question in Hindi, the system transcribes it, retrieves grounded evidence from
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI), and answers —
citing exactly which retrieved passages support each claim, and refusing outright when it can't
find enough evidence.

- **GitHub repo:** https://github.com/Ridreb05/rag
- **Live link:** _to be added before submission_
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

| | P50 | P70 | P95 | P99 | P100 |
|---|---|---|---|---|---|
| **ms** | 76.5 | 85.0 | 105.7 | 154.2 | 623.3 |

Comfortably under the 200ms target through P99; P100 is a single outlier (a slow BM25 call, per
the per-stage breakdown in `reports/latency_benchmark/hi_full1.md`).

**End-to-end** (retrieval + guardrail routing + extractive-or-generative answer), n=150:

| | P50 | P70 | P95 | P99 | P100 |
|---|---|---|---|---|---|
| **ms** | 77.3 | 124.6 | 2734.1 | 3799.1 | 4257.6 |

Mode mix over those 150 queries: 96 extractive, 38 refused, 16 generative. Against the real,
comprehensive index, most queries retrieve a confident-enough top match to skip the LLM entirely
— **P50 and P70 end-to-end are both under the 200ms target on real production data.** Only the
tail (P95 and beyond, driven by the ~10% of queries that route to a real Gemini call) exceeds it
— a real network LLM round-trip cannot fit a 200ms budget on any current serving stack, local or
API-based, regardless of how confident the routing is. Stated plainly: the full pipeline meets
<200ms at P50/P70; it does not at P95+, and that remainder is inherent to calling an LLM at all,
not an implementation gap.

### How the hot path was optimized

Retrieval P50 went 85.6ms → 76.5ms and end-to-end P50 95.6ms → 77.3ms **while the hybrid
retrieval got strictly more correct** — the pipeline is now faster than it started *and* using a
third retrieval signal that was previously being discarded (below). Every change was driven by
per-stage profiling rather than intuition, and each was verified to leave results unchanged:

- **Embedding: 29.4ms → 12.8ms.** `FlagEmbedding.encode()` is built for batch throughput and
  charges a single query for work it cannot use: it re-runs `model.to(device)` and `.eval()`
  across all 568M parameters per call, tokenizes then length-sorts the batch, and most costly of
  all runs the model **twice** — once inside its adaptive batch-size probe loop, then again in
  the real encode loop. `embed_query` now does one tokenize and one forward through the model's
  own dense/sparse heads, replicating FlagEmbedding's exact lexical-weight post-processing. The
  vectors are *bit-identical* to the batch path (max absolute difference 0.0 across dense and
  sparse over real corpus queries) — which is the requirement, since they are matched against an
  index built by that batch path.
- **BM25: 22.0ms → 14.1ms.** `search()` called `Index.reload()` on every query, re-reading index
  metadata and reopening segment readers from disk each time. A reload is only meaningful after a
  commit, so the searcher is now cached and explicitly invalidated on write — index builds stay
  correct, and serving stops paying for a freshness check that can never find anything new.
- **BM25 taken off the critical path.** It is purely lexical and needs only the raw query string,
  yet it used to start *after* the encoder. It now runs concurrently with the GPU embedding call,
  so its cost is absorbed rather than added.

Several changes were measured and deliberately **not** made. Reducing the reranker's `max_length`
does nothing (dynamic padding means sequences are already short) and SDPA attention is already the
default, so reranking is genuinely compute-bound for a 568M cross-encoder — measured at ~4.8ms per
candidate with essentially *zero* fixed cost, which is why the fix below is about scoring fewer
candidates rather than scoring them faster. Dropping high-document-frequency Hindi function words
from BM25 queries is substantially faster (14.6ms → 8.4ms) but changes the top BM25 result on 20%
of queries — a real ranking regression bought for ~1.4ms at P50, since embedding, not BM25, is the
bottleneck of that concurrent phase. Lowering Gemini's `maxOutputTokens` from 2048 looked like an
easy win until measurement showed answers already use only 198 tokens at the median (341 max), so
lowering it would only truncate the longest answers and break JSON-schema parsing. All rejected on
evidence.

### The hybrid was only using two of its three signals

Profiling surfaced a correctness bug worth more than any latency tuning. The BM25 index stores
chunk ids but not chunk text, and passages were only ever collected from the dense and sparse
Qdrant hits — so **any candidate that only BM25 found was silently dropped before reranking.**
BM25 could reorder results it already agreed on, but could never *contribute* one, defeating the
entire reason a lexical signal is in a hybrid design (exact ids, numbers, and proper nouns that
embeddings under-weight).

Measured over 150 real corpus queries: **93% of queries were discarding BM25-only candidates (250
in total), and in 9 of them the discarded chunk was the best available answer.**

The fix uses a property already latent in the design: Qdrant point ids are a pure SHA-256 function
of `chunk_id`, so a chunk id from BM25 is a *client-computable primary key* — those payloads are
fetched by id, with no payload index and no filtered scan. Three refinements keep it cheap:

- The fetch is dispatched **before** reranking so the disk-bound Qdrant read overlaps the
  GPU-bound cross-encoder instead of queueing behind it.
- Recovered chunks are scored in a **second small reranker batch**, which is exact rather than an
  approximation: cross-encoder scores are per-(query, passage) independent, verified by
  reproducing single-batch scores to max |diff| 0.0 with identical ordering.
- An **adaptive gate** skips the whole step when the vector searches already produced a passage
  above the extractive-confidence threshold. A BM25-only chunk would have to beat an
  already-decisive answer to matter; below that bar is exactly where a lexical exact match is most
  likely to *be* the answer, so that is where the extra ~4.8ms/candidate gets spent. This cut the
  feature's cost from +24ms to +12ms at P50.

### Bounding the one unbounded thing

Generation is the only remote dependency left on the answer path, and it had two production
hazards that a latency target makes serious:

- **Retries were bounded by attempt count, not wall-clock.** With a 30s per-request timeout and 3
  retries plus backoff, a single query could occupy ~123 seconds inside a pipeline built around a
  sub-second budget. Retries now run against an overall deadline: no attempt starts that cannot
  finish in the remaining budget, per-request timeouts clamp to the time left, and backoff will
  not sleep past the deadline. Worst case is now deterministic (8s default, against a measured
  2.1s median / 3.2s max real generation).
- **A generation failure returned HTTP 500.** A provider outage, rate limit, or expired budget
  took the whole request down even though retrieval had already succeeded. It now degrades to the
  top reranked passage — already grounded and citable — flagged
  `generation_unavailable_extractive_fallback` so the downgrade is visible rather than silent.
  Both paths are covered by regression tests in `tests/test_harness.py`.

**A real bug was found and fixed while producing this evidence.** The first attempt at this
exact benchmark run came back with 100% of end-to-end queries refused — `qdrant-client==1.19.0`
(pinned in `pyproject.toml`) talking to the pinned `qdrant/qdrant:v1.13.4` server silently
deserializes any `with_vectors=...` response as all-zero arrays (search itself is unaffected;
only asking the client to hand back stored vector *values* is broken). The off-topic guardrail's
corpus-centroid computation was the only code path that did this, so it was silently computing a
zero-vector centroid — making every query register as maximally off-topic and get refused,
regardless of content. Since this same client/server pairing runs in the actual deployment, this
was a live bug, not just a benchmark artifact. Fixed in both `api/main.py` and
`benchmark/latency_benchmark.py` by re-embedding sampled chunk *text* locally instead of asking
Qdrant to return stored vectors (payloads are unaffected by the bug) — verified directly
(centroid norm went from `0.0` to a real unit-normalized `1.0`) before re-running the numbers
above.

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

The production container bakes Qdrant into the same image as the app (`Dockerfile.cloudrun`) so
there's a single deployable unit with no separate vector-database service to provision. The
Hindi `full1` index (964,603 chunks) is built once and baked into the image at build time, so
the deployed container starts serving immediately rather than re-indexing on boot.
`RUNPOD.md`-era tooling (`Dockerfile`, `infrastructure/runpod-entrypoint.sh`) exists purely as
the GPU environment used to *build* that index — it is not the deployment target.

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
- End-to-end latency meets the <200ms target at P50/P70 on the real deployed index, but not at
  P95 and beyond (see Latency above) — the remainder is queries routed to a real Gemini call,
  and no code change makes a real network LLM round-trip fit under 200ms.
