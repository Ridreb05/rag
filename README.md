# ClearAsk — Voice-Enabled RAG over MSMARCO-XI

Submission for **HH Goa 2026 Shortlisting Task 2: Build a Voice-Enabled RAG Model**.

A user asks a question in Hindi — typed or spoken. The system transcribes it, retrieves evidence
from [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI), and answers
with citations to the exact passages supporting each claim — or refuses, when the evidence isn't
there.

| | |
|---|---|
| **GitHub** | https://github.com/Ridreb05/rag |
| **Live demo** | https://clearask-voice-rag.fly.dev/ |
| **Dataset** | `ai4bharat/MSMARCO-XI`, Hindi `validation` split |
| **Index** | 964,603 chunks, version `full1`, indexed in full |

## Requirements at a glance

| # | Requirement | Implementation | Evidence |
|---|---|---|---|
| 1 | Speech-to-text (Sarvam or ElevenLabs) | Sarvam `saarika` REST | [Speech-to-text](#1-speech-to-text) |
| 2 | Non-naive chunking | 3 strategies + metadata-aware identity | [Chunking](#2-chunking) |
| 3 | Under 200ms | **P50 55.8ms · P70 61.2ms · P100 172.8ms** — 150/150 in budget | [Latency](#4-latency) |
| 4 | P50 / P70 / P100 analytics | n=150 deployed, n=1000 in-process | [Latency](#4-latency) |
| 5 | Proper harness, not a raw prompt call | Typed orchestrator: routing, structured output, retries, recovery | [Harness](#5-generation-harness) |
| 6 | Guardrails — knows when *not* to answer | 5 layers; **14 of 28 generated answers refused** on live data | [Guardrails](#6-guardrails) |

## Architecture

```
Voice input → Sarvam STT → Chunking / Retrieval (vector DB) → Answer generation
                            ├─ BGE-M3 dense      ┐
                            ├─ BGE-M3 sparse     ├─ RRF fusion → cross-encoder rerank
                            └─ Tantivy BM25      ┘                      ↓
                                                        guardrails → extractive | generative | refuse
```

`POST /v1/query` (text) and `POST /v1/voice-query` (audio) run the same path; voice adds a
transcription step in front.

Answering is **two-phase**: phase one returns a grounded, cited answer inside the 200ms budget;
phase two (`POST /v1/query/refine`) upgrades it with Gemini's synthesis a few seconds later. This
is how a sub-200ms response and a real LLM call coexist — see
[Two-phase answering](#two-phase-answering).

---

## 1. Speech-to-text

**Sarvam** (`saarika` STT REST API) — one provider, per the task's "pick one."

Implemented as a direct REST client (`pipeline/stt/sarvam_client.py`) against the batch endpoint
(files under 30s), rather than the streaming WebSocket variant. The browser records with
`MediaRecorder` and submits; there is no live streaming transcription.

STT latency is reported as `stt_ms`, separately from the 200ms budget — see
[what the budget covers](#what-the-200ms-covers).

## 2. Chunking

`pipeline/chunking/chunker.py` selects a strategy per passage rather than applying one fixed rule:

1. **Whole-passage (no split)** — MSMARCO-XI passages are pre-segmented and already
   retrieval-sized (median 55 words, p90 91, p99 139). Splitting them would fragment context for
   no retrieval gain.
2. **Sentence-aware packing** — for passages over the 512-token budget: greedily packs whole
   sentences into ~256-token windows with 64-token overlap, so a boundary never lands
   mid-sentence. Overlap carries trailing sentences into the next window.
3. **Fixed-token-window fallback** — when no usable sentence boundary exists, or a single sentence
   exceeds the window. Plain overlapping word windows.
4. **Metadata-aware identity** — every chunk carries `passage_id`, `language`, `chunk_index`,
   `token_count`, and the strategy that produced it, plus a `level`/`parent_id` hook for
   hierarchical chunking.

Chunk-length accounting uses a whitespace token counter, deliberately decoupled from the embedding
model's tokenizer to keep `transformers` out of the chunker. It is an approximation validated
against corpus samples, not an exact token count.

**What the strategies actually did**, across all 964,603 indexed chunks:

| strategy | chunks | share |
|---|---:|---:|
| `whole_passage` | 951,816 | 98.7% |
| `fixed_token_fallback` | 6,606 | 0.7% |
| `sentence_aware` | 6,181 | 0.6% |

953,388 passages produced 964,603 chunks — **1.012 chunks per passage**, median chunk 56 tokens
(p90 94, p99 246, max 512).

**Chunking is a no-op 98.7% of the time on this dataset, and that is the correct outcome.**
MSMARCO-XI ships pre-segmented passages; splitting a 56-token passage hurts retrieval. The
splitting strategies exist for the 1.3% that genuinely exceed the budget, and which passages those
are is decided per passage from measured token counts. Pointed at long-form documents, the same
code splits the majority instead — that is what the `level`/`parent_id` hook is reserved for, inert
here because this dataset has no document structure.

## 3. Retrieval

Three independent signals run concurrently and are fused:

| signal | what it catches |
|---|---|
| **Dense** — BGE-M3 1024-dim, Qdrant cosine ANN | semantic similarity |
| **Sparse (learned)** — BGE-M3 lexical weights, Qdrant named sparse vector | learned term importance |
| **Sparse (lexical)** — Tantivy BM25, embedded | exact IDs, numbers, proper nouns that embeddings under-weight |

Fused with **Reciprocal Rank Fusion** (`k=60`), chosen over weighted score fusion because dense
cosine, learned-sparse and BM25 scores live on incompatible scales — RRF needs only rank order, so
no per-language calibration is required. The top 8 fused candidates are reranked by a BGE
cross-encoder (`bge-reranker-v2-m3`) before the harness sees them.

BM25 also acts as a resilience layer: it is model-independent, so it keeps working if the embedding
service degrades.

## 4. Latency

### Deployed measurement (primary)

Real HTTPS requests to `POST /v1/query` on the deployed GPU Pod, full 964,603-chunk index, RTX
4090, with the API's own 20 req/60s rate limiter left enabled. The figure is the server's own
`pipeline_ms`. n=150.

| | P50 | P70 | P100 |
|---|---:|---:|---:|
| **ms** | **55.8** | **61.2** | **172.8** |

**150 of 150 queries completed inside the 200ms budget**, worst case included. Mode mix: 124
extractive, 26 refused. Raw output: `reports/latency_benchmark/hi_full1_deployed.json`.

Of those 150, 28 were eligible for phase-two refinement: 14 produced a generative answer and 14
were refused after generation by the grounding check. Phase two costs P50 2.76s / P100 4.11s and
sits outside the budget by design — nobody waits on it.

### In-process cross-check

`benchmark/latency_benchmark.py` measures the same pipeline in-process against a local Qdrant
server, isolating per-stage cost. RTX 4060 Laptop GPU; n=1000 for the retrieval sub-stage, n=150
for the full window.

| | P50 | P70 | P100 |
|---|---:|---:|---:|
| **Full window** | 74.1 | 79.0 | 131.5 |
| **Retrieval sub-stage** | 84.9 | 95.2 | 336.4 |

Slower at P50 than the deployment because the 4060 Laptop is a slower GPU — same code, same index.
Both are reported.

The retrieval sub-stage's P100 (336.4ms) is the one figure above target: a rare query where several
stage tails coincide. It is bounded — 623.3ms before the BM25 bound, 457.9ms before the sparse
bound — but not under 200ms. The deployed full-window P100 stays inside budget because the deadline
degrades that query rather than letting it run long. Per-stage P95/P99:
`reports/latency_benchmark/hi_full1.json`.

### What the 200ms covers

The target is **chunking + vector DB retrieval + through to final output** — stages three and four
of the task's pipeline (`Voice input → Speech-to-text → Chunking/Retrieval → Answer generation`).
Reported per request as `pipeline_ms` and enforced by `VOICE_RAG_REQUEST_BUDGET_SECONDS`.

| stage | in budget | cost |
|---|---|---|
| Chunking | yes | 0ms per query — amortised into the offline index build |
| Embedding → dense + sparse + BM25 → RRF → rerank | yes | the bulk |
| Guardrails → answer | yes | the remainder |
| Speech-to-text | no — upstream of the window | reported as `stt_ms` |

Chunking costs nothing per query because MSMARCO-XI passages are chunked once at index build. That
follows from the dataset already being passage-sized; it is not a runtime optimisation.

Speech-to-text sits ahead of the measured window, since the target's clause starts at chunking. It
is still reported: `stt_ms` on the response and result card, and `total_ms` (= `pipeline_ms +
stt_ms`) for the full wall-clock cost of a voice query.

## 5. Generation harness

`pipeline/generation/harness.py` is a typed orchestrator, not a prompt-in/text-out call:

1. **Safety pre-filter** — a deterministic regex gate runs *before* retrieval, so an unsafe query
   never spends an embedding call, a Qdrant round trip, or a reranker pass.
2. **Deadline-aware routing** — the harness receives what remains of the request budget and will
   not start a generation call it cannot finish, degrading to the top reranked passage and flagging
   `deadline_exceeded_extractive_fallback`. That flag also makes the answer eligible for phase two,
   so the deadline *defers* generation rather than cancelling it.
3. **Confidence-routed answering** — the reranker's top score decides the path:
   - `< 0.2` → **refuse**; retrieval found no real support.
   - `0.2 – 0.85` → **generate**, via whichever backend is configured (see
     [Generation backends](#generation-backends)).
   - `≥ 0.85` → **extractive**; a single passage already answers the question, so the LLM is
     skipped entirely.
4. **Error recovery** — provider/model failures are treated as guardrail outcomes, not crashes.
   Both backends implement retry/backoff for transient failures against a wall-clock budget, and a
   generation failure degrades to the top reranked passage rather than a 500 — retrieval already
   succeeded, so that passage is still a grounded answer.
5. **Grounding validation** — each generated claim is re-checked against its cited evidence with a
   multilingual NLI cross-encoder (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`), scored per claim
   rather than as one opaque number. Claims below 0.5 entailment, or with no valid citation, are
   dropped, and the answer is rebuilt from survivors only — never the model's raw prose.

### Generation backends

The generator is a runtime choice (`VOICE_RAG_GENERATION_BACKEND`), not a hardcoded import — the
harness depends on a `Generator` protocol (`generate(request) -> GeneratedAnswer | None`, `.model`),
satisfied by either:

| backend | `VOICE_RAG_GENERATION_BACKEND` | where it runs | structured output |
|---|---|---|---|
| **Gemini** (default) | `gemini` | remote API | JSON `responseSchema`, per-claim citations from the model |
| **Local vLLM** | `vllm` | this process's own GPU, via `pipeline/generation/vllm_service.py` | none — see below |

**Why a local model exists at all:** Gemini is a real network hop (~2.1s measured median), which
two-phase answering works around rather than removes. A model resident on the same GPU, given a
short prompt, has no network hop to pay — small enough that generation could stop being the reason
a request needs two-phase answering at all, for the confidence band that reaches it.

**What `LocalVllmGenerationService` does differently, specifically for latency:**

- Only the top 1-2 reranked candidates go into the prompt (`VOICE_RAG_VLLM_CONTEXT_CHUNKS`, default
  2) — the single largest lever on prefill time, traded against recall on questions that need more
  than 1-2 passages synthesized together.
- **No structured-output schema is requested.** Grammar-constrained decoding has real per-token
  cost, and Gemini pays it so the model can say *which* passage supports each claim. That
  information is already known here — the prompt contains only the chunks that could possibly be
  cited — so the service returns one claim citing every chunk placed in context, and lets the
  existing NLI grounding validator (which already scores a claim against a *list* of candidate
  evidence texts and picks the best match) do the same job it already does for Gemini's citations.
  This is not a weaker grounding guarantee — see `vllm_service.py`'s `generate()` docstring.
- `temperature=0`, `max_tokens` capped (default 20), and Qwen's `enable_thinking` chat-template flag
  disabled — a reasoning model that thinks before answering can spend its entire token budget on
  tokens the harness never sees as an answer.
- Runs as a persistent `vllm serve` process, not started per request; the harness talks to it over
  localhost HTTP with a small, bounded retry budget (a local failure is the server still loading or
  out of memory, not network flakiness — retrying gains little).
- Streaming responses so time-to-first-token is measured, not just total completion time; both are
  logged per request (`vllm_generation_completed`).

**Deployment:** the CPU Fly demo keeps `gemini` — there is no GPU to run a local model on, and this
default is unchanged from before this backend existed. The GPU RunPod Pod can opt into `vllm` via
its Dockerfile/entrypoint (`infrastructure/runpod-entrypoint.sh` starts `vllm serve` in its own venv
before uvicorn, isolated from the app's own torch install — see that file for why).

**What is and isn't verified:** the harness wiring, retry logic, prompt construction, and claim
assembly are unit-tested against a mocked vLLM server (`tests/test_vllm_service.py`) and pass. Real
TTFT/generation-latency numbers on this repo's actual GPU hardware, and the exact
Qwen3.5-4B-Instruct repo id, are **not yet measured** — that requires a live server on a GPU Pod,
which this session did not run. Reported here as implemented, not as benchmarked; the Latency
section above measures the Gemini backend only.

## 6. Guardrails

Five layers, ordered cheapest-first:

| layer | catches | needs the LLM? |
|---|---|---|
| Unsafe-input pre-filter | self-harm, violence-instruction, CSAE-adjacent patterns | no |
| Confidence-based refusal | retrieval below 0.2 — no real support | no |
| Off-topic centroid gate | queries outside the corpus's topic entirely | no |
| Per-claim NLI grounding | claims not entailed by their cited evidence | yes |
| Provider safety classifier | what the local filter misses | yes |

**Evidence it works on live data:** of 28 generated answers in the deployed benchmark run, **14
were refused after generation** because the grounding check rejected their claims. The system
generated, checked its own output against the retrieved evidence, and declined to show half of it.

The off-topic gate is enabled by default and active on the GPU deployment. It is disabled on the
CPU demo for startup-cost reasons — see [Deployment](#deployment).

## Two-phase answering

A live LLM call and a 200ms budget cannot share one response: measured generation is ~2.1s median
(P50 2.76s deployed). The usual resolutions are to drop generation (fast, but no longer RAG) or
accept a multi-second P100 (real RAG, misses the target). This system does neither.

**Phase one** answers from what retrieval already earned, inside the budget. The budget is carried
through the request as a deadline — not a per-stage timeout, since individually-bounded stages
still produce an unbounded total. Before committing to generation, the harness checks what is
actually left; if an LLM call cannot finish in time, it returns the top reranked passage, already
grounded and cited.

**Phase two** does the work that did not fit. `POST /v1/query/refine` re-runs generation against
the candidates the request already retrieved and reranked, and the UI swaps the synthesized answer
into the same card. It carries no deadline, because nobody is blocked on it.

Deployed result: **the response a user waits for is P50 55.8ms, 150/150 inside budget**, and 14 of
28 eligible queries still received a real generative answer seconds later.

**Refinement is offered only when generation was skipped for time.** A high-confidence extractive
answer is never refined — the LLM was skipped on merit, not budget. A refusal is not refined
either; there is nothing grounded to improve.

Two implementation details that matter:

- **Phase two uses its own generator with a 45s retry budget**, not the in-request 8s. Measured: an
  8s budget expired mid-generation on a slower network and degraded the refinement back into the
  same extract it existed to improve. A budget sized for a waiting caller is wrong for one who
  isn't.
- **Pending refinements are TTL'd (5 min) and bounded (256 entries)** — a dict keyed by a
  client-supplied `trace_id` is otherwise a memory leak. It is per-process, so under multiple
  workers a refine can land on a worker that never saw the query; that returns 404 and the client
  keeps the fast answer already on screen. Covered by `tests/test_refinement_store.py`.

## Engineering notes

Each change came from per-stage profiling and was verified to leave results unchanged, or verified
to be an improvement where results did change. Figures are before/after deltas from the run that
measured them.

| change | effect |
|---|---|
| **Embedding fast path.** `FlagEmbedding.encode()` charges a single query for batch-mode overhead: it re-runs `.to(device)`/`.eval()` over 568M params per call and runs the model *twice* (adaptive batch-size probe, then the real encode). `embed_query` does one tokenize, one forward. Output is bit-identical (max diff `0.0`) — required, since these vectors query an index the batch path built. | 29.4ms → 12.8ms |
| **BM25 searcher caching, then off the critical path.** `search()` reloaded the index from disk every call; the searcher is now cached and invalidated on write. BM25 needs only the raw query string, so it starts *before* the encoder and runs underneath it. | 22.0ms → 14.1ms, then absorbed |
| **BM25's tail bounded.** Tantivy's OR-of-terms cost scales with matched postings length, and high-frequency Hindi function words pushed single queries past 400ms (p50 ~15ms, p100 ~460ms — a ~30x gap). Past a 100ms budget BM25 is dropped for that request; dense + sparse still answer it. | retrieval P100 623ms → 303ms |
| **Qdrant sparse search had the same pathology** — and profiling (rather than assuming BM25 was still the culprit) showed it was the larger contributor: p50 6.5ms, p100 386ms. Bounded to the same 100ms. Dense is left unbounded: it is the only signal guaranteed to return for any query, and HNSW's cost doesn't scale with term frequency. | sparse P100 386ms → 101ms; retrieval P100 458ms → 336ms |
| **Generation retry budget.** 30s timeout × 3 retries + backoff allowed ~123s inside a sub-second pipeline. Retries now run against a wall-clock deadline, and generation failure degrades to the top reranked passage instead of a 500. | worst case ~123s → 8s |
| **The 200ms target became a real deadline** carried through the request, instead of something measured after the fact. | full-window P100 3434ms → 173ms deployed |
| **Then the trade-off it created was removed**, not documented. Pre-empting generation met the deadline by never generating — a poor answer to a task asking for both. Two-phase keeps the sub-200ms response *and* the LLM. | 150/150 in budget, 14 generative answers |

Two correctness bugs found while producing this evidence:

- **BM25 was hitting the corpus, but its unique results were silently discarded.** The BM25 index
  stores chunk IDs, not text, and payloads were only collected from dense/sparse Qdrant hits — so a
  BM25-only candidate could never reach reranking, making the lexical signal able to *reorder*
  results but never *contribute* one. Measured across 150 queries: **93% were discarding a
  BM25-only candidate, and in 9 the discarded chunk was the best available answer.** Fixed by
  fetching those chunks by primary key (Qdrant point IDs are a pure function of `chunk_id`),
  dispatched before reranking so it overlaps GPU work, and skipped when the vector search already
  returned a decisive answer.
- **The off-topic gate was refusing 100% of queries.** `qdrant-client==1.19.0` against
  `qdrant/qdrant:v1.13.4` silently returns all-zero vectors for `with_vectors=` responses (search
  itself is unaffected). The centroid computation was the only path reading vectors back, so it
  built a zero-vector centroid. Fixed by re-embedding sampled chunk text locally; centroid norm
  verified `0.0` → `1.0`.

Rejected on evidence: reranker `max_length` tuning (dynamic padding makes it a no-op) and
DF-filtering BM25's high-frequency terms (faster, but changes the top result on 20% of queries).

## Deployment

Two deployments, two jobs:

| | Fly.io (`Dockerfile.fly`) | RunPod GPU Pod (`Dockerfile`) |
|---|---|---|
| index | `api_smoketest`, ~40k chunks (~4% of `full1`) | **`full1`, all 964,603 chunks** |
| hardware | shared CPU | GPU |
| demonstrates | the system answers real queries end to end | the pipeline at designed scale and speed |
| always on | yes | no — started on demand |

The Fly link is permanent because it is cheap to leave running. The RunPod Pod is where the
pipeline performs as designed and is what the Latency section measures.

### RunPod GPU Pod — full index

`infrastructure/runpod-entrypoint.sh` runs Qdrant on localhost and uvicorn on port 8000 behind
RunPod's HTTPS proxy, in one container. It is built to survive stop/start on a persistent network
volume, which is what makes an on-demand Pod practical rather than a 24/7 cost:

- Attach a **network volume at `/workspace`** — index and Hugging Face model cache live there, so a
  restart reuses both instead of re-downloading ~3.8GB of weights and rebuilding a 9GB index.
- `VOICE_RAG_BOOTSTRAP_INDEX=1` is safe to leave on: the builder verifies its state manifest and
  exits quickly once the version is complete. Qdrant storage is isolated per
  `VOICE_RAG_INDEX_VERSION`, so a stale collection cannot start its optimizer and starve a clean
  bootstrap sharing the volume.
- A **bootstrap lock** prevents two Pods on one volume from writing the same BM25 index. Recovery
  from a stale lock after a forced stop is opt-in
  (`VOICE_RAG_RECOVER_STALE_BOOTSTRAP_LOCK=1` for one restart), because clearing it wrongly
  corrupts the index.
- `/v1/health` reports ready only when the state manifest **and** Qdrant's exact point count agree,
  so an interrupted upload cannot appear ready merely because its collection exists.

Required secrets: `SARVAM_API_KEY` (voice returns 503 without it), and `GEMINI_API_KEY` **only**
when `VOICE_RAG_GENERATION_BACKEND=gemini` (default) — startup fails without it in that
configuration. Set `VOICE_RAG_GENERATION_BACKEND=vllm` to run generation locally instead (see
[Generation backends](#generation-backends)); `GEMINI_API_KEY` is not read in that case.

`VOICE_RAG_REQUEST_BUDGET_SECONDS` (default `0.2`) sets the deadline. The default is correct for
normal operation — two-phase answering means generation still runs, just in phase two. Raising it
makes generation run inline instead, which is occasionally useful for debugging the generative path
but pushes the response past the latency target.

### Fly.io — always-on demo

CPU-only, serving the smaller `data/api_smoketest/` index (embedded Qdrant, ~40k chunks). Two
consequences follow from that scope:

- **It refuses often, correctly.** ~40k chunks is ~4% of the corpus, so most queries have no
  supporting passage indexed. Measured over 20 validation queries: 9 answered, 11 refused. On the
  full index the same routing answers the large majority.
- **It will not hit 200ms, and the UI does not pretend otherwise.** On shared CPU the cross-encoder
  reranker alone takes ~16s versus ~44ms on GPU, so `pipeline_ms` lands far over budget and every
  result card renders that state honestly. The <200ms claim belongs to the GPU numbers; this link
  demonstrates correctness, not latency.

The off-topic centroid gate is disabled here (`VOICE_RAG_LOAD_OFF_TOPIC_GATE=0`): building it
re-embeds a text sample locally, which took 30+ minutes on a throttled shared-CPU machine. It
remains on by default everywhere else, and every other guardrail stays active regardless.

## Repository layout

```
src/voice_rag/
  settings.py             # Env-var settings (SARVAM_API_KEY, GEMINI_API_KEY)
  pipeline/
    ingestion/            # MSMARCO-XI schema, HF access, dedup, dataset analysis
    chunking/             # Adaptive chunking strategies
    embeddings/           # BGE-M3 dense + learned-sparse service
    retrieval/            # Qdrant dense/sparse, Tantivy BM25, RRF fusion
    reranking/            # BGE cross-encoder
    guardrails/           # Safety filter, off-topic gate, NLI grounding
    generation/           # Gemini adapter, typed schemas, the harness
    stt/                  # Sarvam speech-to-text client
  api/
    main.py               # FastAPI app: lifespan, routes, readiness, SPA mount
    rate_limit.py         # Per-IP sliding-window rate limit (20 req/60s/worker)
scripts/                  # Resumable full-index builder, smoke-index helper
evaluation/               # Retrieval metrics (Recall@K, MRR, NDCG)
benchmark/                # Latency benchmark, live voice HTTP benchmark
frontend/                 # React 18 + TypeScript + Vite SPA
infrastructure/           # Container entrypoints
reports/                  # Benchmark evidence cited above
```

## Running locally

```powershell
# Backend
uv sync --frozen
cp .env.example .env   # SARVAM_API_KEY, GEMINI_API_KEY
uv run python -m voice_rag.pipeline.ingestion.build_corpus --languages hi --split validation
uv run python -m voice_rag.pipeline.chunking.build_chunks --languages hi --split validation
uv run python scripts/build_full_index.py --language hi --split validation \
    --qdrant-url http://localhost:6333 --index-version full1
uv run uvicorn voice_rag.api.main:app --port 8000

# Frontend
cd frontend && npm ci && npm run dev   # proxies /v1 to :8000
```

Or `docker compose up` for the two-service (Qdrant + app) topology.

## Tests

```powershell
uv run pytest -q      # 108 passed, 7 deselected (slow/GPU/provider), 2026-08-16
uv run pytest -m slow tests/test_ml_integration.py tests/test_gemini_integration.py tests/test_sarvam_integration.py
```

The default suite is fast unit/component coverage. Slow tests load real ML models or call paid
providers and are opt-in.

## Known limitations

- MSMARCO-XI covers 14 Indic languages, but one API process serves one configured language
  collection. The request-level `language` field labels the response; it does not switch indexes.
- Guardrails run after retrieval and reranking, so they prevent an unsafe *answer* but do not avoid
  retrieval cost for an unsafe query that passes the pre-filter.
- The <200ms target is met by the response the user waits for, not by a single request that also
  contains LLM synthesis. Two-phase answering resolves that conflict rather than reframing it — but
  the synthesized answer does arrive seconds later (P50 2.76s). No configuration of this system
  puts a live LLM call inside 200ms, because no current serving stack can.
- The retrieval sub-stage's P100 (336.4ms in-process) exceeds target on rare queries where several
  stage tails coincide, even with BM25 and sparse individually bounded.
- Latency depends materially on GPU: P50 55.8ms on a 4090 versus 74.1ms on a 4060 Laptop, same code
  and index.
- Half the generated answers in the deployed run were refused by the grounding check (14 of 28).
  That is the guardrail working, but it also indicates retrieval surfaces topically-close passages
  that do not support a specific claim — a retrieval-precision limit as much as a generation one.
- No full `TestClient` coverage of `/v1/query` or `/v1/voice-query`. The refinement store's
  eviction, TTL and single-use rules are unit-tested; endpoint wiring is verified against the live
  deployment rather than in CI.
