# ClearAsk — Voice-Enabled RAG over MSMARCO-XI

Submission for **HH Goa 2026 Shortlisting Task 2: Build a Voice-Enabled RAG Model**.

A user asks a question in Hindi — typed or spoken. The system transcribes it, retrieves evidence
from [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI), and answers
with citations to the exact passages supporting each claim — or refuses, when the evidence isn't
there. Generation runs on a model served on the same GPU, so the whole pipeline stays inside a
200ms budget.

| | |
|---|---|
| **GitHub** | https://github.com/Ridreb05/rag |
| **Dataset** | `ai4bharat/MSMARCO-XI`, Hindi `validation` split |
| **Index** | 964,603 chunks, version `full1`, indexed in full |
| **Deployment** | RunPod GPU Pod, started on demand — see [Deployment](#deployment) |

## Requirements at a glance

| # | Requirement | Implementation | Evidence |
|---|---|---|---|
| 1 | Speech-to-text (Sarvam or ElevenLabs) | Sarvam `saarika` REST | [Speech-to-text](#1-speech-to-text) |
| 2 | Non-naive chunking | 3 strategies + metadata-aware identity | [Chunking](#2-chunking) |
| 3 | Under 200ms | retrieval **P50 55.8ms · P70 61.2ms · P100 172.8ms**, 150/150 in budget | [Latency](#4-latency) |
| 4 | P50 / P70 / P100 analytics | n=150 deployed, n=1000 in-process | [Latency](#4-latency) |
| 5 | Proper harness, not a raw prompt call | Typed orchestrator: routing, deadline, retries, recovery, grounding | [Harness](#5-generation-harness) |
| 6 | Guardrails — knows when *not* to answer | 4 layers; **14 of 28 generated answers refused** on live data | [Guardrails](#6-guardrails) |

## Architecture

```
Voice input → Sarvam STT → Chunking / Retrieval (vector DB) → Answer generation
                            ├─ BGE-M3 dense      ┐
                            ├─ BGE-M3 sparse     ├─ RRF fusion → cross-encoder rerank
                            └─ Tantivy BM25      ┘                      ↓
                                                        guardrails → extractive | generative | refuse
```

`POST /v1/query` (text) and `POST /v1/voice-query` (audio) run the same path; voice adds a
transcription step in front. Generation is served by a local vLLM process on the same GPU as
retrieval — see [Generation backend](#generation-backend) for why that choice is what makes the
latency target reachable.

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

### Deployed measurement (primary), n=150

Real HTTPS requests to `POST /v1/query` on the deployed GPU Pod, full 964,603-chunk index, RTX
4090, with the API's own 20 req/60s rate limiter left enabled. The figure is the server's own
`pipeline_ms`.

| | P50 | P70 | P100 |
|---|---:|---:|---:|
| **ms** | **55.8** | **61.2** | **172.8** |

**150 of 150 queries completed inside the 200ms budget**, worst case included. Mode mix: 124
extractive, 26 refused. Raw output: `reports/latency_benchmark/hi_full1_deployed.json`.

This run measured the pipeline with generation deferred (the deadline pre-empted it, and it was
completed out of band — see [Two-phase answering](#two-phase-answering)). It is therefore an
honest measurement of **retrieval, guardrails and extractive answering** at scale, not of inline
generation.

### Inline local generation — measured, not yet benchmarked

With generation served locally on the same GPU it now runs *inside* the request. Per-stage timing
from the deployment:

| stage | ms |
|---|---:|
| embedding | 15 |
| retrieval (dense + sparse + BM25) | 4 |
| BM25 wall (overlapped) | 0 |
| fusion | 0 |
| rerank | 18 |
| **retrieval subtotal** | **37** |
| generation (20 output tokens) | 190 |
| **total** | **227** |

That 227ms is over budget, and the response is the tuning that follows from it: generation cost
decomposes to a 40–60ms fixed per-request overhead plus ~6.5–7.5ms per output token, so the output
cap moved 20 → 14 tokens, which puts the total at ~182–188ms.

**Stated plainly: those are single-query observations and an arithmetic projection, not a
benchmark.** The n=150 figures above were produced under the previous configuration. A full
re-benchmark of the inline-generation configuration has not been run, and until it is, the
sub-200ms claim for inline generation is a projection while the sub-200ms claim for the
retrieval/extractive path is a measurement.

### In-process cross-check

`benchmark/latency_benchmark.py` measures the same pipeline in-process against a local Qdrant
server, isolating per-stage cost. RTX 4060 Laptop GPU; n=1000 for the retrieval sub-stage, n=150
for the full window.

| | P50 | P70 | P100 |
|---|---:|---:|---:|
| **Full window** | 74.1 | 79.0 | 131.5 |
| **Retrieval sub-stage** | 84.9 | 95.2 | 336.4 |

Slower at P50 than the deployment because the 4060 Laptop is a slower GPU — same code, same index.
Both are reported. The retrieval sub-stage's P100 (336.4ms) is the one figure above target: a rare
query where several stage tails coincide. It is bounded — 623.3ms before the BM25 bound, 457.9ms
before the sparse bound — but not under 200ms. Per-stage P95/P99:
`reports/latency_benchmark/hi_full1.json`.

### What the 200ms covers

The target is **chunking + vector DB retrieval + through to final output** — stages three and four
of the task's pipeline (`Voice input → Speech-to-text → Chunking/Retrieval → Answer generation`).
Reported per request as `pipeline_ms` and enforced by `VOICE_RAG_REQUEST_BUDGET_SECONDS`.

| stage | in budget | cost |
|---|---|---|
| Chunking | yes | 0ms per query — amortised into the offline index build |
| Embedding → dense + sparse + BM25 → RRF → rerank | yes | ~37ms measured |
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
   - `0.2 – 0.85` → **generate**.
   - `≥ 0.85` → **extractive**; a single passage already answers the question, so the model is
     skipped entirely — faster *and* less able to distort what the passage says.
4. **Error recovery** — a backend failure is a guardrail outcome, not a crash. Retries run against
   a wall-clock budget rather than an attempt count, and a failure degrades to the top reranked
   passage rather than a 500 — retrieval already succeeded, so that passage is still a grounded
   answer.
5. **Grounding validation** — each generated claim is re-checked against its cited evidence with a
   multilingual NLI cross-encoder (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`), scored per claim
   rather than as one opaque number. Claims below 0.5 entailment, or with no valid citation, are
   dropped, and the answer is rebuilt from survivors only — never the model's raw prose.

Thresholds are environment-overridable (`VOICE_RAG_LOW_CONFIDENCE_THRESHOLD`,
`VOICE_RAG_EXTRACTIVE_THRESHOLD`, `VOICE_RAG_ENTAILMENT_THRESHOLD`) so routing can be tuned against
a running deployment rather than by rebuilding an image.

### Generation backend

**Qwen3.5-4B served by vLLM on the same GPU as retrieval**, FP8 weights, `temperature=0`, output
capped at 14 tokens, reasoning mode disabled.

The backend is a runtime choice (`VOICE_RAG_GENERATION_BACKEND`), not a hardcoded import: the
harness depends on a `Generator` protocol (`generate(request) -> GeneratedAnswer | None`, plus
`.model`), so adding another provider means writing one class and one branch in
`generation/factory.py`. A hosted-API backend previously occupied that seam and was removed once
local generation worked — it measured ~2.1s median, almost entirely network round trip, which no
tuning fits inside 200ms. Carrying it meant maintaining a path the deployment could never use.

**Choices made specifically for latency**, in the order they cost the most:

- **FP8 weights.** Decoding is memory-bandwidth bound: every token reads the whole weight set from
  HBM, so 4B parameters in bf16 move 8GB/token and cap a 4090 (~1008 GB/s) at ~126 tok/s. Against
  a ~144ms generation budget that is under 11 tokens — not a Hindi sentence. FP8 halves bytes per
  token and roughly doubles the ceiling. Ada has native FP8 tensor cores, so this is
  hardware-supported, not emulated.
- **CUDA graphs enabled.** Measured with eager execution: 634ms for 24 tokens, ~40 tok/s against
  that 126 tok/s ceiling — only ~32% of the hardware, the rest being per-token Python and
  kernel-launch overhead.
- **Only the top 1–2 reranked passages** enter the prompt. Fewer context tokens is the largest
  lever on prefill; the trade is recall on questions needing more passages synthesized.
- **No structured-output schema requested.** Grammar-constrained decoding has real per-token cost,
  and such a schema exists to let the model say *which* passage supports a claim. That is already
  known here — the prompt contains only the chunks that could be cited — so the service returns one
  claim citing every context chunk and lets the NLI validator do the checking it already does.
  This is not a weaker guarantee: the validator scores a claim against a *list* of candidate
  evidence texts and picks the best match, so a model-declared citation was only ever a hint about
  which passage to check, never a substitute for checking.
- **Reasoning mode disabled** (`enable_thinking: false`). A model that reasons before answering can
  spend its entire token budget on tokens the harness never sees as an answer.
- **Persistent server, streaming responses.** The model is loaded once, not per request, and
  streaming means time-to-first-token is measured separately from total completion
  (`vllm_generation_completed` logs both).

## 6. Guardrails

Four layers, ordered cheapest-first:

| layer | catches | needs the model? |
|---|---|---|
| Unsafe-input pre-filter | self-harm, violence-instruction, CSAE-adjacent patterns | no |
| Confidence-based refusal | retrieval below 0.2 — no real support | no |
| Off-topic centroid gate | queries outside the corpus's topic entirely | no |
| Per-claim NLI grounding | claims not entailed by their cited evidence | yes |

**Evidence it works on live data:** of 28 generated answers in a deployed benchmark run, **14 were
refused after generation** because the grounding check rejected their claims. The system generated,
checked its own output against the retrieved evidence, and declined to show half of it.

Moving generation to a self-hosted model removed a layer that a hosted API had provided for free:
a provider-side trained safety classifier. The input-side filter is now a keyword/regex list with
no more capable second opinion behind it, which is a real reduction in depth — the answer-side
guardrails above are what carry the weight.

## Two-phase answering

A hosted LLM call and a 200ms budget cannot share one response. Rather than pick between them, the
system answers in two phases:

**Phase one** answers from what retrieval already earned, inside the budget. The budget is carried
through the request as a deadline — not a per-stage timeout, since individually-bounded stages
still produce an unbounded total. Before committing to generation, the harness checks what is
actually left; if a call cannot finish in time, it returns the top reranked passage, already
grounded and cited.

**Phase two** does the work that did not fit. `POST /v1/query/refine` re-runs generation against
the candidates the request already retrieved and reranked, and the UI swaps the synthesized answer
into the same card. It carries no deadline, because nobody is blocked on it.

**With local generation this is now a fallback rather than the normal path.** Generation runs
inline because it fits; two-phase engages when a slow query leaves too little budget. The n=150
benchmark above was produced when generation was remote and *always* deferred, which is why its
mode mix contains no generative answers.

Two implementation details that matter:

- **Phase two uses its own generator with a longer retry budget.** Measured: a budget sized for a
  waiting caller expired mid-generation and degraded the refinement back into the same extract it
  existed to improve. A budget for a caller who is not waiting is a different number.
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
| **The 200ms target became a real deadline** carried through the request, instead of something measured after the fact. | full-window P100 3434ms → 173ms deployed |
| **Two-phase answering** removed the trade-off that deadline created — meeting the budget by never generating is a poor answer to a task that asks for both. | 150/150 in budget, generation preserved |
| **Generation moved on-GPU.** A hosted API's ~2.1s median is almost entirely network round trip; a model on the same GPU has no such hop. | 3240ms → 634ms |
| **CUDA graphs + FP8.** Eager execution ran at ~32% of the hardware's bandwidth ceiling; bf16 then capped useful output at under 11 tokens regardless of scheduling. | 634ms → 190ms @20 tokens |

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

A single deployment: a **RunPod GPU Pod** running the full index and the local model, started when
needed rather than continuously. `infrastructure/runpod-entrypoint.sh` runs Qdrant on localhost,
starts `vllm serve`, and then uvicorn on port 8000 behind RunPod's HTTPS proxy — one container,
three processes.

It is built to survive stop/start on a persistent network volume, which is what makes an on-demand
Pod practical rather than a 24/7 cost:

- Attach a **network volume at `/workspace`** — index, model weights and the Hugging Face cache
  live there, so a restart reuses all three instead of re-downloading ~12GB and rebuilding a 9GB
  index.
- `VOICE_RAG_BOOTSTRAP_INDEX=1` is safe to leave on: the builder verifies its state manifest and
  exits quickly once the version is complete. Qdrant storage is isolated per
  `VOICE_RAG_INDEX_VERSION`, so a stale collection cannot start its optimizer and starve a clean
  bootstrap sharing the volume.
- A **bootstrap lock** prevents two Pods on one volume from writing the same BM25 index. Recovery
  from a stale lock after a forced stop is opt-in
  (`VOICE_RAG_RECOVER_STALE_BOOTSTRAP_LOCK=1` for one restart), because clearing it wrongly
  corrupts the index.
- `/v1/health` reports ready only when the state manifest **and** Qdrant's exact point count agree,
  so an interrupted upload cannot appear ready merely because its collection exists. It also
  reports which generation backend is configured and whether it is reachable — a down model server
  degrades every generative-band query to extractive rather than failing, which is otherwise only
  visible as a mode mix that looks wrong.

Required secret: `SARVAM_API_KEY` (voice returns 503 without it). Generation needs no credentials —
the model is local.

The image builds `vLLM` into its own venv (`/opt/vllm-venv`) on Python 3.12, separate from the
application's venv. Both isolations are load-bearing and were found the hard way: vLLM's wheels
pull a CUDA 12.x torch build that would otherwise upgrade the application's pinned cu118 install,
and flashinfer annotates with `array.array[int]`, which is a syntax error before Python 3.12.

Environment knobs, all changeable without a rebuild:

| variable | default | effect |
|---|---|---|
| `VOICE_RAG_REQUEST_BUDGET_SECONDS` | `0.2` | the deadline itself |
| `VOICE_RAG_VLLM_MAX_TOKENS` | `14` | output cap; ~6.5–7.5ms per token |
| `VOICE_RAG_VLLM_CONTEXT_CHUNKS` | `2` | passages in the prompt; trades recall for prefill |
| `VOICE_RAG_VLLM_QUANTIZATION` | `fp8` | empty serves bf16 |
| `VOICE_RAG_VLLM_ENFORCE_EAGER` | `0` | `1` disables CUDA graphs if capture exhausts memory |
| `VOICE_RAG_VLLM_GPU_MEM_FRACTION` | `0.55` | vLLM's share of the GPU, alongside the retrieval models |
| `VOICE_RAG_LOW_CONFIDENCE_THRESHOLD` | `0.2` | below this, refuse rather than generate |

`Dockerfile.cloudrun` targets the same full-index architecture on Cloud Run and is unused.

## Repository layout

```
src/voice_rag/
  settings.py             # Env-var settings (SARVAM_API_KEY)
  pipeline/
    ingestion/            # MSMARCO-XI schema, HF access, dedup, dataset analysis
    chunking/             # Adaptive chunking strategies
    embeddings/           # BGE-M3 dense + learned-sparse service
    retrieval/            # Qdrant dense/sparse, Tantivy BM25, RRF fusion
    reranking/            # BGE cross-encoder
    guardrails/           # Safety filter, off-topic gate, NLI grounding
    generation/           # vLLM backend, factory, typed schemas, the harness
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
cp .env.example .env   # SARVAM_API_KEY
uv run python -m voice_rag.pipeline.ingestion.build_corpus --languages hi --split validation
uv run python -m voice_rag.pipeline.chunking.build_chunks --languages hi --split validation
uv run python scripts/build_full_index.py --language hi --split validation \
    --qdrant-url http://localhost:6333 --index-version full1
uv run uvicorn voice_rag.api.main:app --port 8000

# Frontend
cd frontend && npm ci && npm run dev   # proxies /v1 to :8000
```

Generation requires a vLLM server on `VOICE_RAG_VLLM_BASE_URL` (default
`http://127.0.0.1:8001/v1`); without one, the harness degrades to extractive answers rather than
failing. Or `docker compose up` for the two-service (Qdrant + app) topology.

## Tests

```powershell
uv run pytest -q      # 105 passed, 6 deselected (slow/GPU/provider), 2026-08-17
uv run pytest -m slow tests/test_ml_integration.py tests/test_sarvam_integration.py
```

The default suite is fast unit/component coverage. Slow tests load real ML models or call paid
providers and are opt-in.

## Known limitations

- **The inline-generation configuration is not benchmarked.** The n=150 figures were produced when
  generation was deferred out of band. Single-query timings and the arithmetic behind the 14-token
  cap are reported above, but a full re-benchmark has not been run.
- **Answer relevance is not checked; groundedness is.** A cross-encoder scores topical relatedness,
  so for a question whose answer is absent from the corpus it can rank a near-miss passage well
  above the refusal threshold. Generation then faithfully summarises it and the NLI validator
  correctly passes it, because the claim genuinely is entailed by the passage cited — and the user
  gets a well-grounded answer to a question they did not ask. Raising
  `VOICE_RAG_LOW_CONFIDENCE_THRESHOLD` trades recall against this; a real fix is a query-to-answer
  relevance check, which costs latency the budget does not currently have.
- **Answers inherit the corpus's vintage.** MSMARCO-XI passages have a fixed date, so a correctly
  grounded answer can be outdated. This is by design: the corpus is the authority, and a system
  that "corrects" it from model memory breaks the guarantee that answers trace to cited evidence.
- **14 output tokens is a short Hindi sentence.** Hindi tokenises at roughly 2–4 tokens per word,
  so longer answers truncate. `VOICE_RAG_VLLM_MAX_TOKENS` trades latency back for words.
- **Input-side safety lost a layer** when generation moved to a self-hosted model — see
  [Guardrails](#6-guardrails).
- MSMARCO-XI covers 14 Indic languages, but one API process serves one configured language
  collection. The request-level `language` field labels the response; it does not switch indexes.
- The retrieval sub-stage's P100 (336.4ms in-process) exceeds target on rare queries where several
  stage tails coincide, even with BM25 and sparse individually bounded.
- Latency depends materially on GPU: P50 55.8ms on a 4090 versus 74.1ms on a 4060 Laptop, same code
  and index.
- Half the generated answers in the deployed run were refused by the grounding check (14 of 28).
  That is the guardrail working, but it also indicates retrieval surfaces topically-close passages
  that do not support a specific claim.
- No full `TestClient` coverage of `/v1/query` or `/v1/voice-query`. The refinement store's
  eviction, TTL and single-use rules are unit-tested; endpoint wiring is verified against the live
  deployment rather than in CI.
