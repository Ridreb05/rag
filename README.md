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

Answering is **two-phase**, because a live LLM call and a 200ms budget cannot share one response:
phase one returns a grounded, cited answer inside the budget (deployed P50 **55.8ms**, 150/150
under 200ms), and phase two (`POST /v1/query/refine`) upgrades it with Gemini's synthesis a few
seconds later. See [Two-phase answering](#two-phase-answering--how-200ms-and-a-real-llm-coexist).

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

**What the strategies actually did to this corpus**, counted across all 964,603 indexed chunks
rather than asserted:

| strategy | chunks | share |
|---|---:|---:|
| `whole_passage` | 951,816 | 98.7% |
| `fixed_token_fallback` | 6,606 | 0.7% |
| `sentence_aware` | 6,181 | 0.6% |

953,388 passages produced 964,603 chunks — **1.012 chunks per passage**, with a median chunk of 56
tokens (p90 94, p99 246, max 512).

Read that honestly: **on this dataset, chunking is a no-op 98.7% of the time.** That is the
correct outcome, not a missing feature — MSMARCO-XI ships pre-segmented passages that are already
retrieval-sized, and splitting a 56-token passage would fragment context and hurt retrieval for no
gain. The splitting strategies exist for the 1.3% that genuinely exceed the budget, and the
decision of *which* passages those are is made per passage from measured token counts, not
assumed. Pointed at long-form documents, the same code splits the overwhelming majority instead —
that is what the `level`/`parent_id` hierarchical hook is reserved for, and it is inert here
because this dataset has no document/section structure to exploit.

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
2. **Deadline-aware routing** — the harness is given what remains of the request's 200ms budget
   and will not start a generation call it cannot finish in time, degrading to the top reranked
   passage instead (flagged `deadline_exceeded_extractive_fallback`, never silent). That flag is
   also what makes the answer eligible for phase-two refinement, so the deadline defers generation
   rather than cancelling it — see [Two-phase answering](#two-phase-answering--how-200ms-and-a-real-llm-coexist).
3. **Confidence-routed answering** — the reranker's top score decides the path:
   - `< 0.2` → refuse outright (below this, retrieval didn't find real support).
   - `0.2 – 0.85` → generate, via Gemini (`gemini-flash-latest`) with structured output
     (`generationConfig.responseSchema` against a JSON schema: `answer_text` plus per-claim
     `cited_chunk_ids`) — this is what makes citations machine-checkable downstream, not
     free-text parsing.
   - `≥ 0.85` → answer extractively (return the top passage directly), skipping the LLM call
     entirely when retrieval is already confident.
4. **Error recovery** — Gemini's own safety filtering returning zero candidates or a blocked
   `finishReason` (`SAFETY`/`PROHIBITED_CONTENT`/`BLOCKLIST`/`RECITATION`) is treated as a
   guardrail outcome, not a crash. Since Gemini is called via a raw REST API rather than a
   provider SDK, retry/backoff for transient failures (network errors, 429, 5xx) is implemented
   explicitly in `gemini_service.py`, with exponential backoff and immediate failure on
   non-retryable 4xx errors.
5. **Grounding validation** (when loaded) — every generated claim is re-checked against its
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

### Headline: measured against the live deployment

The primary numbers come from the **deployed GPU Pod itself** — real HTTPS requests to
`POST /v1/query`, full 964,603-chunk `full1` index, RTX 4090, with the API's own 20 req/60s rate
limiter left enabled rather than bypassed for the measurement. The figure is the server's own
`pipeline_ms`, so it excludes internet round-trip to the machine running the benchmark but
includes every cost the deployment actually pays. n=150.

| | P50 | P70 | P100 |
|---|---|---|---|
| **ms** | **55.8** | **61.2** | **172.8** |

**150 of 150 queries completed inside the 200ms budget — worst case included.** Mode mix: 124
extractive, 26 refused. Raw output: `reports/latency_benchmark/hi_full1_deployed.json`.

Of those 150, **28 were offered a phase-two refinement** (see Two-phase answering): 14 produced a
generative answer, and 14 were **refused after generation** because the grounding check rejected
the claims. Phase two costs P50 2.76s / P100 4.11s and is deliberately outside the budget —
nobody waits on it.

### Cross-checked in-process

`benchmark/latency_benchmark.py` measures the same pipeline in-process against a local Qdrant
server, which isolates stage-by-stage cost. Hardware: RTX 4060 Laptop GPU. 1000 queries for the
retrieval sub-stage, 150 for the full window — all reported, including the slow ones, not a
best-case sample.

| | P50 | P70 | P100 |
|---|---|---|---|
| **ms** | 74.1 | 79.0 | 131.5 |

Slower at P50 than the deployment above because the 4060 Laptop is a slower GPU than the 4090 —
same code, same index. Both are reported rather than only the flattering one.

**Retrieval sub-stage only** (embed → dense+sparse+BM25 → RRF fuse → rerank), the largest slice of
that window, broken out so the cost is attributable rather than a single opaque total, n=1000:

| | P50 | P70 | P100 |
|---|---|---|---|
| **ms** | 84.9 | 95.2 | 336.4 |

Retrieval's own P100 is the one number still above target: a rare query where several stages hit
their tails at once. It is bounded (was 623.3ms before the BM25 bound, 457.9ms before the sparse
bound) but not under 200ms. The full-window figure above is lower because it is a different,
smaller sample (n=150) that doesn't contain that pathological query — both are reported rather
than quoting only the flattering one. Full per-stage P95/P99 breakdown, tracked in this repo so
every number above is checkable rather than asserted: `reports/latency_benchmark/hi_full1.json`.

### What the 200ms covers

The target is **chunking + vector DB retrieval + through to final output** — the third and fourth
stages of the task's own pipeline (`Voice input → Speech-to-text → Chunking/Retrieval (vector DB)
→ Answer generation`). That is the window measured everywhere in this section, reported per
request as `pipeline_ms`, and enforced by `REQUEST_BUDGET_SECONDS` in `api/main.py`:

| stage | in the 200ms window | cost |
|---|---|---|
| Chunking | yes | 0ms per query — see below |
| Embedding → dense + sparse + BM25 → RRF fusion → rerank | yes | the bulk of it |
| Guardrails → extractive-or-generative answer | yes | the rest |
| Speech-to-text (Sarvam) | no — upstream of the window | reported as `stt_ms` |

**Chunking is inside the window and costs 0ms per query.** MSMARCO-XI ships pre-segmented
passages, so chunking runs once when the index is built (`pipeline/chunking/`) and no query
repeats it. That is a real architectural property — per-query work amortised into an offline index
build — but it follows from this dataset already being passage-sized, so it is stated rather than
claimed as a latency win.

**Speech-to-text is outside it.** The target's clause starts at chunking, so voice capture and
transcription sit ahead of the measured window. It is still reported, never hidden: `stt_ms` on
the response and on the result card, and `total_ms` (= `pipeline_ms + stt_ms`) for the full
wall-clock cost of a voice query.

## Two-phase answering — how <200ms and a real LLM coexist

A live LLM call and a 200ms budget cannot both fit in one response. Measured Gemini generation is
~2.1s median (P50 2.76s from the deployment); no deadline arithmetic makes that fit 200ms. The
usual ways out are to drop generation (fast, but then it is not really RAG) or to accept a
multi-second P100 (real RAG, misses the target). This system does neither.

**Phase one** answers from what retrieval already earned, inside the budget:

1. `REQUEST_BUDGET_SECONDS` (default `0.2`) is a deadline carried through the request — not a
   per-stage timeout, since a pipeline of individually-bounded stages still has an unbounded total.
2. Before committing to generation, the harness checks what is actually *left* of that budget. If
   an LLM call cannot finish in time, it returns the top reranked passage instead — a grounded,
   cited answer, flagged `deadline_exceeded_extractive_fallback` so the downgrade is visible.

**Phase two** then does the work that did not fit. `POST /v1/query/refine` re-runs generation
against the candidates the request already retrieved and reranked, and the UI swaps the synthesized
answer into the same card. It carries no deadline: nobody is blocked on it, because the user
already has an answer.

The result, measured on the deployment: **the response a user waits for is P50 55.8ms / P100
172.8ms, 150/150 inside budget** — and 14 of the 28 eligible queries still received a real
generative answer a few seconds later.

**Refinement is offered only when generation was skipped for *time*.** A high-confidence
extractive answer (`≥0.85`) is never refined — the router skipped the LLM because a single passage
already answers the question, not because it ran out of budget. A refusal is not refined either:
there is nothing grounded to improve.

Two details worth stating because they are where this design gets fragile:

- **Phase two gets its own generator with a 45s retry budget**, not the in-request 8s. Measured
  directly: the 8s budget expired mid-generation on a slower network and degraded the refinement
  back into the same extract it existed to improve. A budget sized for a caller who is waiting is
  the wrong budget for a caller who is not.
- **Pending refinements are TTL'd (5 min) and bounded (256)**. A dict keyed by a client-supplied
  `trace_id` is otherwise a memory leak. It is per-process by nature, so with multiple workers a
  refine can land on a worker that never saw the query; that returns 404 and the client simply
  keeps the fast answer already on screen.

Guardrails apply in phase two exactly as in phase one — and this is where they earn their keep.
Of the 28 refinements in the deployed run, **14 were refused after generation** because the NLI
grounding check rejected the model's claims. The system generated, checked its own output against
the retrieved evidence, and declined to show half of it.

### Engineering notes

Each change below came from per-stage profiling and was verified to leave results unchanged — or,
where results did change, verified that the change was an improvement. Absolute numbers across
these notes come from more than one measurement session and, in one case, more than one GPU, so
they are quoted as the before/after deltas actually observed for each fix rather than stitched
into a single running total that no single benchmark run would reproduce:

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

- **Qdrant's sparse search had the same tail as BM25, and was the single largest contributor to
  retrieval's worst case — P100 386.0ms → 101.4ms.** Profiling the stage breakdown (rather than
  assuming BM25 was still the culprit) showed learned-sparse retrieval at p50 6.5ms but p100
  386.0ms: a ~59x typical-to-worst gap, from the same pathology as BM25's, since a learned-sparse
  query is also a union over per-term posting lists and high-document-frequency Hindi terms
  produce very long ones. Bounded to the same 100ms budget. Dense is deliberately left unbounded —
  it is the primary semantic signal, the only one guaranteed to return something for any query,
  and its tail is far smaller (p100 106.3ms) because HNSW's cost doesn't scale with term
  frequency. Retrieval P100 457.9ms → 336.4ms as a result.
- **The 200ms target became a real deadline instead of a post-hoc measurement — full-window P100
  3434.5ms → 131.5ms in-process, 172.8ms on the deployment.** Individually bounding each stage
  still allows an unbounded total, so the budget is now carried through the request: the harness
  receives what is actually left of it and pre-empts generation it cannot finish in time, falling
  back to the already-retrieved top passage.
- **Then the trade-off it created was removed rather than documented.** Pre-empting generation met
  the deadline at the cost of never generating — a bad answer to the task, which asks for both.
  Two-phase answering keeps the sub-200ms response *and* the LLM (see above): 150/150 deployed
  queries inside budget, 14 real generative answers delivered in phase two.

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

Two deployments, for two different jobs:

| | Fly.io (`Dockerfile.fly`) | RunPod GPU Pod (`Dockerfile`) |
|---|---|---|
| index | `api_smoketest`, ~40k chunks (~4% of full1) | **full1, all 964,603 chunks** |
| hardware | shared CPU | GPU |
| shows | that the system answers real queries end to end | the real pipeline at its designed scale and speed |
| always on | yes | no — started when needed, stopped after |

The Fly link is the permanent one because it is cheap enough to leave running; the RunPod Pod is
where the pipeline actually performs as designed, and is what the Latency section measures.

### RunPod GPU Pod — the full-index deployment

`infrastructure/runpod-entrypoint.sh` runs Qdrant on localhost and uvicorn on port 8000 behind
RunPod's HTTPS proxy, in one container. It is written to survive stop/start on a persistent network
volume, which is what makes an on-demand Pod practical rather than a 24/7 cost:

- Attach a **network volume** at `/workspace` (the index and HF model cache live there, so a
  restart re-uses both instead of re-downloading ~3.8GB of weights and rebuilding a 9GB index).
- `VOICE_RAG_DATA_ROOT` (default `/workspace/voice-rag`) — where corpus, Qdrant storage and the
  model cache persist.
- `VOICE_RAG_BOOTSTRAP_INDEX=1` on **first** boot only: builds the index, writes a resumable state
  manifest after every committed batch, and exits quickly on later boots once the manifest says the
  version is complete. Qdrant storage is isolated per `VOICE_RAG_INDEX_VERSION` so a stopped legacy
  collection cannot start its optimizer and starve a clean bootstrap sharing the volume.
- A bootstrap lock directory prevents two Pods on one volume from writing the same BM25 index
  concurrently. Recovery from a stale lock after a forced stop is deliberately opt-in
  (`VOICE_RAG_RECOVER_STALE_BOOTSTRAP_LOCK=1` for a single restart) rather than automatic, because
  clearing it wrongly corrupts the index.
- `/v1/health` reports ready only when the state manifest *and* Qdrant's exact point count agree,
  so an interrupted upload cannot look ready merely because its collection exists.

**Generation and the latency budget.** `VOICE_RAG_REQUEST_BUDGET_SECONDS` (default `0.2`) is the
deadline from the Latency section. At the default, a ~2.1s Gemini call cannot fit, so answers stay
extractive — correct for the latency claim, but it means the generative path never runs. To
exercise the full generative pipeline (structured output, grounding validation) set it high, e.g.
`VOICE_RAG_REQUEST_BUDGET_SECONDS=10`. Both are real configurations of the same code; only that one
budget changes.

To re-run the benchmark against the Pod, point it at the local Qdrant and BM25 paths:

```bash
uv run python benchmark/latency_benchmark.py --language hi --index-version full1 \
    --qdrant-url http://127.0.0.1:6333 --bm25-path "$VOICE_RAG_DATA_ROOT/data/full_index/bm25/hi_validation_full1"
```

### Fly.io — the always-on demo

Runs CPU-only, with the smaller `data/api_smoketest/` index (embedded Qdrant, ~40k chunks) so it
fits a modest, affordable machine. It exists to prove the system answers real voice and text
queries end to end, and two things follow from that scope — both stated here rather than left for
a visitor to discover:

- **It refuses often, and that is correct.** ~40k chunks is about 4% of the corpus, so most
  queries genuinely have no supporting passage indexed. Measured against 20 real validation
  queries: 9 answered, 11 refused. On the full index the same routing answers the large majority
  (see the mode mix in Latency). The refusals demonstrate the confidence gate working, not a
  broken retriever.
- **It will not hit 200ms, and the UI does not pretend otherwise.** On shared CPU the
  cross-encoder reranker alone takes ~16s versus ~44ms on GPU, so `pipeline_ms` lands far over
  budget and every result card renders that over-budget state honestly. The <200ms claim belongs
  to the GPU numbers above; this link demonstrates correctness, not latency.

`Dockerfile.cloudrun` targets the same full-index GPU architecture on Cloud Run, unused here for
reasons unrelated to the code (regional payment-processing failures with Cloud Billing).

The off-topic centroid gate (see Guardrails) is disabled specifically on this CPU deployment
(`VOICE_RAG_LOAD_OFF_TOPIC_GATE=0` in `Dockerfile.fly`) — measured directly: building it re-embeds
a 63-text sample locally, which took 30+ minutes on this account's shared-CPU machines under memory
pressure. It stays on by default everywhere else (`main.py`'s default is `"1"`); confidence-based
refusal and every other guardrail remain active here regardless.

## Tests

```powershell
uv run pytest -q   # 102 passed, 7 deselected (slow/GPU/provider tests) as of 2026-08-16
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
  default fast unit-test suite alone is not live end-to-end proof. The two-phase store's eviction,
  TTL and single-use rules are unit-tested (`tests/test_refinement_store.py`); the endpoint wiring
  around them is verified against the live deployment rather than in CI.
- The <200ms target is met by the response the user waits for, not by a single request that also
  contains LLM synthesis. Two-phase answering is a real resolution of that conflict rather than a
  reframing of it — but the synthesized answer does arrive seconds later (P50 2.76s), and anyone
  reading the target as "one request, LLM included, under 200ms" should know that no configuration
  of this system does that, because no current LLM serving stack can.
- The retrieval sub-stage's own P100 (336.4ms, in-process on the 4060) is above target on a rare
  query that hits several stage tails at once, even with BM25 and sparse search individually
  bounded. The deployed full-window P100 (172.8ms) stays inside budget because the deadline
  degrades that query rather than letting it run long.
- Latency depends materially on GPU: P50 55.8ms on the deployed 4090 versus 74.1ms on an RTX 4060
  Laptop, same code and index. Both are reported above; neither is presented as the only number.
- Half of all generated answers in the deployed run were refused by the grounding check (14 of
  28). That is the guardrail working, but it also means retrieval frequently surfaces passages
  that are topically close without supporting a specific claim — a retrieval-precision limit, not
  only a generation one.
