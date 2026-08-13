# Latency Budget & Caching

## A critical read of "<200ms"

> **The honest version of this requirement:** <200ms is achievable for
> STT-partial → retrieval → fusion → rerank → evidence. It is **not**
> achievable for a fully generated, multi-sentence, grounded natural-language
> answer — a 7–8B parameter model streaming even 60 tokens at a realistic
> 20–40ms/token adds 1.2–2.4s by itself, before accounting for
> time-to-first-token. Any claim that a complete voice-in → spoken-answer-out
> round trip lands under 200ms is not credible with current LLM decoding
> speeds. The architecture instead optimizes two different, honestly-labeled
> numbers: **time-to-first-evidence** (target <200ms) and
> **time-to-first-audio-token** of the streamed answer (target 500–700ms),
> with the full answer arriving progressively rather than all at once.

## Stage-by-stage budget

| Stage | Budget | How it's achieved | In <200ms window? |
|---|---:|---|---|
| STT partial emission | 0ms (async) | Runs continuously on the audio stream; not on the request's critical path. | n/a |
| Query processing (lang-id, normalize) | 2–4ms | Deterministic, in-process, no model call for the default path. | Yes |
| Query embedding (BGE-M3) | 8–15ms | Quantized (INT8) ONNX model on a warm GPU worker; short query text. | Yes |
| Dense retrieval (Qdrant HNSW) | 5–12ms | p99 ~12ms at 10M-vector scale on Qdrant's Rust engine; this corpus is well under that. | Yes |
| Sparse retrieval (BGE-M3 sparse + Tantivy BM25) | 3–8ms | Runs concurrently with dense, not after it. | Yes |
| Fusion (RRF) | <1ms | Rank-only arithmetic over ≤40 candidates. | Yes |
| Reranking (cross-encoder, top-20) | 15–25ms | Batched INT8 ONNX on a warm GPU worker; degrades to skip-rerank on saturation. | Yes (GPU) / marginal (CPU) |
| Guardrail pre-checks (off-topic / low-confidence gate) | 2–5ms | Threshold comparisons + cached centroid lookup, no model call on the fast path. | Yes |
| **Subtotal: time-to-first-evidence** | **~35–70ms** | Comfortably inside 200ms, leaving headroom for network jitter. | |
| Extractive answer (high-confidence path) | +5–10ms | Template fill, no generation model. | Yes |
| Generative answer, time-to-first-token | +150–300ms | Small local model (quantized 7–8B) or fast API model, warm connection. | Marginal / usually no |
| Generative answer, full stream (≈60 tokens) | +1,200–2,400ms | Bounded by decode speed, not by this architecture — streamed to the client/TTS progressively. | No, by design |
| Grounding validation (NLI, per claim) | +10–20ms per claim | Small cross-encoder, runs on claims as they're emitted, overlapped with streaming. | Overlapped, not additive |

Aggressive-optimization levers actually in use: warm GPU worker pools (no
cold model load per request), INT8 quantization on embedder/reranker,
embedding cache and retrieval cache (below) that turn repeat/similar queries
into near-zero-cost lookups, speculative retrieval on partial transcripts
that hides retrieval latency behind the user still speaking, and an
extractive fast-path that avoids generation entirely when confidence is
high.

What is explicitly **not** claimed: that a from-scratch multi-sentence LLM
answer fits in 200ms — it doesn't, on any current serving stack, and
pretending otherwise would be the least credible part of this design.

## Caching

| Layer | Key | Invalidation |
|---|---|---|
| Query embedding cache | `sha256(normalized_query + embed_model_version)` | TTL 24h; hard-invalidated on embed model version bump. |
| Retrieval cache | `sha256(query + top_k + filters + index_version)` | Invalidated whenever `index_version` changes (new corpus build) — never served stale across an index rebuild. |
| Answer cache | `sha256(context_chunk_ids + query + model_version + prompt_version)` | Only a hit if *all four* match — context hash included specifically so a reranked-context change invalidates a cached answer even if the query string is identical. |
| Prefix / streaming cache | Partial-transcript prefix → last speculative candidate set | Per-utterance, in-memory, discarded at end-of-utterance or on reconciliation. |

All cache keys are versioned by pipeline component (embed model, index,
prompt) rather than time-only TTLs where correctness matters — a stale cache
hit that silently mixes an old index with a new prompt version is a worse
failure mode than a cache miss.
