# Implementation Roadmap, Differentiators & Final Architecture

## Implementation roadmap

**Phase 01 — Dataset analysis**
Objective: understand MSMARCO-XI before designing anything.
Tasks: schema profiling, language distribution, passage-length histograms, duplicate/near-duplicate detection, translation-quality spot audit using the decoding-param metadata.
Modules: `ingestion/profile.py`. Tests: schema assertions per split. Output: dataset report feeding chunking/architecture decisions.

**Phase 02 — Data ingestion**
Objective: normalized internal representation.
Tasks: parquet loaders, per-language partitioning, `is_selected` qrel extraction.
Modules: `ingestion/`. Dependencies: Phase 1. Output: normalized internal passage table.

**Phase 03 — Chunking experiments**
Objective: validate the chunking strategy against real data.
Tasks: implement strategies A/B/D, granularity classifier.
Modules: `chunking/`. Tests: boundary correctness on multilingual sentence splitting. Output: chunk manifest + hashes (feeds provenance).

**Phase 04 — Embedding/indexing**
Objective: stand up the retrieval substrate.
Tasks: BGE-M3 serving (ONNX+INT8), Qdrant collection build, payload indexing.
Modules: `embeddings/`, `retrieval/dense/`. Dependencies: Phase 3. Perf: benchmark embed throughput before committing to worker count.

**Phase 05 — Hybrid retrieval**
Objective: dense + sparse + fusion working end to end.
Tasks: Tantivy BM25 index, RRF fusion, per-language filtering.
Modules: `retrieval/sparse/`, `retrieval/fusion/`. Tests: Recall@10 vs. dense-only baseline on eval set.

**Phase 06 — Reranking**
Objective: cross-encoder reranking within budget.
Tasks: bge-reranker-v2-m3 serving, batching, fallback-skip logic.
Modules: `reranking/`. Perf: confirm <25ms batched on target GPU class.

**Phase 07 — STT integration**
Objective: streaming voice input.
Tasks: Sarvam streaming client, partial/final event handling, ElevenLabs fallback adapter.
Modules: `streaming/`, `apps/api-gateway/`. Tests: WER on held-out Indic speech samples.

**Phase 08 — LLM harness**
Objective: typed orchestration layer.
Tasks: typed schemas, extractive/generative router, structured-output decoding, timeouts/circuit breakers.
Modules: `generation/`. Dependencies: Phases 4–6.

**Phase 09 — Guardrails**
Objective: safety and grounding.
Tasks: off-topic centroid gate, moderation classifier, NLI grounding validator, conflict detection.
Modules: `guardrails/`. Dependencies: Phase 8.

**Phase 10 — Streaming / speculative retrieval**
Objective: the perceived-latency differentiator.
Tasks: debounce logic, cancellation tokens, reconciliation.
Modules: `streaming/`. Dependencies: Phases 5–7.

**Phase 11 — Latency optimization**
Objective: validate the latency budget against real measurements.
Tasks: quantization pass, cache layers, warm-pool sizing, cold-start elimination.
Modules: `benchmark/`, cross-cutting. Output: latency budget validated, not assumed.

**Phase 12 — Evaluation**
Objective: quantify retrieval/RAG/speech quality.
Tasks: retrieval/RAG/speech metric harnesses, per-language dashboards.
Modules: `evaluation/`. Dependencies: Phases 5, 8, 9.

**Phase 13 — Web3 provenance**
Objective: verifiable corpus/pipeline commitments.
Tasks: chunk/manifest hashing, Merkle tree builder, Base contract + anchoring job, proof-verification endpoint.
Modules: `provenance/`, `web3/`.
Note: deliberately late in the roadmap — it's an audit/trust feature layered on a corpus that's already stable, not a dependency for core RAG quality.

**Phase 14 — Observability**
Objective: production-grade tracing and metrics.
Tasks: OTel instrumentation across all stages, Prometheus exporters, Grafana dashboards.
Modules: `observability/`.

**Phase 15 — Deployment**
Objective: ship it.
Tasks: IaC for Qdrant/Postgres/Redis/GPU pools, gateway rollout, failure-injection drills, load test at target concurrency.
Modules: `infrastructure/`.

## Differentiators

**01 — Speculative retrieval from partial speech, with real reconciliation.**
Not just "start STT early" — the system runs actual dense retrieval on
unstable partial transcripts, debounces to avoid wasted work, and
reconciles against the final transcript via candidate-set overlap rather
than blindly discarding speculative work. This is a genuine latency win for
the retrieval subsystem, not a cosmetic one.

**02 — Confidence-routed generation: extractive fast-path vs. LLM path.**
High-confidence queries never touch the generator — they get a cited,
templated extractive answer. This is simultaneously a latency optimization
and a hallucination-risk reduction, since an extractive answer is
structurally incapable of inventing a claim.

**03 — Single-encoder hybrid retrieval.**
Using BGE-M3's dense + sparse + multi-vector output from one model call,
instead of standing up a separate SPLADE service, removes an entire
model-serving stack's worth of latency-tail risk and operational burden — a
genuine architecture simplification, not just a model swap.

**04 — Per-claim, entailment-scored grounding.**
Hallucination detection at the sentence/claim level with an explicit NLI
score per citation — not a single opaque "groundedness: 0.8" number for the
whole answer — is what actually lets a guardrail selectively drop one bad
sentence instead of discarding an otherwise-correct answer.

**05 — Index-version Merkle anchoring, not per-query blockchain theater.**
Anchoring one root hash per corpus/pipeline build gives a real, cheap,
verifiable answer to "was this index built from an approved, unmodified
dataset and config" — the actual trust question a knowledge-base operator
has — without touching latency, cost-per-query, or user privacy. It's
deliberately the only place Web3 appears in this system.

## Risks & trade-offs

- **The 200ms number will be misread as "voice-to-voice."** It isn't, and
  stakeholders need the time-to-first-evidence vs. time-to-first-audio-token
  distinction stated up front, not discovered at demo time.
- **Self-hosting BGE-M3 + reranker on warm GPU pools** is real
  infrastructure a small team must own — GPU availability, autoscaling, and
  quantization validation are ongoing work, not a one-time setup.
- **Sarvam is a single external vendor dependency** for the core voice
  input path; the ElevenLabs fallback adapter mitigates but doesn't
  eliminate this, and both providers' reliability/pricing can change.
- **MSMARCO-XI's machine-translated text** may under-perform
  native-authored text on embedding and BM25 quality, particularly for
  morphologically complex languages — per-language eval is not optional,
  it's how this gets caught.
- **14-language evaluation multiplies QA surface** — a global metric can
  hide a badly-served language; dashboards must default to per-language
  breakdowns, not aggregates.
- **Web3 adds real operational surface** (key management for the anchoring
  account, RPC provider dependency, gas price monitoring) for a benefit
  that is audit/trust-only, not core product quality — correctly sequenced
  last (Phase 13) so it never blocks core RAG quality work.
- **License review is required** before any commercial use — MSMARCO-XI
  inherits MS MARCO's original (non-commercial research) terms.
- **The grounding validator's NLI model has uneven language coverage** —
  verified against MoritzLaurer/mDeBERTa-v3-base-mnli-xnli's actual XNLI
  fine-tuning languages: only Hindi and Urdu of MSMARCO-XI's 14 languages
  were directly fine-tuned on; the other 12 rely on the base encoder's
  zero-shot cross-lingual transfer, a real but weaker guarantee. Per-language
  grounding-accuracy eval (not just retrieval eval) is needed before trusting
  this guardrail uniformly across all 14 languages — see
  [Evaluation Results](evaluation-results.md).

## Estimated infrastructure

| Component | Sizing (moderate traffic, ~50–100 peak QPS) |
|---|---|
| GPU inference (embedder + reranker + small local LLM) | 2–4× L4/A10-class nodes, warm pool, autoscaled |
| Qdrant | 3-node cluster (HA) or managed Qdrant Cloud; this corpus (~11.5M rows worth of passages, subset actually indexed) fits comfortably below the 10M-vector tier used in the cited p99 benchmarks |
| Postgres (app DB + provenance) | Managed instance, small-to-medium tier, read replica for evaluation workloads |
| Redis | Managed, single primary + replica, sized for embedding/retrieval/answer cache working set |
| Object storage | S3/MinIO for provenance manifests, evaluation artifacts, ephemeral audio buffering |
| Observability | Self-hosted Prometheus/Grafana or hosted OTel backend |
| Web3 | Base RPC provider subscription (or self-run node), negligible on-chain gas spend given batch anchoring |

Order-of-magnitude only — actual sizing depends on measured throughput from
Phase 11's load tests, not assumed up front.

## Final recommended architecture

Sarvam-driven streaming STT feeding a speculative, debounced query
pipeline; BGE-M3 as a single dense+sparse encoder over Qdrant, fused with an
independent Tantivy BM25 signal via RRF; a quantized multilingual
cross-encoder reranking the top-20 down to a context-budget-sized set; a
confidence-routed generator that answers extractively when it safely can
and defers to a small local LLM (or Claude Haiku 4.5 via API) only when
synthesis is genuinely required; per-claim NLI grounding and
topic/confidence guardrails gating every response before it ships; and a
Merkle-anchored, Base-anchored provenance layer that makes the corpus and
pipeline configuration independently verifiable — without a single byte of
user data ever touching a chain. Latency is optimized honestly: sub-200ms
for retrieval and evidence, streamed and clearly labeled as such for
generation. This is the shape of a system built to be measured, audited,
and trusted — not just demoed.
