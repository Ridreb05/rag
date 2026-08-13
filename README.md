# Voice RAG over MSMARCO-XI, with Web3 where it earns its place

A production-grade architecture blueprint for a 14-language Indic voice search
system — hybrid retrieval, streaming speech, grounded generation with
per-claim citations, and cryptographic provenance anchored only where it adds
real verifiability.

> **STT:** Sarvam Saaras · **Embeddings:** BGE-M3 · **Vector DB:** Qdrant ·
> **Anchor chain:** Base L2 · **Target:** <200ms retrieval, streamed generation

This is a **technical blueprint**, not an implementation. Dataset:
[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).

Also published as a formatted artifact: https://claude.ai/code/artifact/4fd769cb-bd72-4252-a145-76074928e6e1

## Executive summary

This blueprint rejects the naive `voice → embed → search → LLM` pipeline in
favor of five concrete engineering bets, each justified against the actual
dataset and a critical read of the latency target:

- **Hybrid retrieval from one encoder.** BGE-M3 emits dense, sparse (lexical),
  and multi-vector representations from a single forward pass, so hybrid
  search doesn't require running two separate model stacks.
- **Speculative retrieval on partial speech.** Because Sarvam's streaming STT
  emits partial transcripts every ~200–300ms, retrieval starts before the
  user finishes speaking and is reconciled — not restarted — against the
  final transcript.
- **An honest latency budget.** Sub-200ms is achievable for
  STT-partial → retrieval → fusion → rerank → first-evidence. It is *not*
  achievable for a fully generated, multi-sentence grounded answer from an
  LLM — that stage is streamed instead, and the UX is designed around that
  truth rather than hiding it.
- **Provenance, not surveillance.** Web3 is used for exactly one job — a
  Merkle-anchored, versioned commitment to the corpus + pipeline
  configuration on a cheap L2 (Base) — and nowhere else. No user data, no
  vectors, no per-query writes ever reach a chain.
- **Evidence-level grounding.** Every generated sentence is checked against
  its cited chunk with an NLI entailment model before the response leaves
  the guardrail stage, not just scored globally.

## Contents

| # | Document | Covers |
|---|---|---|
| 01 | [Dataset Analysis](docs/01-dataset-analysis.md) | Schema, languages, relevance labels, data-quality risks |
| 02 | [Architecture & Retrieval](docs/02-architecture-and-retrieval.md) | System diagram, chunking, hybrid retrieval, reranking, query understanding, STT selection, streaming |
| 03 | [Harness & Guardrails](docs/03-harness-and-guardrails.md) | LLM orchestration, typed schemas, guardrails, grounding, citations |
| 04 | [Latency & Caching](docs/04-latency-and-caching.md) | Full latency budget breakdown, caching layers |
| 05 | [Web3 & Privacy](docs/05-web3-and-privacy.md) | Provenance anchoring, what Web3 is *not* used for, privacy/security model |
| 06 | [Data & API](docs/06-data-and-api.md) | Database schema, API design |
| 07 | [Observability & Evaluation](docs/07-observability-and-evaluation.md) | Tracing/metrics, evaluation framework, benchmark methodology, failure injection |
| 08 | [Repo & Stack](docs/08-repo-and-stack.md) | Repository structure, technology stack decisions, experiment matrix |
| 09 | [Roadmap & Summary](docs/09-roadmap-and-summary.md) | 15-phase implementation roadmap, differentiators, risks, infra estimate, final architecture |
| — | [Evaluation Results](docs/evaluation-results.md) | **Real measured numbers** from running the actual pipeline on real data — not projections |

## Build status

This isn't just a design document — Phases 1–6 are implemented and run
against real MSMARCO-XI data (see `src/voice_rag/`, `evaluation/`, and
[Evaluation Results](docs/evaluation-results.md)):

| Phase | Status |
|---|---|
| 1. Dataset analysis | Done — real findings in [docs/01-dataset-analysis.md](docs/01-dataset-analysis.md) |
| 2. Ingestion + dedup | Done — `voice_rag.ingestion` |
| 3. Chunking | Done — `voice_rag.chunking`, empirically 99.6%+ of passages need no split |
| 4. Embeddings + indexing | Done — BGE-M3 + Qdrant (embedded local mode), `voice_rag.embeddings`, `voice_rag.retrieval.dense` |
| 5. Hybrid retrieval | Done — BM25 (Tantivy) + RRF fusion, `voice_rag.retrieval.sparse`, `voice_rag.retrieval.fusion` |
| 6. Reranking | Done — bge-reranker-v2-m3, measured MRR +0.14 over fusion alone |
| 7. STT (Sarvam) | Done — `voice_rag.stt`, real TTS→STT round-trip verified (exact transcript match); not yet wired into the API gateway as a streaming endpoint |
| 8. LLM harness | Done — `voice_rag.generation`, confidence-routed extractive/generative harness. **Two backends**: Claude (`anthropic_service.py`, code-verified correct, blocked on Anthropic account credits) and Gemini (`gemini_service.py`, **working now**, verified end-to-end with real grounded citations) |
| 9. Guardrails | Done — off-topic/confidence gate + NLI grounding validator, wired into the harness and verified via the live API |
| API gateway | Done — `voice_rag.apps.api_gateway`, `POST /v1/query` ties retrieval → rerank → guardrails → generation together, with per-IP rate limiting. Verified end-to-end against a real isolated index (both the refusal path and the grounded-answer path) |
| Deployment | Dockerfile + docker-compose + [RunPod deployment guide](docs/runpod-deployment.md) written, matched to the measured latency requirements |
| 10, 13 | Not started — voice streaming endpoint, Web3 anchoring (deferred by explicit user decision — needs a funded testnet wallet) |

Full-corpus indexing (~965K chunks for Hindi) is running as a background
batch job — measured at ~110–140 chunks/sec on the dev GPU (~2–2.5 hours
total). A live progress log is in `data/full_index_build.log`. The
evaluation numbers above and the end-to-end API tests both run on real,
verified subsets built the same way (`evaluation/run_subset_eval.py`,
`scripts/build_smoketest_index.py`) while the full index completes —
**never point two processes at the same local-mode Qdrant path
concurrently** (verified: it corrupts the writer — see
[Evaluation Results](docs/evaluation-results.md)).

**Six real bugs found and fixed this session** (not simulated — each one
broke a real run first): a Tantivy query-injection crash on ordinary text
containing a hyphen, a FlagEmbedding/transformers version incompatibility,
concurrent access corrupting a local-mode Qdrant index, a Windows
git-bash UTF-8 encoding artifact that produced a false guardrail refusal
during testing, a Gemini API key being logged in plaintext (fixed:
header-based auth + suppressed HTTP logging), and Qdrant's embedded local
mode giving 5-10x inflated latency numbers versus the real server. Full
writeups in [docs/evaluation-results.md](docs/evaluation-results.md).

## Sources consulted

- [Sarvam streaming STT docs](https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/streaming-api), [Sarvam STT](https://www.sarvam.ai/speech-to-text)
- [ElevenLabs Realtime STT](https://elevenlabs.io/realtime-speech-to-text-api), [ElevenLabs India](https://elevenlabs.io/india)
- [ai4bharat/MSMARCO-XI dataset card](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
- [IndicRAGSuite paper (arXiv:2506.01615)](https://arxiv.org/abs/2506.01615)
- 2026 vector-DB p99 benchmark roundups (Qdrant/Weaviate/Milvus/pgvector)
- 2026 Ethereum L2 gas-fee comparisons (Base/Linea sub-$0.05, Polygon $0.05–$0.50 post-Dencun)

All external latency/cost/benchmark claims are current as of publication (August 2026) — re-verify before committing to SLAs.
