# Repository Structure, Technology Stack & Experiments

## Repository structure

```text
voice-rag/
├── apps/
│   └── api-gateway/          # FastAPI app, auth, rate limiting, WS handling
├── ingestion/                # MSMARCO-XI loaders, dataset-quality filters
├── chunking/                 # strategies A–F, granularity classifier
├── embeddings/               # BGE-M3 serving, ONNX export/quantization
├── retrieval/
│   ├── dense/                # Qdrant client, HNSW config
│   ├── sparse/                # Tantivy BM25 index
│   └── fusion/                # RRF, learned-fusion experiment
├── reranking/                 # bge-reranker-v2-m3 serving, batching
├── query_processing/          # lang-id, normalize, rewrite router
├── streaming/                 # partial-transcript debounce, reconciliation
├── generation/                # harness, schemas, extractive/generative router
├── guardrails/                # NLI grounding, moderation, off-topic gate
├── provenance/                # hashing, Merkle tree builder, manifest store
├── web3/                      # Base contract, anchoring job, proof verification
├── observability/             # OTel setup, Prometheus exporters, dashboards
├── evaluation/                # retrieval/RAG/speech metric harnesses
├── benchmark/                  # latency harness
├── infrastructure/             # IaC: Qdrant, Postgres, Redis, GPU worker pools
├── tests/
├── scripts/
└── docs/
```

## Recommended technology stack

| Layer | Choice | Runner-up considered | Why the choice wins here |
|---|---|---|---|
| Backend | Python / FastAPI | Node/Fastify | Native async, best model-serving ecosystem interop (ONNX, transformers), typed via Pydantic which the harness already relies on. |
| Vector DB | Qdrant | Milvus, Weaviate, pgvector | Best measured p99 among open-source options at this scale (~12ms @10M vectors on Qdrant's Rust engine vs. Weaviate ~16ms, Milvus ~18ms), native hybrid dense+sparse query support. Milvus is built for billion-scale, which this corpus doesn't need; pgvector is kept for the app DB but not asked to carry ANN search at this latency target. |
| Sparse search | Tantivy (embedded BM25) + BGE-M3 sparse | Elasticsearch/OpenSearch | No extra server to operate; a full ES/OpenSearch cluster is more infra than a 55GB corpus needs. |
| Embeddings | BGE-M3 | Gemini Embedding, LaBSE | Self-hostable (Gemini Embedding is API-only, adds network latency the budget can't absorb), covers all 14 languages, and is the only option producing dense + sparse + multi-vector from one model. |
| Reranker | bge-reranker-v2-m3 | Cohere Rerank 3.5 | Self-hostable and quantizable to fit the latency budget; Cohere's API adds a network hop the 200ms budget can't spare. |
| LLM (generative path) | Small quantized local model (Qwen2.5-7B/Llama-3.1-8B, INT4, vLLM) for lowest latency; Claude Haiku 4.5 as the higher-quality API fallback | Large local models, GPT-4-class API-only | The extractive/generative router means the generative path is only invoked when needed, so it can afford a slightly heavier model than if every query hit it — a local small model wins on latency and cost, Haiku 4.5 wins on instruction-following/structured-output quality when it's worth the extra hop. |
| STT | Sarvam Saaras v3 | ElevenLabs Scribe v2 | See [Architecture & Retrieval](02-architecture-and-retrieval.md#stt-selection). |
| Observability | OpenTelemetry + Prometheus + Grafana | Vendor APM (Datadog etc.) | Open standard, no per-seat licensing at small-team scale, and OTel's span model maps directly onto the harness's stage structure. |
| Web3 anchor chain | Base | Polygon, Linea | See [Web3 & Privacy](05-web3-and-privacy.md#web3-integration--where-it-actually-helps). |

Sources consulted for currency: [Sarvam](https://www.sarvam.ai/speech-to-text),
[ElevenLabs](https://elevenlabs.io/realtime-speech-to-text-api),
Qdrant/Weaviate/Milvus p99 comparisons (2026 vector-DB benchmark roundups),
BGE-M3 multilingual retrieval coverage, and 2026 L2 gas-fee comparisons
(Base/Linea sub-$0.05, Polygon $0.05–$0.50 post-Dencun).

## Experiment matrix

| Axis | Conditions | Primary metric |
|---|---|---|
| Chunking | Whole-passage (default) vs. 256 vs. 512 token forced splits | Recall@10, MRR@10 |
| Retrieval mode | Dense-only vs. BM25-only vs. Hybrid (RRF) | Recall@10, MRR@10, per-language breakdown |
| Reranking | None vs. bge-reranker-v2-m3 vs. full cross-encoder (unquantized) vs. LLM listwise (offline only) | NDCG@10, added latency |
| Embedding model | BGE-M3 vs. Gemini Embedding vs. LaBSE | Recall@10 per language, latency, cost/query |
| Retrieval depth | Top-5 / 10 / 20 / 50 | Recall@K vs. reranker latency trade-off curve |
| Context size | 1 / 3 / 5 / 8 passages in context | Faithfulness, answer relevance, token cost |
| Cache | Enabled vs. disabled | P50/P95 latency delta, hit rate |
| Quantization | FP32 vs. FP16 vs. INT8 (embedder + reranker) | Latency delta vs. Recall@10/NDCG@10 delta |
| Fusion strategy | RRF vs. weighted score fusion vs. learned logistic fusion | NDCG@10, per-language stability |
