# Database Design & API

## Database design

```sql
-- Application DB (Postgres)
users(id, email, created_at, ...)
sessions(id, user_id, started_at, expires_at)
requests(id, trace_id, session_id, stage_timings jsonb,
         status, pipeline_version, model_version, created_at)
pipeline_versions(version, embed_model_hash, chunking_config_hash,
                   reranker_version, prompt_version, index_version, created_at)

-- Vector DB (Qdrant) — collection "chunks_v{index_version}"
point {
  id: chunk_id,
  vectors: { dense: float[1024], sparse: sparse_vec },
  payload: {
    doc_id, query_id, language, text, source: "eng"|"translated",
    is_selected_eval: bool|null, level: "passage"|"section"|"doc",
    parent_id: str|null, index_version
  }
}

-- Cache (Redis)
embed:{sha256}         -> float[1024]           TTL 24h
retrieval:{sha256}     -> [chunk_id,...]         TTL until index_version bump
answer:{sha256}        -> AnswerResponse json    TTL until any version bump

-- Provenance store (Postgres + content-addressed object storage)
document_hashes(doc_id, sha256, index_version)
chunk_hashes(chunk_id, sha256, index_version)
merkle_roots(index_version, root_hash, manifest_hash, tx_hash,
             chain_id, block_number, anchored_at)
answer_manifests(trace_id, query_hash, cited_chunk_ids jsonb,
                  index_version, model_version, confidence,
                  signature, created_at)   -- off-chain, batched if ever anchored
```

## API design

| Endpoint | Purpose |
|---|---|
| `POST /v1/query` | Text-in RAG query. Body: `{query, language?, top_k?}`. Returns `AnswerResponse`. |
| `GET /v1/query/{id}` | Fetch a completed query's result by `trace_id`. |
| `WS /v1/voice/stream` | Bidirectional: client streams audio frames; server emits `partial_transcript`, `final_transcript`, `evidence`, `answer_token`, `answer_complete`, `error` events in that order (evidence arrives before the answer completes, per the staged latency design). |
| `GET /v1/evidence/{id}` | Resolved citations for a given response: chunk text, doc_id, score, entailment score. |
| `GET /v1/pipeline/version` | Current active `pipeline_versions` row — embed/chunking/reranker/prompt/index versions in use. |
| `GET /v1/provenance/{document_id}` | Merkle proof + anchored root + Base tx hash for a given document, for independent verification. |
| `GET /v1/health` | Liveness + dependency status (Qdrant, Redis, STT provider, reranker pool). |
| `GET /v1/metrics` | Prometheus scrape endpoint. |

Auth: bearer API keys at the gateway, scoped per client; errors follow a
single `{error: {code, message, trace_id}}` shape with standard HTTP status
codes (400 malformed input, 401/403 auth, 404 unknown id, 429 rate-limited,
503 dependency circuit open).
