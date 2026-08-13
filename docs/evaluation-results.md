# Real Evaluation Results (2026-08-13)

Unlike the rest of `docs/`, this file reports **measured** numbers from
running the actual pipeline against real MSMARCO-XI data on this machine
(RTX 4060 Laptop GPU, 8GB VRAM), not projected/estimated figures. Reproduce
with:

```
uv run python -m evaluation.run_subset_eval --language hi --n-queries 200 --rerank
```

## What this evaluates

A **closed-world subset**: 200 randomly sampled Hindi validation queries
(seed 42), and only the 1,990 unique passages those queries actually
reference (their own 10-passage candidate pools, deduplicated). This is
deliberately not a full-corpus evaluation — embedding the complete
~965K-chunk Hindi corpus measures at 72 chunks/sec on this GPU, i.e. ~3.7
hours, which doesn't fit an interactive session. The same code indexes the
full corpus; only the data volume differs. A full-corpus run is queued as
follow-up work (see [Roadmap](09-roadmap-and-summary.md) Phase 11) and
should be run as a background/batch job, not interactively.

Because it's closed-world (small candidate pool, not the full ~950K-passage
corpus), these numbers are **optimistic relative to open-corpus retrieval**
— recall is inflated by the small pool size. They are still a real,
apples-to-apples measurement of *relative* pipeline quality (fusion vs.
reranking), which is what this run was designed to isolate.

Of the 200 sampled queries, 91 have zero `is_selected=1` passages in their
pool (consistent with the ~45% zero-relevant-passage rate found in Phase 1's
dataset analysis) and are excluded from scoring, per
[docs/01-dataset-analysis.md](01-dataset-analysis.md)'s finding that these
must not be scored as retrieval failures.

## Results (109 scoreable queries)

| Metric | Fused (dense + BGE-sparse + BM25, RRF) | + bge-reranker-v2-m3 (top-20) | Δ |
|---|---:|---:|---:|
| Recall@10 | 0.8853 | 0.9174 | +0.032 |
| MRR | 0.4827 | 0.6227 | **+0.140** |
| NDCG@10 | 0.5753 | 0.6949 | +0.120 |
| HitRate@5 | 0.7523 | 0.8532 | +0.101 |

**Reranking's biggest measured effect is on rank quality, not raw recall** —
Recall@10 barely moves (the relevant passage is already in the top 10 most
of the time) but MRR jumps 14 points, meaning fusion alone frequently buries
the right passage at rank 6-10 while the cross-encoder correctly promotes
it toward rank 1. This is a real, data-grounded confirmation of the
architecture's reranking design (docs/02-architecture-and-retrieval.md,
docs/06-reranking) — accuracy gain is concentrated exactly where a cheap
RRF fusion is weakest: fine-grained relevance ordering among plausible
candidates, not coarse recall.

## Six real bugs found and fixed this session

Each is documented in code comments at its fix site; summarized here
because they're the kind of thing "don't assume the schema/library
contract, and verify rather than trust your own benchmark" is meant to
catch. None of these were staged for effect — each broke a real run first.

1. **Tantivy query-string injection on ordinary text.** `Index.parse_query`
   treats input as a Lucene-style query language, where `-` means NOT. A
   real Hindi passage containing a hyphen ("...वाहन - यान ले जाना...")
   crashed with a syntax error. Fixed in
   `voice_rag/retrieval/sparse/bm25_index.py` by building an OR-of-terms
   boolean query directly instead of parsing raw text as a query string —
   the general lesson (never parse untrusted/arbitrary text as a query
   language) is also a security control per
   [docs/05-web3-and-privacy.md](05-web3-and-privacy.md#security).
2. **FlagEmbedding's reranker is incompatible with transformers 5.x.**
   `FlagReranker.compute_score` calls `tokenizer.prepare_for_model`, which
   `transformers==5.15.0` no longer exposes. FlagEmbedding's package
   metadata claims `transformers<6.0.0` support, but its actual code
   predates the 5.x tokenizer API change. Downgrading transformers to a
   compatible 4.x line isn't viable either — this project needs
   `huggingface-hub>=1.27.0`, and every transformers 4.x release requires
   `huggingface-hub<1.0` (confirmed by an explicit failed resolution
   attempt, not assumed). Fixed by loading bge-reranker-v2-m3 directly via
   `transformers.AutoModelForSequenceClassification` instead of through
   FlagEmbedding's wrapper — see `voice_rag/reranking/service.py`.

3. **Qdrant's embedded local mode is single-writer, backed by SQLite.**
   Starting the FastAPI dev server pointed at the same
   `data/full_index/qdrant` path the background full-corpus indexing job
   was actively writing to killed the indexing job mid-run with
   `sqlite3.OperationalError: database is locked`. This is a live
   demonstration of exactly the limitation the architecture blueprint
   already called out (docs/08-repo-and-stack.md: local mode is for
   dev/correctness testing, not the production topology) — it just took
   causing the actual failure to confirm it concretely rather than taking
   the docs' word for it. Fix: never point two processes at the same
   local-mode Qdrant path concurrently. The indexing job was restarted
   (upserts are idempotent via chunk_id-derived point IDs, so
   re-processing already-written rows is safe, just slower); the API
   server is tested against a small isolated index built specifically for
   that purpose instead.

4. **Testing non-ASCII payloads via Windows git-bash `curl` silently
   mangles UTF-8**, producing a garbage query and a misleadingly low
   retrieval confidence — a real query that scores 0.995 on an exact-match
   passage came back as a false "refused" through `curl -d`, and came back
   correct through Python `httpx`. This is a testing-tool artifact, not a
   pipeline bug, but it's the kind of thing that would have been reported
   as a false guardrail bug without cross-checking with a second HTTP
   client. Documented in [docs/runpod-deployment.md](runpod-deployment.md).

5. **Security: the Gemini key was being logged in plaintext.** `httpx`'s
   own logger writes the full request URL (including query parameters) at
   INFO level, and the Gemini REST client was passing the API key as a
   `?key=...` query parameter — with the benchmark/CLI scripts in this
   repo setting root logging to INFO, every Gemini call wrote the key
   straight into the log output. Verified directly: it showed up in a
   background task's log file mid-benchmark-run. Fixed two ways in
   `voice_rag/generation/gemini_service.py`: the key now goes in the
   `x-goog-api-key` header instead of the URL (confirmed working against
   the real API), and `httpx`'s logger is forced to WARNING regardless of
   the caller's root logging level, as defense in depth. The Sarvam and
   Claude clients were never at risk from this specific issue — both
   already use header-based auth (`api-subscription-key` and the SDK's
   `Authorization` header respectively), and httpx's default INFO logging
   only includes the request line (method + URL + status), not headers.

6. **Qdrant's embedded local mode gives misleadingly slow latency
   numbers, specifically for sparse vector search.** The first latency
   benchmark run (40,000-chunk index, local mode) measured
   retrieval-pipeline P50 ≈ 920ms — nowhere near the target. Isolating the
   cause: re-measured with zero competing background processes and local
   mode was still ~600ms for sparse search alone, ruling out contention as
   the explanation. Switched to a real Qdrant server (same index contents,
   same queries, via Docker) and dense retrieval dropped 145ms→13ms P50
   while sparse retrieval dropped 602ms→8ms P50 — an order of magnitude,
   confirming local mode's sparse search lacks the real server's optimized
   inverted-index structures at any meaningful scale. Combined with tuning
   retrieval/rerank depth from top-20 to top-10 per signal (roughly
   halving rerank latency), the real-server retrieval pipeline measured
   P99 = 170.20ms across 1,000 real queries — under the 200ms target with
   real margin. Full numbers:
   [docs/latency-benchmark-results.md](latency-benchmark-results.md);
   methodology and the original (wrong) local-mode numbers:
   [SUBMISSION.md](../SUBMISSION.md). The general lesson, consistent with
   the repo-wide local-mode warnings already in `docs/08-repo-and-stack.md`:
   local mode is a correctness-testing convenience, never a latency
   benchmarking substitute for the real server.

## Cross-lingual sanity check

Before running the full evaluation, a direct embedding similarity check
confirmed BGE-M3's cross-lingual alignment on real translated text:
`cosine("what is diabetes" [en], "मधुमेह क्या है" [hi]) = 0.844` vs.
`cosine("what is diabetes" [en], "capital of france" [en]) = 0.368` — clear
semantic separation, validating the core cross-lingual dense retrieval
hypothesis this architecture depends on.

## Infrastructure actually used

- Embeddings: BGE-M3 (`BAAI/bge-m3`) via FlagEmbedding, fp16, CUDA
- Reranker: bge-reranker-v2-m3 (`BAAI/bge-reranker-v2-m3`) via transformers
  `AutoModelForSequenceClassification`, fp16, CUDA
- Dense + BGE-sparse index: Qdrant embedded local mode (no server — Docker
  Desktop was not running on this machine; see
  [Roadmap](09-roadmap-and-summary.md) for the production server/cluster
  requirement)
- BM25: Tantivy, embedded, no server
- GPU: NVIDIA RTX 4060 Laptop GPU, 8GB VRAM, CUDA 13.0 wheel (torch 2.13.0+cu130)
