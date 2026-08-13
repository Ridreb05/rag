# Deploying to RunPod

Concrete steps to get the API gateway running on a rented GPU pod with a
public URL, matching the [deployment decision](09-roadmap-and-summary.md):
persistent GPU for the retrieval hot path (embedding + reranker), Gemini
(or Claude, once credits are added) for generation over the API.

## Why RunPod, briefly

The `<200ms` retrieval target requires the GPU co-located with the app —
measured on the dev machine, single-query BGE-M3 embed calls run
20–35/sec and the full retrieval chain lands around 35–70ms end-to-end
when everything's warm. CPU-only or API-based embedding/reranking would
blow that budget (see [Latency & Caching](04-latency-and-caching.md)).
Serverless GPU has unpredictable cold starts; a persistent rented pod
matches what was actually measured locally.

## 1. Create the pod

1. Sign up at [runpod.io](https://runpod.io), add billing.
2. **Deploy a Pod** → choose a GPU. A T4 or L4-class GPU is enough — BGE-M3
   + the reranker together are ~2-3GB in fp16, comfortably under an 8GB
   card, same as this project's dev GPU (RTX 4060 Laptop, 8GB).
3. Choose a template with Docker support (any "RunPod PyTorch" or base
   Ubuntu + Docker template works — this project builds its own image, it
   doesn't need a pre-built PyTorch template).
4. **Important:** check the pod's NVIDIA driver version before deploying —
   this project's `torch` build is pinned to CUDA 13.0
   (`torch==2.13.0+cu130` in `pyproject.toml`). If the pod's driver is
   older and only supports CUDA 12.x, either pick a newer GPU template or
   re-pin `pyproject.toml`'s `pytorch-cu130` index to `pytorch-cu124` (see
   the comment there for how the pin works) before building the image.
5. Expose port `8000` (the app) when configuring the pod — RunPod gives you
   a proxy URL like `https://<pod-id>-8000.proxy.runpod.net`. That's your
   public link once the app is running.

## 2. Get the code and secrets onto the pod

```bash
# SSH into the pod (RunPod gives you the command on the pod's page)
git clone <your-repo-url> voice-rag
cd voice-rag

# .env is gitignored — create it directly on the pod, never commit it
cat > .env <<'EOF'
SARVAM_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
EOF
```

## 3. Bring up the stack

```bash
docker compose up -d --build
```

This starts two containers: `qdrant` (a real server, not the embedded
local mode used in dev — see the note in `docker-compose.yml`) and `app`
(the FastAPI gateway, GPU-attached).

## 4. Populate the index

The BM25 (Tantivy) index is a plain directory — build it once anywhere and
copy it in (`docker-compose.yml` mounts `data/full_index/bm25`). The dense
index has to be built **against the running Qdrant server**, not copied
from a local-mode build (verified: local mode's SQLite storage format
isn't compatible with the server — see
[Evaluation Results](evaluation-results.md)):

```bash
# From inside the pod, with the stack already running:
uv run python scripts/build_full_index.py \
  --language hi --split validation \
  --qdrant-url http://localhost:6333 \
  --index-version full1
```

This is the same script already tested during development — measured at
~140 chunks/sec on an RTX 4060 laptop GPU (~965K chunks ≈ 100–150 min for
the full Hindi validation corpus); a datacenter T4/L4 should be
comparable or faster. It logs progress and an ETA as it runs — see
`data/full_index_build.log`.

**Do not run anything else against the same Qdrant path/server while this
is in progress.** Verified directly during development: a second process
touching the same local-mode index mid-write killed the indexing job with
`sqlite3.OperationalError: database is locked`. The real Qdrant server
handles concurrent access correctly (that's exactly the problem it solves
over local mode) — this warning applies only if you're still using local
mode for some reason instead of the server in `docker-compose.yml`.

## 5. Verify

```bash
curl http://localhost:8000/v1/health
```

Then from outside the pod, the RunPod proxy URL
(`https://<pod-id>-8000.proxy.runpod.net/v1/health`) should return the
same thing — that's your public link.

**Testing with non-English queries:** if testing from a Windows machine's
git-bash `curl`, be aware — verified directly during development — that
passing Devanagari/other non-ASCII text through bash command-line
arguments to `curl -d` can silently mangle the UTF-8 encoding before it
reaches the server, producing a garbage query and a misleadingly low
retrieval confidence. Use a proper HTTP client (Python `httpx`/`requests`,
Postman, `curl --data-binary @file.json` reading from a UTF-8-saved file)
instead of inline shell arguments for non-ASCII test payloads.

## 6. What's already protected

- **API keys never leave the server** — `.env` is read only by the FastAPI
  process; the client only ever talks to your `/v1/query` endpoint (see
  [Web3 & Privacy](05-web3-and-privacy.md#security)).
- **Rate limiting** is on by default (20 requests/min/IP) — protects the
  Gemini/Claude budget from a public link getting hammered. See
  `voice_rag/apps/api_gateway/rate_limit.py` for the known limitation
  (per-process, not shared across multiple app replicas — not an issue at
  the single-pod scale this guide describes).

## Known gaps in this deployment path

- **Sarvam voice streaming (Phase 7/10) isn't wired into the API gateway
  yet** — `POST /v1/query` is text-in/text-out. The Sarvam STT/TTS clients
  are built and tested (`voice_rag/stt/`) but not yet connected to a
  `WS /v1/voice/stream` endpoint.
- **Single language, single index version** at a time, set via
  `VOICE_RAG_LANGUAGE`/`VOICE_RAG_INDEX_VERSION` env vars — multi-language
  routing (per docs/02) isn't implemented in the gateway yet.
- **Web3 provenance anchoring (Phase 13)** is intentionally not part of
  this deployment — deferred per the roadmap's own sequencing (it's an
  audit feature, not a dependency for the app to work) and by explicit
  user decision this session.
