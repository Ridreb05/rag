# Deployment runbook

## Before deploying

1. Copy `.env.example` to `.env`, then provide newly generated `SARVAM_API_KEY` and `GEMINI_API_KEY`; never commit `.env` or paste it into terminal output.
2. Build the Hindi corpus and chunks, then index them into the same Qdrant instance the API will use. Use a versioned value such as `VOICE_RAG_INDEX_VERSION=full1` and record it with the benchmark.
3. Verify `GET /v1/health` returns `200` before exposing the public URL. A `503` means Qdrant, the expected collection, or BM25 is not ready.

## Local container validation

```powershell
docker compose up -d qdrant
uv run python scripts/build_full_index.py --language hi --split train --qdrant-url http://localhost:6333 --index-version full1
docker compose up --build app
```

Open `http://localhost:8000`, then verify:

```powershell
Invoke-WebRequest http://localhost:8000/v1/health
```

The app container needs its BM25 directory mounted and Qdrant's persistent volume must be retained. Do not deploy against an empty Qdrant collection. Compose deliberately binds Qdrant only to `127.0.0.1`; expose only the app behind an HTTPS reverse proxy or managed ingress.

## Submission evidence

Run both benchmark paths on the final deployment hardware and retain their JSON outputs:

```powershell
uv run python -m benchmark.latency_benchmark --qdrant-url http://localhost:6333
uv run python -m benchmark.voice_e2e_benchmark --audio-dir data/benchmark_audio --api-url https://YOUR-LIVE-URL
```

Report the real P50/P70/P100 wall-clock values from the voice benchmark separately from retrieval-only latency. The voice benchmark sends the same multipart request used by the browser and includes STT and generation.
