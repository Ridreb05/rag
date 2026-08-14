# RunPod deployment

This image is deliberately a **single-Pod** design: Qdrant runs on private
`127.0.0.1:6333` and FastAPI serves the built frontend on port `8000`.
RunPod does not support Docker Compose on Pods.

## Publish the image

Push this repository to GitHub's `master` branch, then open the **Actions** tab
and wait for `Publish RunPod image` to succeed. The image is:

```text
ghcr.io/ridreb05/voice-rag:runpod
```

If the GitHub package is private, make it public in the package settings or
configure registry credentials in the RunPod template.

## Create the Pod

1. Select **Community Cloud** and an **RTX 4090 24 GB**.
2. Set the container image to the published GHCR image.
3. Attach an 80–100 GB persistent volume at `/workspace`.
4. Expose HTTP port `8000` only. Do not expose `6333`.
5. Add `SARVAM_API_KEY`, `GEMINI_API_KEY`, `VOICE_RAG_GENERATION_BACKEND=gemini`, `VOICE_RAG_BOOTSTRAP_INDEX=1`, and `VOICE_RAG_INDEX_VERSION=full1` as environment variables.
6. For the first deployment, leave `VOICE_RAG_INDEX_SPLIT=validation`. It is the practical demo corpus. Use a separately documented train/validation evaluation run for final retrieval-quality claims.

The first startup downloads and indexes the corpus. It writes an atomic,
versioned checkpoint after every committed embedding batch. If you must stop a
Pod, keep the same network volume and use the same `VOICE_RAG_INDEX_VERSION`:
the next Pod resumes from the checkpoint instead of redoing prior batches.
Keep `VOICE_RAG_BOOTSTRAP_INDEX=1` on restarts; a completed index is detected
immediately and skips bootstrap work. Do not reuse an incomplete index under a
different version.

For a low-cost live-demo index, set both `VOICE_RAG_INDEX_VERSION=demo100k`
and `VOICE_RAG_BOOTSTRAP_LIMIT=100000`. That is intentionally a 100k-chunk
subset, not a full-corpus benchmark. Omit `VOICE_RAG_BOOTSTRAP_LIMIT` for the
full Hindi validation index.

Qdrant storage is isolated by `VOICE_RAG_INDEX_VERSION`. This means an old,
incomplete `full1` collection is never loaded while `demo100k` is booting and
cannot consume CPU in the background. Leave `VOICE_RAG_QDRANT_STORAGE_PATH`
unset unless you deliberately manage that storage yourself.

After uploads, the logs print Qdrant finalization status every 15 seconds and
stop after 15 minutes if search indexes are still not ready. That failure is
intentional: the state remains `optimizing`, so the next start only resumes
finalization and never re-embeds the corpus. It prevents an invisible,
open-ended bill.

## Verify and benchmark

RunPod exposes the app as:

```text
https://POD_ID-8000.proxy.runpod.net
```

Open `/v1/health` first. It returns `200` only when Qdrant, BM25, the
versioned completion manifest, and the expected exact Qdrant point count all
agree. A `503` means bootstrap is incomplete or an index artifact is missing.

From your computer, run:

```powershell
uv run python -m benchmark.voice_e2e_benchmark --audio-dir data/benchmark_audio --api-url https://POD_ID-8000.proxy.runpod.net
```

Stop the Pod when you are not testing. Keep the persistent volume only as
long as you need the index for the demo.
