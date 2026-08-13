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
5. Add `SARVAM_API_KEY`, `GEMINI_API_KEY`, `VOICE_RAG_GENERATION_BACKEND=gemini`, and `VOICE_RAG_BOOTSTRAP_INDEX=1` as environment variables.
6. For the first deployment, leave `VOICE_RAG_INDEX_SPLIT=validation`. It is the practical demo corpus. Use a separately documented train/validation evaluation run for final retrieval-quality claims.

The first startup downloads and indexes the corpus. Do not stop the Pod while
that bootstrap is running. When it finishes, it starts the API automatically.

## Verify and benchmark

RunPod exposes the app as:

```text
https://POD_ID-8000.proxy.runpod.net
```

Open `/v1/health` first. After a successful bootstrap, change
`VOICE_RAG_BOOTSTRAP_INDEX` to `0` before future restarts; the persistent
volume retains Qdrant and BM25 data.

From your computer, run:

```powershell
uv run python -m benchmark.voice_e2e_benchmark --audio-dir data/benchmark_audio --api-url https://POD_ID-8000.proxy.runpod.net
```

Stop the Pod when you are not testing. Keep the persistent volume only as
long as you need the index for the demo.
