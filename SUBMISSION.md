# Submission checklist

- [ ] Public GitHub repository URL added to the form.
- [ ] Public live demo URL added to the form.
- [ ] `/v1/health` returns `200` on the public deployment.
- [ ] Text query, microphone permission, transcription, answer, sources, and refusal flows demonstrated in a clean browser session.
- [ ] `uv run pytest` passes.
- [ ] `npm ci && npm run build` passes in `frontend/`.
- [ ] Retrieval evaluation is run against a held-out validation set, not an index built solely from that evaluation sample.
- [ ] `reports/voice_e2e_benchmark.json` contains real P50, P70, and P100 measurements from the final deployed system.
- [ ] README states that retrieval and full voice-to-answer latency are separate measurements.
- [ ] No `.env`, API key, generated index, or audio recording containing private data is committed.
