"""Measure the actual browser-equivalent voice request path against a running API.

This intentionally calls ``/v1/voice-query`` over HTTP rather than importing
pipeline internals: its wall-clock number includes upload, Sarvam STT,
retrieval, reranking, guardrails, and answer generation. Supply a directory
of short, real Hindi recordings (webm, wav, m4a, or mp4) and archive the JSON
output with the submission.

Example:
    uv run python -m benchmark.voice_e2e_benchmark --audio-dir data/benchmark_audio
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import statistics
import time
from pathlib import Path

import httpx

SUPPORTED_SUFFIXES = {".webm", ".wav", ".mp3", ".m4a", ".mp4", ".ogg"}


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("no successful voice requests to summarize")
    ordered = sorted(values)

    def pick(percent: float) -> float:
        index = round((len(ordered) - 1) * percent)
        return ordered[index]

    return {"n": len(ordered), "p50_ms": pick(0.50), "p70_ms": pick(0.70), "p95_ms": pick(0.95), "p100_ms": ordered[-1], "mean_ms": statistics.fmean(ordered)}


def run(audio_dir: Path, api_url: str, timeout_seconds: float) -> dict:
    files = [path for path in sorted(audio_dir.iterdir()) if path.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        raise ValueError(f"no supported audio files found in {audio_dir}")

    rows: list[dict] = []
    with httpx.Client(timeout=timeout_seconds) as client:
        for path in files:
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            started_at = time.perf_counter()
            response = client.post(
                f"{api_url.rstrip('/')}/v1/voice-query",
                data={"language": "hi", "top_k": "10"},
                files={"audio": (path.name, path.read_bytes(), mime_type)},
            )
            wall_ms = (time.perf_counter() - started_at) * 1000
            row = {"file": path.name, "wall_ms": wall_ms, "status_code": response.status_code}
            if response.is_success:
                body = response.json()
                row.update({"mode": body["mode"], "trace_id": body["trace_id"], "pipeline_latency_ms": body["latency_ms"]})
            else:
                row["error"] = response.text
            rows.append(row)

    successful = [row["wall_ms"] for row in rows if row["status_code"] == 200]
    return {"api_url": api_url, "audio_files": len(files), "successful_requests": len(successful), "wall_clock": percentiles(successful), "requests": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=Path("reports/voice_e2e_benchmark.json"))
    args = parser.parse_args()

    result = run(args.audio_dir, args.api_url, args.timeout_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["wall_clock"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
