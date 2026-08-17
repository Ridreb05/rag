"""Latency benchmark against a deployed API over HTTPS.

The companion to ``benchmark/latency_benchmark.py``: that one imports the
pipeline and measures it in-process, which isolates per-stage cost but runs
on whatever GPU the developer has. This one sends real HTTPS requests to a
running deployment and reports the server's own ``pipeline_ms``, so the
number is the one a caller actually gets from the shipped configuration.

Queries come from the same validation set the in-process benchmark samples,
with the same seed, so the two are comparable rather than merely similar.

The deployment rate limits ``/v1/query`` to 20 requests per 60s per IP, so
the client paces itself inside that window; a 150-query run therefore takes
about eight minutes of mostly waiting. ``/v1/query/refine`` is not rate
limited.

Usage:
    uv run python -m benchmark.deployed_benchmark \
        --api-url https://<pod>.proxy.runpod.net --n-queries 150
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("reports/latency_benchmark")
PERCENTILES = [50, 70, 95, 99, 100]

# Mirrors api/rate_limit.py's middleware configuration.
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_S = 60.0

# The RunPod proxy rejects some default client user agents with a 403.
HEADERS = {"user-agent": "voice-rag-benchmark/1.0"}


def percentile_report(values_ms: list[float]) -> dict[str, float]:
    """Percentiles without numpy's interpolation, so p100 is a real observation."""
    ordered = sorted(values_ms)
    report = {
        f"p{p}": ordered[min(round((len(ordered) - 1) * p / 100), len(ordered) - 1)]
        for p in PERCENTILES
    }
    return report | {"mean": statistics.fmean(ordered), "min": ordered[0], "n": len(ordered)}


class RateLimitedClient:
    """Keeps the caller inside the deployment's sliding window.

    Pacing beats retrying: a 429 costs a round trip and tells us nothing,
    and a benchmark that collects them measures the rate limiter rather
    than the pipeline.
    """

    def __init__(self, client: httpx.Client, api_url: str):
        self._client = client
        self._url = api_url.rstrip("/")
        self._sent: collections.deque[float] = collections.deque()

    def _throttle(self) -> None:
        while True:
            now = time.monotonic()
            while self._sent and now - self._sent[0] > RATE_LIMIT_WINDOW_S:
                self._sent.popleft()
            if len(self._sent) < RATE_LIMIT_REQUESTS:
                self._sent.append(now)
                return
            time.sleep(max(RATE_LIMIT_WINDOW_S - (now - self._sent[0]) + 0.25, 0.1))

    def post(self, path: str, payload: dict, *, rate_limited: bool = True) -> tuple[dict, float]:
        if rate_limited:
            self._throttle()
        started = time.perf_counter()
        response = self._client.post(f"{self._url}{path}", json=payload, headers=HEADERS)
        wall_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        return response.json(), wall_ms


def run(
    api_url: str,
    language: str,
    index_version: str,
    n_queries: int,
    n_refine: int,
    top_k: int,
    seed: int,
    timeout_seconds: float,
) -> dict:
    queries_df = pd.read_parquet(PROCESSED_DIR / language / "validation_queries.parquet")
    sample = queries_df.sample(n=min(n_queries, len(queries_df)), random_state=seed).reset_index(drop=True)

    stage_keys = [
        "embedding_ms",
        "retrieval_ms",
        "bm25_wall_ms",
        "fusion_ms",
        "rerank_ms",
        "generation_ms",
    ]
    stages: dict[str, list[float]] = {key: [] for key in stage_keys}
    pipeline_ms: list[float] = []
    wall_ms: list[float] = []
    modes: collections.Counter[str] = collections.Counter()
    guardrails: collections.Counter[str] = collections.Counter()
    refinable: list[str] = []
    budget_ms = 0.0
    # Per-request rows, so "which mode were the over-budget requests in?" is a
    # lookup rather than an inference from the stage distributions.
    rows: list[dict] = []

    with httpx.Client(timeout=timeout_seconds) as raw:
        client = RateLimitedClient(raw, api_url)

        print(f"Warming up against {api_url} ...", flush=True)
        client.post("/v1/query", {"query": "वार्मअप", "language": language, "top_k": top_k})

        for i, row in enumerate(sample.itertuples()):
            query_text = row.query_text
            if not isinstance(query_text, str) or not query_text.strip():
                continue

            body, elapsed_ms = client.post(
                "/v1/query", {"query": query_text, "language": language, "top_k": top_k}
            )
            latency = body["latency_ms"]
            pipeline_ms.append(latency["pipeline_ms"])
            wall_ms.append(elapsed_ms)
            budget_ms = latency.get("budget_ms", budget_ms)
            for key in stage_keys:
                if key in latency:
                    stages[key].append(latency[key])
            modes[body["mode"]] += 1
            guardrails.update(body.get("guardrail_flags", []))
            if body.get("refinement_available"):
                refinable.append(body["trace_id"])
            rows.append(
                {
                    "mode": body["mode"],
                    "pipeline_ms": latency["pipeline_ms"],
                    "generation_ms": latency.get("generation_ms"),
                    "wall_ms": elapsed_ms,
                    "over_budget": latency["pipeline_ms"] > latency.get("budget_ms", budget_ms),
                    "guardrail_flags": body.get("guardrail_flags", []),
                }
            )

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(sample)}  p50 so far {statistics.median(pipeline_ms):.1f}ms", flush=True)

    result = {
        "measurement": "deployed API over HTTPS, server-reported pipeline_ms",
        "api_url": api_url,
        "language": language,
        "index_version": index_version,
        "budget_ms": budget_ms,
        "n": len(pipeline_ms),
        "pipeline_ms": percentile_report(pipeline_ms),
        "wall_clock_ms": percentile_report(wall_ms),
        "stage_ms": {key: percentile_report(vals) for key, vals in stages.items() if vals},
        "within_budget": sum(1 for v in pipeline_ms if v <= budget_ms),
        "mode_breakdown": dict(modes),
        "guardrail_flags": dict(guardrails),
        "over_budget_modes": dict(collections.Counter(r["mode"] for r in rows if r["over_budget"])),
        "requests": rows,
    }

    # Phase two is a separate budget on purpose (see /v1/query/refine's docstring),
    # so it is reported separately rather than folded into the headline number.
    if n_refine and refinable:
        refine_ms: list[float] = []
        refine_modes: collections.Counter[str] = collections.Counter()
        with httpx.Client(timeout=timeout_seconds) as raw:
            client = RateLimitedClient(raw, api_url)
            for trace_id in refinable[:n_refine]:
                body, elapsed_ms = client.post(
                    "/v1/query/refine", {"trace_id": trace_id}, rate_limited=False
                )
                refine_ms.append(elapsed_ms)
                refine_modes[body["mode"]] += 1
        result["phase_two_refinement_ms"] = percentile_report(refine_ms)
        result["phase_two_modes"] = dict(refine_modes)
    result["refinement_available_count"] = len(refinable)

    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--index-version", default="full1")
    parser.add_argument("--n-queries", type=int, default=150)
    parser.add_argument("--n-refine", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run(
        api_url=args.api_url,
        language=args.language,
        index_version=args.index_version,
        n_queries=args.n_queries,
        n_refine=args.n_refine,
        top_k=args.top_k,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
    )

    out_path = args.output or RESULTS_DIR / f"{args.language}_{args.index_version}_deployed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    pipeline = result["pipeline_ms"]
    print()
    print(f"n={pipeline['n']}  budget={result['budget_ms']}ms")
    print(f"pipeline_ms  P50 {pipeline['p50']:.1f}  P70 {pipeline['p70']:.1f}  P100 {pipeline['p100']:.1f}")
    print(f"within budget: {result['within_budget']}/{pipeline['n']}")
    print(f"modes: {result['mode_breakdown']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
