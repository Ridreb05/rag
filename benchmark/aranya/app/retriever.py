"""HTTP-backed stand-in for the in-process retriever benchmark.py expects.

search() returns the same three fields the benchmark reads — embed_ms,
search_ms, total_ms — taken from the server's own stage timings so the
comparison against the budget is server-side, apples to apples. Network
time is recorded separately and printed at exit, because a pod that meets
its pipeline budget can still be slow to a caller in another country.
"""

from __future__ import annotations

import atexit
import collections
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.config import (
    BASE_URL,
    QUERY_LANGUAGE,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_S,
    REQUEST_TIMEOUT_S,
)


@dataclass
class SearchResponse:
    embed_ms: float
    search_ms: float
    total_ms: float
    wall_ms: float
    mode: str
    n_evidence: int


_sent: collections.deque[float] = collections.deque()
_wall: list[float] = []
_server_total: list[float] = []
_modes: collections.Counter[str] = collections.Counter()


def _throttle() -> None:
    """Stay inside the pod's per-IP sliding window instead of eating 429s."""
    while True:
        now = time.monotonic()
        while _sent and now - _sent[0] > RATE_LIMIT_WINDOW_S:
            _sent.popleft()
        if len(_sent) < RATE_LIMIT_REQUESTS:
            _sent.append(now)
            return
        wait = RATE_LIMIT_WINDOW_S - (now - _sent[0]) + 0.25
        time.sleep(max(wait, 0.1))


def _post(path: str, payload: dict) -> tuple[dict, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        # The RunPod proxy 403s urllib's default User-Agent.
        headers={"content-type": "application/json", "user-agent": "curl/8.0"},
        method="POST",
    )
    for attempt in range(5):
        _throttle()
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                data = json.loads(resp.read())
            return data, (time.perf_counter() - start) * 1000
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 4:
                time.sleep(RATE_LIMIT_WINDOW_S / RATE_LIMIT_REQUESTS + 1)
                continue
            raise
    raise RuntimeError("rate limited repeatedly")


def search(query: str, top_k: int = 5) -> SearchResponse:
    data, wall_ms = _post(
        "/v1/query",
        {"query": query, "language": QUERY_LANGUAGE, "top_k": top_k},
    )
    lat = data["latency_ms"]
    resp = SearchResponse(
        embed_ms=lat.get("embedding_ms", 0.0),
        # The pod's search stage is dense retrieval plus BM25 fusion.
        search_ms=lat.get("retrieval_ms", 0.0) + lat.get("fusion_ms", 0.0),
        total_ms=lat.get("total_ms", 0.0),
        wall_ms=wall_ms,
        mode=data.get("mode", "?"),
        n_evidence=len(data.get("evidence", [])),
    )
    _wall.append(wall_ms)
    _server_total.append(resp.total_ms)
    _modes[resp.mode] += 1
    return resp


def warmup() -> None:
    """One untimed request so model load and connection setup are excluded."""
    search("warmup", top_k=1)
    _wall.pop()
    _server_total.pop()
    _modes.subtract(["refused"])


def _report_wall() -> None:
    if not _wall:
        return
    ordered = sorted(_wall)
    p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
    overhead = statistics.mean(_wall) - statistics.mean(_server_total)
    print(f"\nClient-observed wall clock against {BASE_URL}")
    print(
        f"  avg {statistics.mean(_wall):.1f}ms | p95 {p95:.1f}ms | "
        f"network + proxy overhead ~{overhead:.1f}ms"
    )
    modes = ", ".join(f"{m}={c}" for m, c in _modes.most_common() if c > 0)
    print(f"  answer modes: {modes}")


atexit.register(_report_wall)
