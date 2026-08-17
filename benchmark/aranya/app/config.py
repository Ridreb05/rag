"""Configuration for the live-pod run of benchmark.py."""

import os

BASE_URL = os.environ.get(
    "RAG_BASE_URL", "https://kqipnoh0es5c6s-8000.proxy.runpod.net"
).rstrip("/")

# The deployed service reports its own budget as latency_ms.budget_ms (200ms).
# benchmark.py's docstring assumes 50ms, which belongs to a different service;
# comparing the pod against the pod's own budget is the meaningful check.
LATENCY_BUDGET_MS = float(os.environ.get("LATENCY_BUDGET_MS", "200"))

# /v1/query is rate limited to 20 requests per 60s per IP, so the client
# paces itself rather than collecting 429s.
RATE_LIMIT_REQUESTS = int(os.environ.get("RAG_RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_S = float(os.environ.get("RAG_RATE_LIMIT_WINDOW_S", "60"))

REQUEST_TIMEOUT_S = float(os.environ.get("RAG_REQUEST_TIMEOUT_S", "120"))
QUERY_LANGUAGE = os.environ.get("RAG_QUERY_LANGUAGE", "hi")
