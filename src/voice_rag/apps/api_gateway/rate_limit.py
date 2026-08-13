"""Per-IP rate limiting (docs/05-web3-and-privacy.md#security).

Deliberately simple: an in-memory sliding window, no Redis. This is a
public-link demo protection against someone draining the Claude API budget,
not a production multi-tenant rate limiter — the known limitation is that
this doesn't share state across multiple worker processes (each process
gets its own budget). For a single-process deployment (the RunPod setup in
docs/runpod-deployment.md), that's the actual topology, so it's not a gap
in practice; scaling to multiple workers later means moving this to Redis
(`INCR` + `EXPIRE`), not rewriting the interface.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class InMemoryRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 20, window_seconds: float = 60.0, paths: tuple[str, ...] = ("/v1/query",)):
        super().__init__(app)
        self._limiter = InMemoryRateLimiter(max_requests, window_seconds)
        self._paths = paths

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._paths:
            client_ip = request.client.host if request.client else "unknown"
            if not self._limiter.allow(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"error": {"code": 429, "message": "Rate limit exceeded, try again shortly."}},
                )
        return await call_next(request)
