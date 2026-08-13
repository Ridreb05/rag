import time

from voice_rag.apps.api_gateway.rate_limit import InMemoryRateLimiter
from voice_rag.apps.api_gateway.rate_limit import RateLimitMiddleware


def test_allows_up_to_max_requests():
    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False


def test_different_keys_have_independent_budgets():
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("a") is True
    assert limiter.allow("b") is True
    assert limiter.allow("a") is False
    assert limiter.allow("b") is False


def test_window_expiry_allows_new_requests():
    # Generous margin between window and sleep — a tight margin (e.g. 0.05s
    # window / 0.06s sleep) is flaky under real scheduler jitter, verified
    # directly: it failed intermittently in this exact suite.
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=0.1)
    assert limiter.allow("x") is True
    assert limiter.allow("x") is False
    time.sleep(0.3)
    assert limiter.allow("x") is True


def test_voice_endpoint_is_rate_limited_by_default():
    middleware = RateLimitMiddleware(app=lambda scope, receive, send: None)
    assert "/v1/query" in middleware._paths
    assert "/v1/voice-query" in middleware._paths
