"""Tests the retry/error-recovery logic in GeminiGenerationService using
httpx's MockTransport — no real network or API key needed, so this runs in
the default fast suite (unlike test_gemini_integration.py)."""

import httpx
import pytest

from voice_rag.pipeline.generation.gemini_service import GeminiGenerationService

VALID_BODY = {
    "candidates": [
        {
            "content": {"parts": [{"text": '{"answer_text": "ok", "claims": []}'}]},
            "finishReason": "STOP",
        }
    ]
}


def make_service(transport: httpx.MockTransport) -> GeminiGenerationService:
    svc = GeminiGenerationService(api_key="fake-key", max_retries=3, backoff_base_seconds=0.01)
    svc._client = httpx.Client(transport=transport)
    return svc


def test_retries_on_500_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, json={"error": "server error"})
        return httpx.Response(200, json=VALID_BODY)

    svc = make_service(httpx.MockTransport(handler))
    resp = svc._post_with_retries({}, trace_id="t1")

    assert resp.status_code == 200
    assert calls["n"] == 3


def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=VALID_BODY)

    svc = make_service(httpx.MockTransport(handler))
    resp = svc._post_with_retries({}, trace_id="t2")

    assert resp.status_code == 200
    assert calls["n"] == 2


def test_gives_up_after_max_retries():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": "always fails"})

    svc = make_service(httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        svc._post_with_retries({}, trace_id="t3")

    assert calls["n"] == svc.max_retries + 1  # initial attempt + retries, no more


def test_does_not_retry_on_400_client_error():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    svc = make_service(httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        svc._post_with_retries({}, trace_id="t4")

    assert calls["n"] == 1  # no retries for a non-retryable client error
