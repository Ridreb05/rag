"""Tests LocalVllmGenerationService against httpx.MockTransport — no real
vLLM server or GPU needed, so this runs in the default fast suite. Verifies
the contract the harness depends on (prompt construction, retry/timeout
behavior, claim assembly) rather than real generation quality or timing,
which only a live server can produce — see test_vllm_integration.py.
"""

import json

import httpx
import pytest

from voice_rag.pipeline.generation.schemas import GenerationRequest, RetrievalCandidate
from voice_rag.pipeline.generation.vllm_service import LocalVllmGenerationService


def sse_body(*token_deltas: str) -> str:
    """A minimal OpenAI-compatible SSE stream: one `data:` line per token
    delta, terminated by [DONE] — the exact shape vLLM's server emits."""
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": d}, "index": 0, "finish_reason": None}]})
        for d in token_deltas
    ]
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def make_service(transport: httpx.MockTransport, **kwargs) -> LocalVllmGenerationService:
    svc = LocalVllmGenerationService(max_retries=2, backoff_base_seconds=0.01, **kwargs)
    svc._client = httpx.Client(transport=transport)
    return svc


def make_request(n_candidates: int = 3) -> GenerationRequest:
    return GenerationRequest(
        trace_id="t1",
        query_final="मधुमेह क्या है?",
        query_language="hi",
        candidates=[
            RetrievalCandidate(chunk_id=f"c{i}", doc_id=f"d{i}", language="hi", text=f"passage {i}", rerank_score=0.5)
            for i in range(n_candidates)
        ],
        retrieval_confidence=0.5,
        mode="generative",
    )


def test_generate_returns_answer_citing_context_chunks_only():
    """Citations come from which chunks were placed in context, not from
    anything the model claims — see vllm_service.py's generate() docstring
    for why that still preserves grounding validation."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse_body("मधुमेह ", "एक बीमारी है।"))

    svc = make_service(httpx.MockTransport(handler), context_chunks=2)
    result = svc.generate(make_request(n_candidates=5))

    assert result is not None
    assert result.answer_text == "मधुमेह एक बीमारी है।"
    assert len(result.claims) == 1
    # Only the top 2 of 5 candidates were placed in context.
    assert result.claims[0].cited_chunk_ids == ["c0", "c1"]


def test_tolerates_role_only_and_usage_only_chunks():
    """A real vLLM stream opens with a role-only delta and can close with a
    usage chunk carrying an empty `choices` list. Indexing into those blindly
    raises, and the harness's only recovery is to discard a generation that
    actually succeeded — so the parser skips them instead."""
    body = "\n\n".join(
        [
            'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}',
            'data: {"choices":[{"delta":{"content":"real answer"},"index":0}]}',
            'data: {"choices":[],"usage":{"total_tokens":42}}',
            "data: [DONE]",
        ]
    ) + "\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    svc = make_service(httpx.MockTransport(handler))
    result = svc.generate(make_request())

    assert result is not None
    assert result.answer_text == "real answer"


def test_empty_completion_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse_body())

    svc = make_service(httpx.MockTransport(handler))
    assert svc.generate(make_request()) is None


def test_retries_on_500_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(500, json={"error": "server error"})
        return httpx.Response(200, text=sse_body("ok"))

    svc = make_service(httpx.MockTransport(handler))
    result = svc.generate(make_request())

    assert result is not None
    assert calls["n"] == 2


def test_does_not_retry_on_400_client_error():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    svc = make_service(httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        svc.generate(make_request())

    assert calls["n"] == 1


def test_gives_up_after_max_retries():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": "always fails"})

    svc = make_service(httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        svc.generate(make_request())

    assert calls["n"] == svc.max_retries + 1


def test_prompt_uses_only_configured_context_chunk_count():
    """The single largest latency lever this service has — verified as a
    request-shape assertion, since a live server is the only place the
    resulting prefill-time saving is actually observable."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = httpx.Request.read(request) and request.content
        return httpx.Response(200, text=sse_body("ok"))

    svc = make_service(httpx.MockTransport(handler), context_chunks=1)
    svc.generate(make_request(n_candidates=4))

    body = json.loads(captured["payload"])
    user_message = body["messages"][1]["content"]
    assert "[1]" in user_message
    assert "[2]" not in user_message  # only the top 1 chunk reached the prompt
    assert body["max_tokens"] == svc.max_tokens
    assert body["temperature"] == 0.0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
