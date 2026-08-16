"""Tests the /v1/translate endpoint against a stub backend.

Translation is a display aid for readers who do not read the corpus language.
The properties worth pinning are that it stays off the query path and that its
failures never take a request down with them — the caller already has the
original answer rendered and keeps it either way.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from voice_rag.api import main
from voice_rag.pipeline.generation.vllm_service import LocalVllmGenerationService


class StubHarness:
    def __init__(self, generator):
        self.generator = generator


class GeneratorWithoutTranslation:
    """A backend that implements generation but not translation — the
    endpoint must say so precisely rather than raising."""

    model = "no-translate-v0"

    def generate(self, request, max_tokens=2048):
        return None


def sse(*deltas: str) -> str:
    import json

    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": d}, "index": 0}]}) for d in deltas]
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def make_vllm(handler) -> LocalVllmGenerationService:
    svc = LocalVllmGenerationService(max_retries=0, backoff_base_seconds=0.01)
    svc._client = httpx.Client(transport=httpx.MockTransport(handler))
    return svc


@pytest.fixture
def client(monkeypatch):
    # /v1/translate is the only route under test; the app's other routes need a
    # warm model pool this test deliberately does not build.
    return TestClient(main.app, raise_server_exceptions=False)


def test_translates_via_the_refinement_generator(client, monkeypatch):
    svc = make_vllm(lambda request: httpx.Response(200, text=sse("Yes, ", "Delta flies to Bangalore.")))
    monkeypatch.setattr(main.services, "refine_harness", StubHarness(svc), raising=False)

    resp = client.post("/v1/translate", json={"text": "क्या डेल्टा बैंगलोर के लिए उड़ान भरता है?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "Yes, Delta flies to Bangalore."
    assert body["latency_ms"]["translation_ms"] >= 0


def test_backend_without_translation_reports_501_not_500(client, monkeypatch):
    monkeypatch.setattr(main.services, "refine_harness", StubHarness(GeneratorWithoutTranslation()), raising=False)

    resp = client.post("/v1/translate", json={"text": "कुछ पाठ"})

    assert resp.status_code == 501


def test_backend_failure_is_503_so_the_original_answer_survives(client, monkeypatch):
    svc = make_vllm(lambda request: httpx.Response(500, json={"error": "model server down"}))
    monkeypatch.setattr(main.services, "refine_harness", StubHarness(svc), raising=False)

    resp = client.post("/v1/translate", json={"text": "कुछ पाठ"})

    # Not a 500: the client keeps the answer it already rendered.
    assert resp.status_code == 503


def test_empty_text_is_rejected_before_reaching_the_model(client, monkeypatch):
    called = {"n": 0}

    def handler(request):
        called["n"] += 1
        return httpx.Response(200, text=sse("x"))

    monkeypatch.setattr(main.services, "refine_harness", StubHarness(make_vllm(handler)), raising=False)

    resp = client.post("/v1/translate", json={"text": ""})

    assert resp.status_code == 422
    assert called["n"] == 0


def test_oversized_text_is_rejected(client, monkeypatch):
    """An unbounded field here would be an open text-generation endpoint."""
    monkeypatch.setattr(main.services, "refine_harness", StubHarness(GeneratorWithoutTranslation()), raising=False)

    resp = client.post("/v1/translate", json={"text": "क" * 4001})

    assert resp.status_code == 422
