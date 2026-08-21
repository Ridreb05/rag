"""Regression coverage for per-language retrieval routing.

Before this, `_answer_query` always searched the single process-wide
`services.collection`/`services.bm25`, ignoring the `language` argument
entirely. These tests build a minimal fake `Services` (no real Qdrant/BM25/
GPU models) and prove `language` actually selects which language's data gets
queried -- and that an unconfigured language 400s instead of silently
falling back to the wrong one.
"""

import numpy as np
import pytest
from fastapi import HTTPException

from voice_rag.api import main
from voice_rag.pipeline.embeddings.service import EmbeddingResult
from voice_rag.pipeline.generation.harness import GenerationHarness


class _SpyBm25:
    def __init__(self):
        self.search_calls = 0

    def search(self, query_text, top_k):
        self.search_calls += 1
        return []


class _SpyQdrantClient:
    def __init__(self):
        self.query_points_collections: list[str] = []

    def query_points(self, collection_name, query, using, limit):
        self.query_points_collections.append(collection_name)
        return type("Result", (), {"points": []})()

    def retrieve(self, collection_name, ids, with_payload=True):
        return []


class _FakeEmbedding:
    def embed_query(self, text, max_length=512):
        return EmbeddingResult(dense=np.zeros((1, 4), dtype=np.float32), sparse=[{}])


class _FakeGenerator:
    model = "fake-model-v0"

    def generate(self, request, max_tokens=2048):
        return None


def _make_services(languages: list[str]) -> main.Services:
    services = main.Services()
    services.embedding = _FakeEmbedding()
    services.qdrant_client = _SpyQdrantClient()
    services.collections = {lang: f"chunks_{lang}_vtest" for lang in languages}
    services.bm25_indexes = {lang: _SpyBm25() for lang in languages}
    services.off_topic_gates = {lang: None for lang in languages}
    services.harness = GenerationHarness(generator=_FakeGenerator())
    services.refine_harness = services.harness
    services.stt = None
    return services


def test_answer_query_routes_bm25_search_to_the_requested_language(monkeypatch):
    services = _make_services(["hi", "bn"])
    monkeypatch.setattr(main, "services", services)

    main._answer_query("query text", "hi", 10)

    assert services.bm25_indexes["hi"].search_calls == 1
    assert services.bm25_indexes["bn"].search_calls == 0

    main._answer_query("query text", "bn", 10)

    assert services.bm25_indexes["hi"].search_calls == 1  # unchanged
    assert services.bm25_indexes["bn"].search_calls == 1


def test_answer_query_routes_qdrant_search_to_the_requested_language(monkeypatch):
    services = _make_services(["hi", "bn"])
    monkeypatch.setattr(main, "services", services)

    main._answer_query("query text", "bn", 10)

    assert services.qdrant_client.query_points_collections == ["chunks_bn_vtest", "chunks_bn_vtest"]


def test_answer_query_rejects_unconfigured_language(monkeypatch):
    services = _make_services(["hi", "bn"])
    monkeypatch.setattr(main, "services", services)

    with pytest.raises(HTTPException) as exc_info:
        main._answer_query("query text", "fr", 10)

    assert exc_info.value.status_code == 400
