import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _load_builder_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_full_index.py"
    spec = importlib.util.spec_from_file_location("build_full_index_under_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeQdrant:
    def __init__(self):
        self.exists = False
        self.points = 0
        self.created: list[dict] = []
        self.updated: list[dict] = []

    def collection_exists(self, _name):
        return self.exists

    def create_collection(self, **kwargs):
        self.exists = True
        self.created.append(kwargs)

    def delete_collection(self, _name):
        self.exists = False
        self.points = 0

    def count(self, **_kwargs):
        return SimpleNamespace(count=self.points)

    def update_collection(self, **kwargs):
        self.updated.append(kwargs)


class _FakeVector(list):
    def tolist(self):
        return list(self)


def test_builder_uses_bulk_mode_checkpoints_then_finalizes(monkeypatch, tmp_path):
    builder = _load_builder_module()
    client = _FakeQdrant()
    chunks = pd.DataFrame(
        [
            {"chunk_id": "c1", "passage_id": "p1", "language": "hi", "text": "first"},
            {"chunk_id": "c2", "passage_id": "p2", "language": "hi", "text": "second"},
        ]
    )
    bm25_batches: list[list[str]] = []
    finalization_calls: list[tuple[str, int]] = []

    class _FakeEmbeddingService:
        def embed(self, texts, batch_size):
            assert batch_size == 128
            return SimpleNamespace(
                dense=[_FakeVector([0.0] * 1024) for _ in texts],
                sparse=[{1: 1.0} for _ in texts],
            )

    class _FakeBm25:
        def __init__(self, _path):
            pass

        def upsert_batch(self, chunk_ids, _texts):
            bm25_batches.append(chunk_ids)

    def _upsert(_client, _collection, indexed_chunks, *, batch_size):
        assert batch_size == 1024
        client.points += len(indexed_chunks)
        return len(indexed_chunks)

    def _wait(_client, collection, expected_points, **_kwargs):
        finalization_calls.append((collection, expected_points))
        assert client.points == expected_points

    monkeypatch.setattr(builder, "INDEX_DIR", tmp_path / "full_index")
    monkeypatch.setattr(builder, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(builder.pd, "read_parquet", lambda _path: chunks)
    monkeypatch.setattr(builder, "get_client", lambda **_kwargs: client)
    monkeypatch.setattr(builder, "EmbeddingService", _FakeEmbeddingService)
    monkeypatch.setattr(builder, "Bm25Index", _FakeBm25)
    monkeypatch.setattr(builder, "upsert_chunks", _upsert)
    monkeypatch.setattr(builder, "_wait_for_search_indexes", _wait)

    builder.main(["--index-version", "test1", "--reset", "--embed-chunk-size", "2"])

    state_path = tmp_path / "full_index" / "hi_validation_test1.state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["stage"] == "completed"
    assert state["next_offset"] == 2
    assert state["dense_hnsw_deferred"] is False
    assert client.created[0]["hnsw_config"].m == 0
    assert client.created[0]["optimizers_config"].indexing_threshold == 0
    assert client.updated[0]["hnsw_config"].m == 16
    assert bm25_batches == [["c1", "c2"]]
    assert finalization_calls == [("chunks_hi_vtest1", 2)]

    # Simulate a Pod stopping after all uploads but before Qdrant reports its
    # final graph/sparse index ready. A restart must only finalize — never
    # reload the embedding model or replay already committed rows.
    state["stage"] = "optimizing"
    state["dense_hnsw_deferred"] = True
    state["completed_at"] = None
    state_path.write_text(json.dumps(state), encoding="utf-8")
    bm25_batches.clear()
    finalization_calls.clear()

    class _MustNotEmbed:
        def __init__(self):
            raise AssertionError("optimizer resume must not initialize the embedding model")

    def _must_not_upsert(*_args, **_kwargs):
        raise AssertionError("optimizer resume must not re-upload chunks")

    monkeypatch.setattr(builder, "EmbeddingService", _MustNotEmbed)
    monkeypatch.setattr(builder, "upsert_chunks", _must_not_upsert)
    builder.main(["--index-version", "test1", "--embed-chunk-size", "2"])

    resumed_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert resumed_state["stage"] == "completed"
    assert bm25_batches == []
    assert finalization_calls == [("chunks_hi_vtest1", 2)]


def test_builder_parser_accepts_faster_durable_upserts():
    builder = _load_builder_module()

    args = builder._parse_args(["--upsert-batch-size", "1024", "--optimizer-wait-seconds", "15"])

    assert args.upsert_batch_size == 1024
    assert args.optimizer_wait_seconds == 15
