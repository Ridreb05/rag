from voice_rag.pipeline.retrieval.dense.index import (
    DEFAULT_HNSW_M,
    DEFAULT_INDEXING_THRESHOLD_KB,
    enable_dense_hnsw,
    ensure_collection,
    verify_bulk_upload_config,
    verify_search_index_config,
)


class _FakeClient:
    def __init__(self, exists: bool = False):
        self.exists = exists
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.updated: list[dict] = []

    def collection_exists(self, name: str) -> bool:
        return self.exists

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)

    def create_collection(self, **kwargs) -> None:
        self.created.append(kwargs)

    def update_collection(self, **kwargs) -> None:
        self.updated.append(kwargs)


def _collection_info(hnsw_m: int, indexing_threshold: int):
    config = type(
        "Config",
        (),
        {
            "hnsw_config": type("Hnsw", (), {"m": hnsw_m})(),
            "optimizer_config": type("Optimizers", (), {"indexing_threshold": indexing_threshold})(),
        },
    )()
    return type("CollectionInfo", (), {"config": config})()


def test_bulk_collection_defers_hnsw_and_optimizers_and_keeps_hybrid_schema():
    client = _FakeClient()

    ensure_collection(client, "chunks_hi_vbulk", defer_dense_hnsw=True)

    created = client.created[0]
    assert created["hnsw_config"].m == 0
    assert created["optimizers_config"].indexing_threshold == 0
    assert created["vectors_config"]["dense"].size == 1024
    assert set(created["sparse_vectors_config"]) == {"sparse"}


def test_collection_creation_no_longer_builds_unused_language_payload_index():
    """The fake intentionally has no create_payload_index method."""
    client = _FakeClient()

    ensure_collection(client, "chunks_hi_vfull1")

    assert len(client.created) == 1
    assert client.created[0]["hnsw_config"] is None
    assert client.created[0]["optimizers_config"] is None


def test_enable_dense_hnsw_restores_normal_graph_setting():
    client = _FakeClient()

    enable_dense_hnsw(client, "chunks_hi_vbulk")

    assert len(client.updated) == 1
    assert client.updated[0]["collection_name"] == "chunks_hi_vbulk"
    assert client.updated[0]["hnsw_config"].m == DEFAULT_HNSW_M
    assert client.updated[0]["optimizers_config"].indexing_threshold == DEFAULT_INDEXING_THRESHOLD_KB


def test_enable_dense_hnsw_rejects_disabled_value():
    client = _FakeClient()

    try:
        enable_dense_hnsw(client, "chunks_hi_vbulk", m=0)
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_verify_bulk_upload_config_fails_fast_on_server_defaults():
    client = _FakeClient()
    client.get_collection = lambda _name: _collection_info(DEFAULT_HNSW_M, DEFAULT_INDEXING_THRESHOLD_KB)

    try:
        verify_bulk_upload_config(client, "chunks_hi_vbulk")
    except RuntimeError as exc:
        assert "did not apply" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError")


def test_verify_search_index_config_requires_restored_settings():
    client = _FakeClient()
    client.get_collection = lambda _name: _collection_info(DEFAULT_HNSW_M, DEFAULT_INDEXING_THRESHOLD_KB)
    verify_search_index_config(client, "chunks_hi_vbulk")
