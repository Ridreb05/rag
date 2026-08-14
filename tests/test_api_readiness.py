import json

from voice_rag.apps.api_gateway import main


def test_completed_index_state_requires_manifest_completion(monkeypatch, tmp_path):
    state_path = tmp_path / "hi_validation_full1.state.json"
    monkeypatch.setattr(main, "REQUIRE_COMPLETE_INDEX", True)
    monkeypatch.setattr(main, "INDEX_STATE_PATH", state_path)

    state_path.write_text(
        json.dumps({"stage": "indexing", "total": 12, "next_offset": 12, "dense_hnsw_deferred": True}),
        encoding="utf-8",
    )
    assert main._read_completed_index_state() == (False, 12)

    state_path.write_text(
        json.dumps({"stage": "completed", "total": 12, "next_offset": 12, "dense_hnsw_deferred": False}),
        encoding="utf-8",
    )
    assert main._read_completed_index_state() == (True, 12)


def test_completed_index_state_is_not_required_for_local_dev(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "REQUIRE_COMPLETE_INDEX", False)
    monkeypatch.setattr(main, "INDEX_STATE_PATH", tmp_path / "missing.state.json")

    assert main._read_completed_index_state() == (True, None)
