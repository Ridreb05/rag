import json

from voice_rag.api import main


def test_completed_index_state_requires_manifest_completion(monkeypatch, tmp_path):
    state_path = tmp_path / "hi_validation_full1.state.json"
    monkeypatch.setattr(main, "REQUIRE_COMPLETE_INDEX", True)
    monkeypatch.setitem(main.INDEX_STATE_PATHS, "hi", state_path)

    state_path.write_text(
        json.dumps({"stage": "indexing", "total": 12, "next_offset": 12, "dense_hnsw_deferred": True}),
        encoding="utf-8",
    )
    assert main._read_completed_index_state("hi") == (False, 12)

    state_path.write_text(
        json.dumps({"stage": "completed", "total": 12, "next_offset": 12, "dense_hnsw_deferred": False}),
        encoding="utf-8",
    )
    assert main._read_completed_index_state("hi") == (True, 12)


def test_completed_index_state_is_not_required_for_local_dev(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "REQUIRE_COMPLETE_INDEX", False)
    monkeypatch.setitem(main.INDEX_STATE_PATHS, "hi", tmp_path / "missing.state.json")

    assert main._read_completed_index_state("hi") == (True, None)


def test_completed_index_state_is_independent_per_language(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "REQUIRE_COMPLETE_INDEX", True)
    hi_path = tmp_path / "hi.state.json"
    bn_path = tmp_path / "bn.state.json"
    monkeypatch.setitem(main.INDEX_STATE_PATHS, "hi", hi_path)
    monkeypatch.setitem(main.INDEX_STATE_PATHS, "bn", bn_path)

    hi_path.write_text(
        json.dumps({"stage": "completed", "total": 5, "next_offset": 5, "dense_hnsw_deferred": False}),
        encoding="utf-8",
    )
    bn_path.write_text(
        json.dumps({"stage": "indexing", "total": 5, "next_offset": 2, "dense_hnsw_deferred": True}),
        encoding="utf-8",
    )

    assert main._read_completed_index_state("hi") == (True, 5)
    assert main._read_completed_index_state("bn") == (False, 5)
