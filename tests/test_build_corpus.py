from voice_rag.pipeline.ingestion.build_corpus import _content_hash


def test_content_hash_normalizes_whitespace():
    assert _content_hash("hello   world") == _content_hash("hello world")


def test_content_hash_is_case_insensitive():
    assert _content_hash("Hello World") == _content_hash("hello world")


def test_content_hash_strips_leading_trailing_whitespace():
    assert _content_hash("  hello world  ") == _content_hash("hello world")


def test_content_hash_distinguishes_different_text():
    assert _content_hash("hello world") != _content_hash("goodbye world")


def test_content_hash_is_deterministic_length():
    h = _content_hash("some passage text")
    assert len(h) == 64  # sha256 hex digest
