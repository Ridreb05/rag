from voice_rag.chunking.chunker import ChunkingConfig, chunk_passage
from voice_rag.chunking.sentence_split import split_sentences
from voice_rag.chunking.tokenizer import whitespace_token_counter


def test_short_passage_is_kept_whole():
    text = "This is a short passage with a handful of words."
    chunks = chunk_passage("p1", "en", text)
    assert len(chunks) == 1
    assert chunks[0].strategy == "whole_passage"
    assert chunks[0].text == text
    assert chunks[0].chunk_id == "p1#0"


def test_empty_text_produces_no_chunks():
    assert chunk_passage("p1", "en", "") == []
    assert chunk_passage("p1", "en", "   ") == []


def test_long_passage_with_sentences_is_split():
    sentence = "This is one sentence with several words in it for testing purposes."
    text = " ".join([sentence] * 40)  # well over the default 512-token ceiling
    config = ChunkingConfig(max_tokens_before_split=100, window_tokens=50, overlap_tokens=10)
    chunks = chunk_passage("p2", "en", text, config=config)
    assert len(chunks) > 1
    assert all(c.strategy in ("sentence_aware", "fixed_token_fallback") for c in chunks)
    # chunk_index must be contiguous starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # every window should respect the budget with some slack for whole sentences
    for c in chunks:
        assert c.token_count <= config.window_tokens * 1.5 + 5


def test_chunk_ids_are_unique_within_a_passage():
    sentence = "Another test sentence used to force splitting behavior here."
    text = " ".join([sentence] * 30)
    config = ChunkingConfig(max_tokens_before_split=80, window_tokens=40, overlap_tokens=8)
    chunks = chunk_passage("p3", "hi", text, config=config)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_pathologically_long_single_sentence_falls_back_to_fixed_token():
    # one giant "sentence" with no terminators at all
    text = " ".join(["word"] * 300)
    config = ChunkingConfig(max_tokens_before_split=100, window_tokens=50, overlap_tokens=10)
    chunks = chunk_passage("p4", "en", text, config=config)
    assert len(chunks) > 1
    assert all(c.strategy == "fixed_token_fallback" for c in chunks)


def test_overlap_must_be_smaller_than_window():
    import pytest

    from voice_rag.chunking.chunker import _fixed_token_windows

    with pytest.raises(ValueError):
        _fixed_token_windows(["a"] * 20, window_tokens=10, overlap_tokens=10)


def test_split_sentences_devanagari_danda():
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है।"
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].endswith("।")


def test_split_sentences_latin_punctuation():
    text = "This is one. This is two! Is this three?"
    sentences = split_sentences(text)
    assert len(sentences) == 3


def test_split_sentences_empty_string():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_whitespace_token_counter():
    assert whitespace_token_counter("") == 0
    assert whitespace_token_counter("one two three") == 3
