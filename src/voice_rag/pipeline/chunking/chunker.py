"""Chunking strategies: fixed-token fallback, sentence-aware (default),
metadata-aware tagging, and a schema hook for hierarchical chunking (inert
on this dataset).

Per the dataset analysis: passages are already short, pre-segmented units
(median 34-60 words depending on language). This chunker's default behavior
for the overwhelming majority of passages is therefore a no-op — return the
passage as a single chunk — and only engages sentence-aware or fixed-token
splitting for the long tail that exceeds ``max_tokens_before_split``.
"""

from __future__ import annotations

from pydantic import BaseModel

from voice_rag.pipeline.chunking.sentence_split import split_sentences
from voice_rag.pipeline.chunking.tokenizer import TokenCounter, whitespace_token_counter


class ChunkingConfig(BaseModel):
    max_tokens_before_split: int = 512  # passages at or under this are kept whole (Strategy B default)
    window_tokens: int = 256  # Strategy A/B window size once splitting is triggered
    overlap_tokens: int = 64  # Strategy A/B overlap between consecutive windows


class Chunk(BaseModel):
    chunk_id: str
    passage_id: str
    language: str
    chunk_index: int
    text: str
    token_count: int
    strategy: str  # "whole_passage" | "sentence_aware" | "fixed_token_fallback"
    # Hierarchical-chunking hook: inert for MSMARCO-XI (no document/section
    # structure), populated once the system is pointed at real long-form
    # documents.
    level: str = "passage"
    parent_id: str | None = None


def _fixed_token_windows(words: list[str], window_tokens: int, overlap_tokens: int) -> list[list[str]]:
    """Strategy A: fallback fixed-size word windows with overlap, for text that
    sentence-aware packing can't reduce below the window size on its own
    (a single pathologically long "sentence")."""
    if overlap_tokens >= window_tokens:
        raise ValueError("overlap_tokens must be smaller than window_tokens")
    step = window_tokens - overlap_tokens
    windows = []
    for start in range(0, len(words), step):
        window = words[start : start + window_tokens]
        if window:
            windows.append(window)
        if start + window_tokens >= len(words):
            break
    return windows


def _pack_sentences(
    sentences: list[str], token_counter: TokenCounter, window_tokens: int, overlap_tokens: int
) -> list[str]:
    """Strategy B: greedily pack sentences into windows <= window_tokens,
    carrying the trailing sentences of one window into the start of the next
    for overlap. A single sentence that alone exceeds window_tokens is
    passed through untouched here — the caller falls back to Strategy A for it."""
    windows: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = token_counter(sent)
        if current and current_tokens + sent_tokens > window_tokens:
            windows.append(current)
            # carry trailing sentences worth ~overlap_tokens into the next window
            overlap: list[str] = []
            overlap_count = 0
            for s in reversed(current):
                t = token_counter(s)
                if overlap_count + t > overlap_tokens and overlap:
                    break
                overlap.insert(0, s)
                overlap_count += t
            current = list(overlap)
            current_tokens = overlap_count
        current.append(sent)
        current_tokens += sent_tokens
    if current:
        windows.append(current)
    return [" ".join(w) for w in windows]


def chunk_passage(
    passage_id: str,
    language: str,
    text: str,
    config: ChunkingConfig | None = None,
    token_counter: TokenCounter = whitespace_token_counter,
) -> list[Chunk]:
    config = config or ChunkingConfig()
    text = text.strip()
    if not text:
        return []

    total_tokens = token_counter(text)
    if total_tokens <= config.max_tokens_before_split:
        return [
            Chunk(
                chunk_id=f"{passage_id}#0",
                passage_id=passage_id,
                language=language,
                chunk_index=0,
                text=text,
                token_count=total_tokens,
                strategy="whole_passage",
            )
        ]

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        # No usable sentence boundaries in a long passage — go straight to
        # fixed-token windows (Strategy A) rather than packing a "sentence"
        # that is the entire passage.
        words = text.split()
        windows = _fixed_token_windows(words, config.window_tokens, config.overlap_tokens)
        return [
            Chunk(
                chunk_id=f"{passage_id}#{i}",
                passage_id=passage_id,
                language=language,
                chunk_index=i,
                text=" ".join(w),
                token_count=token_counter(" ".join(w)),
                strategy="fixed_token_fallback",
            )
            for i, w in enumerate(windows)
        ]

    packed = _pack_sentences(sentences, token_counter, config.window_tokens, config.overlap_tokens)
    chunks: list[Chunk] = []
    for i, window_text in enumerate(packed):
        wt = token_counter(window_text)
        if wt > config.window_tokens * 1.5:
            # One oversized sentence dominated this window — split it directly
            # with the fixed-token fallback instead of shipping an oversized chunk.
            words = window_text.split()
            sub_windows = _fixed_token_windows(words, config.window_tokens, config.overlap_tokens)
            for sw in sub_windows:
                sw_text = " ".join(sw)
                chunks.append(
                    Chunk(
                        chunk_id=f"{passage_id}#{len(chunks)}",
                        passage_id=passage_id,
                        language=language,
                        chunk_index=len(chunks),
                        text=sw_text,
                        token_count=token_counter(sw_text),
                        strategy="fixed_token_fallback",
                    )
                )
        else:
            chunks.append(
                Chunk(
                    chunk_id=f"{passage_id}#{len(chunks)}",
                    passage_id=passage_id,
                    language=language,
                    chunk_index=len(chunks),
                    text=window_text,
                    token_count=wt,
                    strategy="sentence_aware",
                )
            )
    return chunks
