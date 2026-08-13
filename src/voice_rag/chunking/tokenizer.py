"""Token-counting abstraction used by the chunker.

Chunk-length decisions need a token count, but which tokenizer is "correct"
depends on which embedding model ends up indexing the text (Phase 4, BGE-M3 —
an XLM-RoBERTa-family SentencePiece tokenizer, a heavy transformers
dependency this module deliberately does not require). Phase 1's analysis
also found that whitespace word counts under-count information density for
compounding languages like Sanskrit, so the default here is explicitly an
approximation, not presented as exact.

Callers that need exact BGE-M3 token counts should pass a ``token_counter``
built from that model's tokenizer (see embeddings/service.py once Phase 4
lands); the chunker itself is written against the ``TokenCounter`` protocol
below and does not care which implementation it gets.
"""

from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    def __call__(self, text: str) -> int: ...


def whitespace_token_counter(text: str) -> int:
    """Approximate token count via whitespace splitting.

    Known to under-count for compounding/agglutinative languages (Sanskrit,
    Tamil, Malayalam) since a single whitespace-delimited "word" can carry
    multiple morphemes' worth of information — verified against real
    MSMARCO-XI passages in Phase 1 (Sanskrit mean 42.9 words/passage vs.
    Hindi 60.0, not because Sanskrit passages are shorter in content, but
    because a whitespace splitter counts compounded words as one token).
    Adequate for a first-pass length budget; not a substitute for the real
    embedding model's tokenizer when precision matters (e.g. tight context
    windows in the generation stage).
    """
    if not text:
        return 0
    return len(text.split())
