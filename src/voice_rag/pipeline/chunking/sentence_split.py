"""Script-aware sentence splitting without a heavy NLP dependency.

A proper Indic sentence splitter (e.g. indic-nlp-library) pulls in a chain
of dependencies (morfessor, etc.) that are easy to get wrong on a fresh
machine. Since dataset analysis found MSMARCO-XI passages are already
short (median 34-60 words depending on language) and splitting is only
needed for the long tail, a regex-based splitter is a deliberately simpler,
dependency-free choice — the accuracy ceiling that matters here is "don't
mangle the rare long passage," not "publication-quality sentence
segmentation."

Recognizes:
- Devanagari/Bengali-family danda and double danda (used by Hindi, Marathi,
  Nepali, Sanskrit, Bengali, Assamese): । ॥
- Urdu/Arabic-script full stop: ۔
- Standard Latin terminators: . ! ?

Known limitation: a period immediately followed by whitespace is always
treated as a sentence boundary, so abbreviations ("Dr. Smith") will be
split where a real sentence boundary detector would not. This is an
accepted trade-off given the dataset's passage lengths — see module
docstring.
"""

from __future__ import annotations

import re

_TERMINATORS = "।॥۔!?."
_SENTENCE_BOUNDARY = re.compile(rf"(?<=[{_TERMINATORS}])\s+")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]
