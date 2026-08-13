"""Strategy F — query-aware retrieval granularity.

Classifies a query as narrow / broad / ambiguous using only cheap,
deterministic signals so this stays out of the latency-critical model path
(see docs/02-architecture-and-retrieval.md and docs/04-latency-and-caching.md).

Honesty note vs. the original design sketch: the blueprint described this as
"token count + named-entity count from a fast NER tagger." A real NER
tagger is a model call with its own latency and dependency cost, and is
overkill for a 3-way coarse classification. This implementation uses token
count + a digit/numeral check (catches specific-fact queries: dates, codes,
quantities) instead of NER. Swapping in a real multilingual NER tagger
later is a drop-in replacement of ``_looks_specific`` — the classifier's
public interface (``classify_granularity``) does not need to change.
"""

from __future__ import annotations

import re
from enum import Enum

_DIGIT_PATTERN = re.compile(r"\d")

# Devanagari and other Indic digit blocks, so a numeral written in-script
# (e.g. Devanagari ०-९) is still detected as "specific" — verified these
# ranges appear in MSMARCO-XI passages via the Phase 1 script-consistency check.
_INDIC_DIGIT_PATTERN = re.compile(
    r"[०-९০-৯੦-੯૦-૯୦-୯"
    r"௦-௯౦-౯೦-೯൦-൯۰-۹]"
)


class Granularity(str, Enum):
    NARROW = "narrow"
    BROAD = "broad"
    AMBIGUOUS = "ambiguous"


def _looks_specific(query: str) -> bool:
    return bool(_DIGIT_PATTERN.search(query) or _INDIC_DIGIT_PATTERN.search(query))


def classify_granularity(query: str, narrow_max_tokens: int = 3, broad_min_tokens: int = 8) -> Granularity:
    """Cheap, deterministic — must stay well under 5ms per call.

    - narrow: very short queries, or any query containing a digit/numeral
      (specific-fact lookups: "ICD-10 code for...", "population in 2011").
    - broad: long queries (>= broad_min_tokens words) with no digit signal —
      likely to need synthesis across multiple passages.
    - ambiguous: everything else; caller retrieves at both granularities
      and lets the reranker's score spread decide (see docs/02).
    """
    query = query.strip()
    if not query:
        return Granularity.AMBIGUOUS

    token_count = len(query.split())

    if token_count <= narrow_max_tokens or _looks_specific(query):
        return Granularity.NARROW
    if token_count >= broad_min_tokens:
        return Granularity.BROAD
    return Granularity.AMBIGUOUS
