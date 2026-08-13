"""Unsafe/inappropriate input pre-filter.

This is a cheap, deterministic FIRST layer — explicitly not the primary
safety mechanism. The primary mechanism is each generation provider's own
trained safety classifier, already wired into both generation backends:
Gemini's `finishReason` handling (SAFETY/PROHIBITED_CONTENT/BLOCKLIST/
RECITATION -> None) in gemini_service.py, and Claude's `stop_reason ==
"refusal"` handling in anthropic_service.py. Those are far more capable
than any keyword list.

What this pre-filter buys that the provider classifiers can't: it runs
before retrieval, so an obviously unsafe query never spends an embedding
call, a Qdrant/BM25 round trip, or a reranker pass — and it still fires on
the extractive fast-path, which never reaches an LLM (and therefore never
reaches a provider's safety classifier) at all.
"""

from __future__ import annotations

import re

# Deliberately small and named by category, not exhaustive — a keyword list
# is not a substitute for a trained classifier (see module docstring). Each
# pattern targets clearly unambiguous phrasing to minimize false positives
# on legitimate queries (e.g. a medical/educational question about suicide
# prevention resources should not necessarily trip this — that tradeoff is
# a product decision, not something a regex should resolve; keep patterns
# narrow rather than broad).
_UNSAFE_PATTERNS: dict[str, re.Pattern[str]] = {
    "self_harm": re.compile(r"\b(kill myself|commit suicide|end my (own )?life)\b", re.IGNORECASE),
    "violence_instructions": re.compile(
        r"\bhow (do i|to) (make|build|construct) a (bomb|explosive|pipe bomb)\b", re.IGNORECASE
    ),
    "csae": re.compile(r"\b(child sexual abuse|csam)\b", re.IGNORECASE),
}


def check_unsafe_input(text: str) -> str | None:
    """Returns the matched category name, or None if nothing matched."""
    for category, pattern in _UNSAFE_PATTERNS.items():
        if pattern.search(text):
            return category
    return None
