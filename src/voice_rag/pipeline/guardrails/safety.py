"""Unsafe/inappropriate input pre-filter.

A cheap, deterministic gate, and — since generation moved to a locally
served model — the only input-side safety layer this system has. It was
written as a first pass in front of a hosted provider's trained safety
classifier; that classifier went away with the remote backend, so this
keyword/regex list is no longer backed by a more capable second opinion.
Worth stating plainly rather than leaving the module's original framing to
imply a depth that is no longer there: a keyword list has poor recall on
adversarial phrasing, and the answer-side guardrails (confidence refusal,
off-topic gate, NLI grounding) are what actually carry the weight.

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
