"""Integration tests that load real models (BGE-M3, bge-reranker-v2-m3, the
NLI grounding model). Excluded from the default test run (see pyproject.toml
addopts) because they need GPU/network and take minutes to download models
on a cold cache. Run explicitly with:

    uv run pytest -m slow tests/test_ml_integration.py
"""

import pytest

pytestmark = pytest.mark.slow


def test_bge_m3_cross_lingual_similarity_exceeds_unrelated_pair():
    from voice_rag.pipeline.embeddings.service import EmbeddingService

    svc = EmbeddingService()
    result = svc.embed(["what is diabetes", "capital of france", "मधुमेह क्या है"])
    dense = result.dense

    def cosine(a, b):
        import numpy as np

        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    cross_lingual_same_meaning = cosine(dense[0], dense[2])
    unrelated = cosine(dense[0], dense[1])
    assert cross_lingual_same_meaning > unrelated
    assert cross_lingual_same_meaning > 0.7  # verified empirically at ~0.84


def test_reranker_scores_relevant_passage_higher():
    from voice_rag.pipeline.reranking.service import RerankerService

    svc = RerankerService()
    scores = svc.rerank(
        "what is diabetes",
        [
            "diabetes is a chronic disease that affects how the body processes blood sugar",
            "the eiffel tower is located in paris, france",
        ],
    )
    assert scores[0] > 0.9
    assert scores[1] < 0.1
    assert scores[0] > scores[1]


def test_grounding_validator_distinguishes_entailment_from_contradiction():
    from voice_rag.pipeline.guardrails.grounding import GroundingValidator

    v = GroundingValidator()
    evidence = ["Diabetes is a chronic disease that affects how the body processes blood sugar."]

    grounded = v.validate_claim("Diabetes affects blood sugar levels.", evidence)
    unsupported = v.validate_claim("Diabetes is caused by eating too much rice.", evidence)

    assert grounded.entailment_score > 0.9
    assert unsupported.entailment_score < 0.1
    assert grounded.entailment_score > unsupported.entailment_score


def test_grounding_validator_detects_contradiction():
    from voice_rag.pipeline.guardrails.grounding import GroundingValidator

    v = GroundingValidator()
    score = v.detect_contradiction("The Eiffel Tower is in Paris.", "The Eiffel Tower is in London.")
    assert score > 0.9


def test_grounding_validator_works_cross_lingually_for_hindi():
    from voice_rag.pipeline.guardrails.grounding import GroundingValidator

    v = GroundingValidator()
    result = v.validate_claim(
        "मधुमेह रक्त शर्करा को प्रभावित करता है।",
        ["मधुमेह एक पुरानी बीमारी है जो रक्त शर्करा को प्रभावित करती है।"],
    )
    assert result.entailment_score > 0.9
