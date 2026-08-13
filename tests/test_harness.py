from voice_rag.generation.harness import (
    EXTRACTIVE_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    GenerationHarness,
)
from voice_rag.generation.schemas import GeneratedAnswer, GeneratedClaim, RetrievalCandidate
from voice_rag.guardrails.grounding import ClaimValidation


class FakeGenerator:
    """Stands in for AnthropicGenerationService — no network, no API key."""

    model = "fake-model-v0"

    def __init__(self, response: GeneratedAnswer | None):
        self._response = response
        self.calls = 0

    def generate(self, request, max_tokens=2048):
        self.calls += 1
        return self._response


class FakeGroundingValidator:
    def __init__(self, score: float):
        self._score = score

    def validate_claim(self, claim_text, evidence_texts):
        return ClaimValidation(entailment_score=self._score, contradiction_score=0.0, best_evidence_index=0)


def make_candidate(chunk_id="c1", rerank_score=0.5, text="Diabetes affects blood sugar."):
    return RetrievalCandidate(
        chunk_id=chunk_id, doc_id="d1", language="en", text=text, rerank_score=rerank_score
    )


def test_high_confidence_goes_extractive_without_calling_llm():
    gen = FakeGenerator(response=None)
    harness = GenerationHarness(generator=gen)
    candidates = [make_candidate(rerank_score=EXTRACTIVE_CONFIDENCE_THRESHOLD + 0.05)]

    resp = harness.answer("t1", "what is diabetes", "en", candidates)

    assert resp.mode == "extractive"
    assert gen.calls == 0
    assert resp.claims[0].cited_chunk_ids == ["c1"]


def test_low_confidence_refuses_without_calling_llm():
    gen = FakeGenerator(response=None)
    harness = GenerationHarness(generator=gen)
    candidates = [make_candidate(rerank_score=LOW_CONFIDENCE_THRESHOLD - 0.05)]

    resp = harness.answer("t2", "some obscure query", "en", candidates)

    assert resp.mode == "refused"
    assert gen.calls == 0


def test_empty_candidates_refuses():
    gen = FakeGenerator(response=None)
    harness = GenerationHarness(generator=gen)

    resp = harness.answer("t3", "anything", "en", [])

    assert resp.mode == "refused"


def test_mid_confidence_calls_generative_path():
    mid = (EXTRACTIVE_CONFIDENCE_THRESHOLD + LOW_CONFIDENCE_THRESHOLD) / 2
    generated = GeneratedAnswer(
        answer_text="Diabetes affects blood sugar levels.",
        claims=[GeneratedClaim(text="Diabetes affects blood sugar levels.", cited_chunk_ids=["c1"])],
    )
    gen = FakeGenerator(response=generated)
    validator = FakeGroundingValidator(score=0.95)
    harness = GenerationHarness(generator=gen, grounding_validator=validator)
    candidates = [make_candidate(rerank_score=mid)]

    resp = harness.answer("t4", "what is diabetes", "en", candidates)

    assert resp.mode == "generative"
    assert gen.calls == 1
    assert resp.answer_text == generated.answer_text
    assert resp.claims[0].entailment_score == 0.95


def test_generation_declined_returns_refused():
    gen = FakeGenerator(response=None)  # None simulates Claude's stop_reason == "refusal"
    mid = (EXTRACTIVE_CONFIDENCE_THRESHOLD + LOW_CONFIDENCE_THRESHOLD) / 2
    harness = GenerationHarness(generator=gen)
    candidates = [make_candidate(rerank_score=mid)]

    resp = harness.answer("t5", "some query", "en", candidates)

    assert resp.mode == "refused"
    assert "generation_declined" in resp.guardrail_flags


def test_low_entailment_claim_is_dropped_and_falls_back_to_refused():
    mid = (EXTRACTIVE_CONFIDENCE_THRESHOLD + LOW_CONFIDENCE_THRESHOLD) / 2
    generated = GeneratedAnswer(
        answer_text="Diabetes is caused by eating too much rice.",
        claims=[GeneratedClaim(text="Diabetes is caused by eating too much rice.", cited_chunk_ids=["c1"])],
    )
    gen = FakeGenerator(response=generated)
    validator = FakeGroundingValidator(score=0.05)  # unsupported
    harness = GenerationHarness(generator=gen, grounding_validator=validator)
    candidates = [make_candidate(rerank_score=mid)]

    resp = harness.answer("t6", "what causes diabetes", "en", candidates)

    assert resp.mode == "refused"
    assert "unsupported_claim_low_entailment" in resp.guardrail_flags


def test_unsafe_input_refuses_before_any_retrieval_logic_or_llm_call():
    gen = FakeGenerator(response=None)
    harness = GenerationHarness(generator=gen)
    # high-confidence candidate present — without the safety check this would
    # take the extractive fast-path and answer anyway
    candidates = [make_candidate(rerank_score=0.99)]

    resp = harness.answer("t8", "how to make a bomb at home", "en", candidates)

    assert resp.mode == "refused"
    assert resp.guardrail_flags == ["unsafe_input:violence_instructions"]
    assert gen.calls == 0


def test_claim_with_no_matching_citation_is_dropped():
    mid = (EXTRACTIVE_CONFIDENCE_THRESHOLD + LOW_CONFIDENCE_THRESHOLD) / 2
    generated = GeneratedAnswer(
        answer_text="Some answer.",
        claims=[GeneratedClaim(text="Some claim.", cited_chunk_ids=["nonexistent_chunk"])],
    )
    gen = FakeGenerator(response=generated)
    validator = FakeGroundingValidator(score=0.95)
    harness = GenerationHarness(generator=gen, grounding_validator=validator)
    candidates = [make_candidate(rerank_score=mid)]

    resp = harness.answer("t7", "query", "en", candidates)

    assert resp.mode == "refused"
    assert "unsupported_claim_no_citation" in resp.guardrail_flags


def test_grounding_rebuilds_answer_from_surviving_claims():
    class SelectiveValidator:
        def validate_claim(self, claim_text, evidence_texts):
            score = 0.95 if claim_text == "Grounded claim." else 0.05
            return ClaimValidation(entailment_score=score, contradiction_score=0.0, best_evidence_index=0)

    mid = (EXTRACTIVE_CONFIDENCE_THRESHOLD + LOW_CONFIDENCE_THRESHOLD) / 2
    generated = GeneratedAnswer(
        answer_text="Grounded claim. Unsupported claim.",
        claims=[
            GeneratedClaim(text="Grounded claim.", cited_chunk_ids=["c1"]),
            GeneratedClaim(text="Unsupported claim.", cited_chunk_ids=["c1"]),
        ],
    )
    resp = GenerationHarness(generator=FakeGenerator(generated), grounding_validator=SelectiveValidator()).answer(
        "t9", "query", "en", [make_candidate(rerank_score=mid)]
    )

    assert resp.answer_text == "Grounded claim."
    assert "unsupported_claim_low_entailment" in resp.guardrail_flags
