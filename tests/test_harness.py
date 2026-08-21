import numpy as np

from voice_rag.pipeline.generation.harness import (
    EXTRACTIVE_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    GenerationHarness,
)
from voice_rag.pipeline.generation.schemas import GeneratedAnswer, GeneratedClaim, RetrievalCandidate
from voice_rag.pipeline.guardrails.grounding import ClaimValidation
from voice_rag.pipeline.guardrails.off_topic import OffTopicGate


class FakeGenerator:
    """Stands in for a real generation backend — no server, no network."""

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
    gen = FakeGenerator(response=None)  # None simulates the backend declining to return a candidate
    mid = (EXTRACTIVE_CONFIDENCE_THRESHOLD + LOW_CONFIDENCE_THRESHOLD) / 2
    harness = GenerationHarness(generator=gen)
    candidates = [make_candidate(rerank_score=mid)]

    resp = harness.answer("t5", "some query", "en", candidates)

    assert resp.mode == "refused"
    assert "generation_declined" in resp.guardrail_flags


class ExplodingGenerator:
    """Stands in for a provider outage, rate limit, or expired wall-clock budget."""

    model = "exploding-model-v0"

    def __init__(self, exc: Exception):
        self._exc = exc

    def generate(self, request, max_tokens=2048):
        raise self._exc


def test_generator_failure_degrades_to_extractive_instead_of_erroring():
    """A remote generation failure must not become a 5xx: retrieval already
    succeeded, so the top reranked passage is still a grounded answer."""
    mid = (EXTRACTIVE_CONFIDENCE_THRESHOLD + LOW_CONFIDENCE_THRESHOLD) / 2
    harness = GenerationHarness(generator=ExplodingGenerator(TimeoutError("budget exceeded")))
    candidates = [make_candidate(rerank_score=mid, text="Grounded passage text.")]

    resp = harness.answer("t6", "some query", "en", candidates)

    assert resp.mode == "extractive"
    assert resp.answer_text == "Grounded passage text."
    assert resp.claims[0].cited_chunk_ids == ["c1"]
    assert "generation_unavailable_extractive_fallback" in resp.guardrail_flags


def test_generation_skipped_when_remaining_budget_is_too_small():
    """The request-level deadline must be able to pre-empt generation, not just
    time it out after the fact — retrieval's top passage is already grounded,
    so the budget is respected by answering extractively rather than by
    starting a call that cannot finish in time."""
    mid = (EXTRACTIVE_CONFIDENCE_THRESHOLD + LOW_CONFIDENCE_THRESHOLD) / 2
    # Would raise if generation were reached, proving it was never attempted.
    harness = GenerationHarness(generator=ExplodingGenerator(AssertionError("generation must not run")))
    candidates = [make_candidate(rerank_score=mid, text="Grounded passage text.")]

    resp = harness.answer(
        "t8",
        "some query",
        "en",
        candidates,
        remaining_budget_seconds=0.05,
        min_generation_budget_seconds=1.5,
    )

    assert resp.mode == "extractive"
    assert resp.answer_text == "Grounded passage text."
    assert "deadline_exceeded_extractive_fallback" in resp.guardrail_flags


def test_generation_still_runs_when_budget_is_sufficient():
    """The deadline must not disable generation outright — with budget to
    spare, the generative path is still taken."""
    mid = (EXTRACTIVE_CONFIDENCE_THRESHOLD + LOW_CONFIDENCE_THRESHOLD) / 2
    generated = GeneratedAnswer(
        answer_text="Generated answer.",
        claims=[GeneratedClaim(text="Generated answer.", cited_chunk_ids=["c1"])],
    )
    gen = FakeGenerator(response=generated)
    harness = GenerationHarness(generator=gen)
    candidates = [make_candidate(rerank_score=mid, text="Grounded passage text.")]

    resp = harness.answer(
        "t9",
        "some query",
        "en",
        candidates,
        remaining_budget_seconds=30.0,
        min_generation_budget_seconds=1.5,
    )

    assert resp.mode == "generative"
    assert gen.calls == 1


def test_generator_failure_with_no_candidates_refuses():
    """With nothing retrieved there is no grounded fallback, so refuse rather
    than invent an answer."""
    harness = GenerationHarness(generator=ExplodingGenerator(RuntimeError("provider down")))
    # Confidence high enough to reach generation, but the candidate list is
    # emptied to simulate having no usable evidence at fallback time.
    harness_resp = harness.answer("t7", "some query", "en", [])

    assert harness_resp.mode == "refused"


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


def test_shared_harness_instance_uses_per_call_off_topic_gate_override():
    """One GenerationHarness instance, built with no gate at construction
    (mirrors main.py once the gate is resolved per-call from
    services.off_topic_gates[language] rather than baked into the harness),
    must apply whichever gate is passed to a given .answer() call — proving
    per-language retrieval routing actually reaches the guardrail, not just
    the generation prompt.

    Uses a real generated answer (not None) and asserts on gen.calls, not
    just resp.mode: a None response also ends in mode="refused" (via
    "generation_declined"), which would make an off-topic refusal
    indistinguishable from a declined generation if mode were the only
    signal checked."""
    generated = GeneratedAnswer(
        answer_text="Diabetes affects blood sugar levels.",
        claims=[GeneratedClaim(text="Diabetes affects blood sugar levels.", cited_chunk_ids=["c1"])],
    )
    gen = FakeGenerator(response=generated)
    harness = GenerationHarness(generator=gen)  # no off_topic_gate at construction

    candidates = [make_candidate(rerank_score=0.5)]  # between LOW and EXTRACTIVE thresholds
    query_embedding = np.array([1.0, 0.0])

    gate_accepts = OffTopicGate(centroid=np.array([1.0, 0.0]), similarity_threshold=0.35)
    gate_refuses = OffTopicGate(centroid=np.array([0.0, 1.0]), similarity_threshold=0.35)

    resp_a = harness.answer(
        "ta", "query", "hi", candidates, query_embedding=query_embedding, off_topic_gate=gate_accepts
    )
    assert resp_a.mode == "generative"
    assert gen.calls == 1

    resp_b = harness.answer(
        "tb", "query", "bn", candidates, query_embedding=query_embedding, off_topic_gate=gate_refuses
    )
    assert resp_b.mode == "refused"
    assert gen.calls == 1  # unchanged -- refused before generation was ever attempted


def test_answer_without_gate_override_falls_back_to_constructor_gate():
    """Confirms the sentinel default preserves today's behavior for every
    existing caller that never passes off_topic_gate to .answer() — e.g. the
    tests above, and benchmark/latency_benchmark.py's single fixed-gate
    harness."""
    gen = FakeGenerator(response=None)
    gate = OffTopicGate(centroid=np.array([0.0, 1.0]), similarity_threshold=0.35)
    harness = GenerationHarness(generator=gen, off_topic_gate=gate)
    candidates = [make_candidate(rerank_score=0.5)]
    query_embedding = np.array([1.0, 0.0])  # orthogonal to the gate's centroid -> off-topic

    resp = harness.answer("tc", "query", "hi", candidates, query_embedding=query_embedding)

    assert resp.mode == "refused"
