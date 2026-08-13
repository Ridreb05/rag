"""Phase 8 — the orchestrator (docs/03-harness-and-guardrails.md).

Ties together the guardrail gate, the extractive/generative router, the
generator, and the grounding validator into one typed call. This is the
"LLM Harness" stage in the architecture diagram — deliberately not just
prompt -> LLM -> response.
"""

from __future__ import annotations

import logging

import numpy as np

from voice_rag.generation.anthropic_service import AnthropicGenerationService
from voice_rag.generation.schemas import AnswerResponse, Claim, GenerationRequest, RetrievalCandidate
from voice_rag.guardrails.grounding import GroundingValidator
from voice_rag.guardrails.off_topic import REFUSAL_MESSAGE, OffTopicGate, should_refuse
from voice_rag.guardrails.safety import check_unsafe_input

logger = logging.getLogger(__name__)

EXTRACTIVE_CONFIDENCE_THRESHOLD = 0.85  # rerank_score above this -> skip the LLM entirely
LOW_CONFIDENCE_THRESHOLD = 0.2  # rerank_score below this -> refuse rather than generate
ENTAILMENT_THRESHOLD = 0.5  # per-claim grounding cutoff


def _refused(trace_id: str, model_version: str) -> AnswerResponse:
    return AnswerResponse(
        trace_id=trace_id,
        answer_text=REFUSAL_MESSAGE,
        claims=[],
        confidence=0.0,
        guardrail_flags=["low_confidence_or_off_topic"],
        model_version=model_version,
        mode="refused",
    )


def _extractive_answer(trace_id: str, candidates: list[RetrievalCandidate], model_version: str) -> AnswerResponse:
    top = candidates[0]
    return AnswerResponse(
        trace_id=trace_id,
        answer_text=top.text,
        claims=[Claim(text=top.text, cited_chunk_ids=[top.chunk_id], entailment_score=1.0)],
        confidence=top.rerank_score or 0.0,
        model_version=model_version,
        mode="extractive",
    )


class GenerationHarness:
    def __init__(
        self,
        generator: AnthropicGenerationService | None = None,
        grounding_validator: GroundingValidator | None = None,
        off_topic_gate: OffTopicGate | None = None,
    ):
        self.generator = generator or AnthropicGenerationService()
        self.grounding_validator = grounding_validator  # optional — heavy model, load lazily by caller
        self.off_topic_gate = off_topic_gate

    def answer(
        self,
        trace_id: str,
        query_final: str,
        query_language: str,
        candidates: list[RetrievalCandidate],
        query_embedding: np.ndarray | None = None,
    ) -> AnswerResponse:
        unsafe_category = check_unsafe_input(query_final)
        if unsafe_category is not None:
            resp = _refused(trace_id, self.generator.model)
            resp.guardrail_flags = [f"unsafe_input:{unsafe_category}"]
            return resp

        retrieval_confidence = candidates[0].rerank_score or 0.0 if candidates else 0.0

        if self.off_topic_gate is not None and query_embedding is not None:
            if should_refuse(self.off_topic_gate, query_embedding, retrieval_confidence, LOW_CONFIDENCE_THRESHOLD):
                return _refused(trace_id, self.generator.model)
        elif not candidates or retrieval_confidence < LOW_CONFIDENCE_THRESHOLD:
            return _refused(trace_id, self.generator.model)

        if retrieval_confidence >= EXTRACTIVE_CONFIDENCE_THRESHOLD:
            return _extractive_answer(trace_id, candidates, self.generator.model)

        req = GenerationRequest(
            trace_id=trace_id,
            query_final=query_final,
            query_language=query_language,
            candidates=candidates,
            retrieval_confidence=retrieval_confidence,
            mode="generative",
        )
        generated = self.generator.generate(req)
        if generated is None:
            # Claude's own safety classifiers declined — surface as a guardrail
            # outcome, not a crash (docs/03 Safety Guardrail stage).
            resp = _refused(trace_id, self.generator.model)
            resp.guardrail_flags = ["generation_declined"]
            return resp

        claims = [Claim(text=c.text, cited_chunk_ids=c.cited_chunk_ids) for c in generated.claims]
        guardrail_flags: list[str] = []

        if self.grounding_validator is not None:
            chunk_text_by_id = {c.chunk_id: c.text for c in candidates}
            kept_claims = []
            for claim in claims:
                evidence_texts = [chunk_text_by_id[cid] for cid in claim.cited_chunk_ids if cid in chunk_text_by_id]
                if not evidence_texts:
                    guardrail_flags.append("unsupported_claim_no_citation")
                    continue
                validation = self.grounding_validator.validate_claim(claim.text, evidence_texts)
                claim.entailment_score = validation.entailment_score
                if validation.entailment_score < ENTAILMENT_THRESHOLD:
                    guardrail_flags.append("unsupported_claim_low_entailment")
                    continue
                kept_claims.append(claim)
            claims = kept_claims

        if not claims:
            resp = _refused(trace_id, self.generator.model)
            resp.guardrail_flags = guardrail_flags or ["no_grounded_claims"]
            return resp

        return AnswerResponse(
            trace_id=trace_id,
            answer_text=generated.answer_text,
            claims=claims,
            confidence=retrieval_confidence,
            guardrail_flags=guardrail_flags,
            model_version=self.generator.model,
            mode="generative",
        )
