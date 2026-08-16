"""Typed schemas for the generation stage.

These are the Pydantic models the harness passes between stages, wired to
the real generator.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrievalCandidate(BaseModel):
    chunk_id: str
    doc_id: str
    language: str
    text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float | None = None


class GenerationRequest(BaseModel):
    trace_id: str
    query_final: str
    query_language: str
    candidates: list[RetrievalCandidate]
    retrieval_confidence: float
    mode: Literal["extractive", "generative"]


class Claim(BaseModel):
    text: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    entailment_score: float | None = None  # filled in by the grounding validator, not the generator


class GeneratedClaim(BaseModel):
    """What the LLM itself produces — a subset of Claim's fields.
    entailment_score is added afterward by the grounding validator, never by the generator."""

    text: str
    cited_chunk_ids: list[str] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    """What a generation backend returns: answer text plus per-claim
    citations. Backends may obtain this differently — a provider API can
    constrain its output to this schema directly, while vllm_service.py
    builds it from a plain completion (see its generate() docstring for why
    that does not weaken grounding)."""

    answer_text: str
    claims: list[GeneratedClaim]


class AnswerResponse(BaseModel):
    trace_id: str
    answer_text: str
    claims: list[Claim]
    confidence: float
    guardrail_flags: list[str] = Field(default_factory=list)
    pipeline_version: str = "v0"
    model_version: str
    mode: Literal["extractive", "generative", "refused"]
