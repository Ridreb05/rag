"""Grounding / hallucination-detection validator.

Checks a generated claim against its cited evidence chunk(s) with a
multilingual NLI cross-encoder, scoring entailment probability per claim
rather than producing one opaque groundedness number for the whole answer.

Model note — a real, verified coverage gap, not assumed: this uses
MoritzLaurer/mDeBERTa-v3-base-mnli-xnli, fine-tuned on XNLI. XNLI's
fine-tuning data covers 15 languages, and only 2 of MSMARCO-XI's 14
languages are among them (Hindi, Urdu — confirmed against XNLI's language
list). The other 12 languages rely on the base mDeBERTa-v3 encoder's
zero-shot cross-lingual transfer (it was pretrained on CC100, which does
cover all 14), which is a real but weaker guarantee than direct fine-tuning.
This is a known, flagged follow-up risk, not silently assumed to work
uniformly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Verified against the model's own config.json id2label — not assumed.
_ENTAILMENT_INDEX = 0
_NEUTRAL_INDEX = 1
_CONTRADICTION_INDEX = 2


@dataclass
class ClaimValidation:
    entailment_score: float
    contradiction_score: float
    best_evidence_index: int  # which evidence text produced entailment_score


class GroundingValidator:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        # Keep the heavy optional model import inside the validator. This lets
        # the API start with its other guardrails if a deployment image has an
        # incompatible optional Transformers backend; ``main`` logs that
        # degraded state instead of crashing the entire service.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self.device)
        self._model.eval()
        # sanity-check the label ordering we hardcoded above against the
        # model actually loaded, so a future model swap fails loudly
        # instead of silently scoring the wrong class.
        id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
        assert id2label[_ENTAILMENT_INDEX] == "entailment", id2label
        assert id2label[_CONTRADICTION_INDEX] == "contradiction", id2label

    @torch.inference_mode()
    def _scores(self, premise_hypothesis_pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
        """Returns [(entailment_prob, contradiction_prob), ...]."""
        if not premise_hypothesis_pairs:
            return []
        premises = [p for p, _ in premise_hypothesis_pairs]
        hypotheses = [h for _, h in premise_hypothesis_pairs]
        inputs = self._tokenizer(
            premises, hypotheses, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return [
            (float(probs[i, _ENTAILMENT_INDEX]), float(probs[i, _CONTRADICTION_INDEX]))
            for i in range(probs.shape[0])
        ]

    def validate_claim(self, claim_text: str, evidence_texts: list[str]) -> ClaimValidation:
        """A claim is grounded if ANY cited evidence chunk entails it —
        a claim can cite multiple chunks, so this checks all of them."""
        if not evidence_texts:
            return ClaimValidation(entailment_score=0.0, contradiction_score=0.0, best_evidence_index=-1)
        pairs = [(evidence, claim_text) for evidence in evidence_texts]  # premise=evidence, hypothesis=claim
        scored = self._scores(pairs)
        best_idx = max(range(len(scored)), key=lambda i: scored[i][0])
        entailment, contradiction = scored[best_idx]
        return ClaimValidation(
            entailment_score=entailment, contradiction_score=contradiction, best_evidence_index=best_idx
        )

    def detect_contradiction(self, text_a: str, text_b: str) -> float:
        """Symmetric-ish contradiction check between two passages, for the
        conflicting-sources guardrail — max of both premise/hypothesis
        orderings, since NLI contradiction is not perfectly symmetric in
        practice."""
        scored = self._scores([(text_a, text_b), (text_b, text_a)])
        return max(c for _, c in scored)
