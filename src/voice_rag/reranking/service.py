"""Phase 6 — cross-encoder reranking (docs/02-architecture-and-retrieval.md).

bge-reranker-v2-m3 is a standard `AutoModelForSequenceClassification`
(XLM-RoBERTa-based, single output logit; score = sigmoid(logit)). This
loads it directly via `transformers` rather than through FlagEmbedding's
`FlagReranker` wrapper.

Why not FlagEmbedding, given the embedding service (embeddings/service.py)
does use it: verified directly — `FlagReranker.compute_score_single_gpu`
calls `tokenizer.prepare_for_model`, a method `transformers==5.15.0` no
longer exposes on `XLMRobertaTokenizer` (`AttributeError`). FlagEmbedding's
own metadata declares `transformers<6.0.0`, but its actual reranker code
predates the transformers 5.x tokenizer API change — the declared
constraint is looser than what the code supports. Downgrading transformers
to a compatible 4.x line isn't viable either: this project depends on
`huggingface-hub>=1.27.0` (a major-version-1.x release), and every
transformers 4.x release requires `huggingface-hub<1.0` — confirmed via
`uv add "transformers<5"`, which failed dependency resolution outright.
Loading the reranker model directly through `transformers`' stable
Auto-class API sidesteps both problems and removes a fragile indirection.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class RerankerService:
    def __init__(self, model_name: str = MODEL_NAME, use_fp16: bool | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if use_fp16 is None:
            use_fp16 = self.device == "cuda"
        self._dtype = torch.float16 if use_fp16 else torch.float32

        logger.info("Loading reranker %s on device=%s fp16=%s", model_name, self.device, use_fp16)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=self._dtype)
        self._model.to(self.device)
        self._model.eval()

    @torch.inference_mode()
    def rerank(self, query: str, candidates: list[str], batch_size: int = 32, max_length: int = 512) -> list[float]:
        """Returns one relevance score in [0, 1] per candidate, same order as input."""
        if not candidates:
            return []
        scores: list[float] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            pairs = [[query, c] for c in batch]
            inputs = self._tokenizer(
                pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(self.device)
            logits = self._model(**inputs).logits.view(-1).float()
            batch_scores = torch.sigmoid(logits).tolist()
            scores.extend(batch_scores)
        return scores
