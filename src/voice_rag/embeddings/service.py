"""Phase 4 — BGE-M3 embedding service.

Wraps FlagEmbedding's BGEM3FlagModel (the reference implementation) behind a
narrow interface so the rest of the system depends on ``EmbeddingService``,
not on FlagEmbedding directly — swapping the backend later (e.g. an ONNX
INT8-quantized export for the latency budget in docs/04-latency-and-caching.md)
means changing this file only.

BGE-M3 produces three representations from a single forward pass:
- dense: a 1024-dim vector (cosine/dot-product ANN search)
- sparse (a.k.a. "lexical weights"): a token_id -> weight dict, the model's
  own learned sparse retrieval signal — this is what stands in for a
  separately-trained SPLADE model in this architecture (see
  docs/02-architecture-and-retrieval.md's single-encoder hybrid decision).
- colbert: per-token multi-vectors (not used in this system; late-interaction
  scoring is not part of the current hybrid design, so it's left disabled by
  default to save compute).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-m3"


@dataclass
class EmbeddingResult:
    dense: np.ndarray  # shape (n, 1024), float32
    sparse: list[dict[int, float]]  # token_id -> weight, one dict per input text


class EmbeddingService:
    def __init__(self, model_name: str = MODEL_NAME, use_fp16: bool | None = None, device: str | None = None):
        from FlagEmbedding import BGEM3FlagModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if use_fp16 is None:
            use_fp16 = self.device == "cuda"
        logger.info("Loading %s on device=%s fp16=%s", model_name, self.device, use_fp16)
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=self.device)

    def embed(self, texts: list[str], batch_size: int = 32, max_length: int = 512) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(dense=np.zeros((0, 1024), dtype=np.float32), sparse=[])
        out = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        sparse = [{int(k): float(v) for k, v in d.items()} for d in out["lexical_weights"]]
        return EmbeddingResult(dense=dense, sparse=sparse)

    def embed_query(self, text: str, max_length: int = 512) -> EmbeddingResult:
        """Single-query convenience wrapper — this is the latency-critical
        call path (docs/04-latency-and-caching.md), batch_size is irrelevant here."""
        return self.embed([text], batch_size=1, max_length=max_length)
