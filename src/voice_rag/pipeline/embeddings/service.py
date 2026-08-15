"""BGE-M3 embedding service.

Wraps FlagEmbedding's BGEM3FlagModel (the reference implementation) behind a
narrow interface so the rest of the system depends on ``EmbeddingService``,
not on FlagEmbedding directly — swapping the backend later (e.g. an ONNX
INT8-quantized export for a tighter latency budget) means changing this
file only.

BGE-M3 produces three representations from a single forward pass:
- dense: a 1024-dim vector (cosine/dot-product ANN search)
- sparse (a.k.a. "lexical weights"): a token_id -> weight dict, the model's
  own learned sparse retrieval signal — this is what stands in for a
  separately-trained SPLADE model in this architecture's single-encoder
  hybrid design.
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

        # Handles for the single-query fast path (see embed_query). Bound once
        # here so the per-call path never re-does setup work.
        self._inner = self._model.model
        self._tokenizer = self._model.tokenizer
        # BGEM3FlagModel leaves the weights on CPU until its first encode()
        # call moves them. Do it once here instead, so the device is settled
        # before any query arrives and the first request doesn't pay for it.
        self._torch_device = torch.device(self.device)
        self._inner.to(self._torch_device)
        self._inner.eval()
        # Same special-token exclusion FlagEmbedding applies when it builds
        # lexical weights — kept identical so fast-path sparse vectors match
        # the ones the index was built with.
        self._unused_token_ids: set[int] = set()
        for token_name in ("cls_token", "eos_token", "pad_token", "unk_token"):
            token = self._tokenizer.special_tokens_map.get(token_name)
            if token is not None:
                self._unused_token_ids.add(int(self._tokenizer.convert_tokens_to_ids(token)))

    def embed(self, texts: list[str], batch_size: int = 32, max_length: int = 512) -> EmbeddingResult:
        """Batch path (index building, corpus sampling) — goes through
        FlagEmbedding's own batching, which is the right tool when throughput
        rather than per-call latency is what matters."""
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

    @torch.inference_mode()
    def embed_query(self, text: str, max_length: int = 512) -> EmbeddingResult:
        """Latency-critical single-query path — one tokenize, one forward.

        Deliberately bypasses ``FlagEmbedding.encode`` rather than calling it
        with a batch of one. Profiled directly (RTX 4060, 13-token Hindi
        queries): ``encode`` costs ~31.5ms while the underlying forward pass
        is ~13ms. The ~18ms gap is per-call wrapper work that a single query
        gets no benefit from — ``encode`` re-runs ``model.to(device)`` and
        ``model.eval()`` over all 568M parameters on every call, tokenizes
        then length-sorts the batch, and most expensively runs the model
        **twice**: once inside its adaptive batch-size probe loop
        (``while flag is False``) and again in the real encode loop.

        This path reuses the model's own ``_dense_embedding`` /
        ``_sparse_embedding`` heads and replicates FlagEmbedding's exact
        lexical-weight post-processing, so the vectors it produces are
        numerically identical to the batch path's — verified against it over
        real corpus queries, which matters because these vectors are compared
        against an index built with the batch path.
        """
        batch = self._tokenizer(
            [text], padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(self._torch_device)
        out = self._inner(batch, return_dense=True, return_sparse=True, return_colbert_vecs=False)

        dense = out["dense_vecs"].float().cpu().numpy().astype(np.float32)
        token_weights = out["sparse_vecs"].squeeze(-1)[0].float().cpu().numpy()
        input_ids = batch["input_ids"][0].tolist()

        sparse: dict[int, float] = {}
        for weight, token_id in zip(token_weights, input_ids, strict=True):
            if token_id in self._unused_token_ids:
                continue
            weight = float(weight)
            if weight > 0 and weight > sparse.get(token_id, 0.0):
                sparse[token_id] = weight
        return EmbeddingResult(dense=dense, sparse=[sparse])
