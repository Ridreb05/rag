"""Picks a generation backend at runtime, so the harness's generator is a
configuration choice rather than a hardcoded import — the "another
model/provider can still be plugged in later" requirement.

Default is `gemini`, unchanged from before this module existed: the CPU Fly
deployment has no GPU to run a local model on, and this module must not
silently change what an already-working, already-submitted deployment does.
The GPU RunPod deployment opts into `vllm` explicitly via its own Dockerfile.
"""

from __future__ import annotations

import os

from voice_rag.pipeline.generation.gemini_service import GeminiGenerationService
from voice_rag.pipeline.generation.vllm_service import LocalVllmGenerationService

_BACKENDS = ("gemini", "vllm")


def build_generator(*, total_budget_seconds: float | None = None):
    """`total_budget_seconds` is forwarded when the caller wants a budget
    different from the backend's own default — main.py uses this to give
    phase-two refinement a longer budget than the in-request path, on
    whichever backend is configured."""
    backend = os.environ.get("VOICE_RAG_GENERATION_BACKEND", "gemini").strip().lower()
    kwargs = {} if total_budget_seconds is None else {"total_budget_seconds": total_budget_seconds}

    if backend == "gemini":
        return GeminiGenerationService(**kwargs)
    if backend == "vllm":
        return LocalVllmGenerationService(**kwargs)
    raise ValueError(f"Unknown VOICE_RAG_GENERATION_BACKEND={backend!r}; expected one of {_BACKENDS}")
