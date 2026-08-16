"""Builds the generation backend, keeping the harness's generator a
configuration choice rather than a hardcoded import — the task's "another
model/provider can still be plugged in later" requirement.

One backend ships today: a local vLLM server (`vllm`). A remote-API backend
lived here previously and was removed once local generation was working, for
one reason worth recording — the remote call was ~2.1s median, almost all of
it network round trip, which no amount of tuning fits inside a 200ms budget.
Local generation is what makes the latency target reachable at all, so
carrying a second backend meant maintaining a path the deployment could never
actually use.

The seam is deliberately still here. `Generator` in harness.py is a Protocol,
not a base class, so adding a backend means writing one class with a
`generate()` method and a `.model` attribute and adding a branch below —
nothing else in the pipeline changes.
"""

from __future__ import annotations

import os

from voice_rag.pipeline.generation.vllm_service import LocalVllmGenerationService

_BACKENDS = ("vllm",)


def build_generator(*, total_budget_seconds: float | None = None):
    """`total_budget_seconds` is forwarded when the caller wants a budget
    different from the backend's own default — main.py uses this to give
    phase-two refinement a longer budget than the in-request path."""
    backend = os.environ.get("VOICE_RAG_GENERATION_BACKEND", "vllm").strip().lower()
    kwargs = {} if total_budget_seconds is None else {"total_budget_seconds": total_budget_seconds}

    if backend == "vllm":
        return LocalVllmGenerationService(**kwargs)
    raise ValueError(f"Unknown VOICE_RAG_GENERATION_BACKEND={backend!r}; expected one of {_BACKENDS}")
