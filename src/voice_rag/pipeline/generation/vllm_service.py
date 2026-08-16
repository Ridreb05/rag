"""Generative path via a local vLLM server — an alternative backend to
GeminiGenerationService, selected at runtime by
generation/factory.py:build_generator(). Exposes the same
`generate(GenerationRequest) -> GeneratedAnswer | None` interface the harness
expects, so no other part of the pipeline changes to use it.

Exists specifically to fit generation inside a latency-sensitive budget:
Gemini is a real network hop to another company's servers (~2.1s measured
median), which two-phase answering works around rather than eliminates. A
model resident in this process's own GPU, given a short prompt and a small
output cap, can answer in well under a second — small enough that, combined
with the confidence router already skipping the LLM whenever a single
passage suffices, generation stops being the thing that decides whether a
request fits its budget.

Design decisions made for latency, in the order they cost the most:

1. Only the top `context_chunks` candidates (already reranked, so this is a
   slice, not a re-sort) go into the prompt. Fewer context tokens is the
   single largest lever on prefill time, and MSMARCO-XI passages are short
   enough that 1-2 of them are normally sufficient context for a direct
   question — this is a real trade against recall on questions that
   genuinely need to synthesize across more passages, not a free win.
2. No structured-output schema is requested from the model (contrast
   gemini_service.py's `responseSchema`). Grammar-constrained decoding has
   real per-token overhead, and it exists there to let the model tell us
   which passage supports which claim. That information doesn't need to be
   asked for here: since the prompt already contains only the chunks the
   model could possibly be citing, the citation is known before the model
   generates anything. One GeneratedClaim citing every chunk placed in
   context is returned instead — see generate()'s docstring for why this
   does not weaken grounding validation.
3. `enable_thinking` is disabled via `chat_template_kwargs` where the
   configured model's chat template supports it (Qwen's hybrid
   reasoning models do). A model that reasons before answering can spend its
   entire token budget on reasoning tokens the harness never sees as an
   answer at all.
"""

from __future__ import annotations

import json
import logging
import os
import time

import httpx

from voice_rag.pipeline.generation.schemas import GeneratedAnswer, GeneratedClaim, GenerationRequest

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
# Verify this against the model actually served — model naming here tracks
# what was requested, not a repo id independently confirmed to exist at the
# time this was written. `LocalVllmGenerationService` never hardcodes it
# beyond this default; `VOICE_RAG_VLLM_MODEL` overrides it.
DEFAULT_MODEL = "Qwen/Qwen3.5-4B-Instruct"

SYSTEM_INSTRUCTION = (
    "Answer only using the passages given below, in the same language as the question. "
    "If they do not answer it, say so. Be brief — a sentence or two."
)


class LocalVllmGenerationService:
    """Talks to a local vLLM OpenAI-compatible server (`vllm serve`) over
    HTTP on localhost. Not a provider SDK, for the same reason
    gemini_service.py isn't one: one endpoint doesn't justify the dependency,
    and a raw client keeps retry/timeout/streaming behavior fully visible and
    testable with httpx.MockTransport rather than hidden in a library."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        context_chunks: int = 2,
        max_tokens: int = 20,
        timeout: float = 5.0,
        max_retries: int = 1,
        backoff_base_seconds: float = 0.2,
        total_budget_seconds: float = 3.0,
    ):
        self.base_url = (base_url or os.environ.get("VOICE_RAG_VLLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or os.environ.get("VOICE_RAG_VLLM_MODEL", DEFAULT_MODEL)
        self.context_chunks = context_chunks
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        # Deliberately much smaller than Gemini's 8s default: a local model
        # failing is not a network blip that clears up on retry, it is the
        # server still loading or out of memory — a short budget surfaces
        # that as a fast extractive fallback instead of a slow one.
        self.total_budget_seconds = total_budget_seconds
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def is_ready(self, timeout: float = 2.0) -> bool:
        """A cheap, short-timeout readiness probe for /v1/health — not called
        on the request path. Unlike Gemini (a third party's uptime, not
        this deployment's to report on), a broken local vLLM server means
        this deployment silently answers every generative-band query
        extractively, which is worth surfacing rather than only discoverable
        by noticing the mode mix looks wrong."""
        try:
            resp = httpx.get(f"{self.base_url.removesuffix('/v1')}/health", timeout=timeout)
            return resp.status_code == 200
        except httpx.TransportError:
            return False

    def _stream_with_retries(self, payload: dict, trace_id: str) -> tuple[str, float, float]:
        """Returns (completion_text, ttft_ms, total_ms). Same retry contract
        as GeminiGenerationService._post_with_retries: bounded by a wall-clock
        deadline, not just an attempt count, and only transient failures are
        retried."""
        deadline = time.monotonic() + self.total_budget_seconds
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            t_start = time.monotonic()
            t_first_token: float | None = None
            chunks: list[str] = []
            try:
                with self._client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload, timeout=min(self._client.timeout.read or remaining, remaining)
                ) as resp:
                    if resp.status_code == 429 or resp.status_code >= 500:
                        resp.read()
                        raise httpx.HTTPStatusError(
                            f"retryable status {resp.status_code}", request=resp.request, response=resp
                        )
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[len("data: ") :]
                        if data == "[DONE]":
                            break
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                        if delta:
                            if t_first_token is None:
                                t_first_token = time.monotonic()
                            chunks.append(delta)
                total_ms = (time.monotonic() - t_start) * 1000
                ttft_ms = (t_first_token - t_start) * 1000 if t_first_token is not None else total_ms
                return "".join(chunks), ttft_ms, total_ms
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                is_retryable = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and (exc.response.status_code == 429 or exc.response.status_code >= 500)
                )
                if not is_retryable or attempt == self.max_retries:
                    raise
                delay = self.backoff_base_seconds * (2**attempt)
                if time.monotonic() + delay >= deadline:
                    raise
                logger.warning(
                    "vllm_call_failed trace_id=%s attempt=%d/%d error=%s retrying_in_s=%.2f",
                    trace_id,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise TimeoutError(f"vLLM generation exceeded its {self.total_budget_seconds}s budget for trace_id={trace_id}")

    def generate(self, request: GenerationRequest, max_tokens: int | None = None) -> GeneratedAnswer | None:
        """Returns exactly one claim citing every chunk placed in context,
        not chunks the model names itself. This is not a weaker grounding
        signal than Gemini's per-claim citations: the NLI validator
        (guardrails/grounding.py) already scores a claim against a *list* of
        candidate evidence texts and picks the best-matching one
        (`best_evidence_index`) rather than trusting a claimed citation
        blindly — GeminiGenerationService's own citations are a hint for
        *which* passage to check, not a substitute for checking. Handing the
        validator both context chunks here and letting it find the match (or
        fail to, and correctly reject the claim) reaches the same guarantee
        with zero output-schema cost on the model."""
        top = request.candidates[: self.context_chunks]
        context_text = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(top, start=1))
        user_content = f"Passages:\n{context_text}\n\nQuestion ({request.query_language}): {request.query_final}"

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": 0.0,
            "stream": True,
            # Qwen's hybrid-reasoning chat template reads this key to skip
            # emitting <think> tokens. A model/template that doesn't
            # recognize it should ignore an unknown chat_template_kwargs
            # entry rather than error — verify against the served model if
            # generation starts failing after a model swap.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        text, ttft_ms, total_ms = self._stream_with_retries(payload, trace_id=request.trace_id)
        logger.info(
            "vllm_generation_completed trace_id=%s ttft_ms=%.1f total_ms=%.1f context_chunks=%d output_chars=%d",
            request.trace_id,
            ttft_ms,
            total_ms,
            len(top),
            len(text),
        )

        text = text.strip()
        if not text:
            logger.warning("vLLM returned an empty completion for trace_id=%s", request.trace_id)
            return None

        return GeneratedAnswer(
            answer_text=text,
            claims=[GeneratedClaim(text=text, cited_chunk_ids=[c.chunk_id for c in top])],
        )
