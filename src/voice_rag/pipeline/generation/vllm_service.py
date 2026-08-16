"""Generative path via a local vLLM server — the generation backend,
built by generation/factory.py:build_generator(). Implements the
`generate(GenerationRequest) -> GeneratedAnswer | None` interface the harness
expects (see harness.py's `Generator` protocol).

Exists specifically to fit generation inside a latency-sensitive budget. The
hosted-API backend this replaced measured ~2.1s median, almost entirely
network round trip to another company's servers — impossible to fit a 200ms
budget, which is why two-phase answering existed to work around it. A model
resident on this process's own GPU has no such hop: measured on the
deployment, generation runs in ~150ms, so it fits inside the request.

Design decisions made for latency, in the order they cost the most:

1. Only the top `context_chunks` candidates (already reranked, so this is a
   slice, not a re-sort) go into the prompt. Fewer context tokens is the
   single largest lever on prefill time, and MSMARCO-XI passages are short
   enough that 1-2 of them are normally sufficient context for a direct
   question — this is a real trade against recall on questions that
   genuinely need to synthesize across more passages, not a free win.
2. No structured-output schema is requested from the model (contrast
   a hosted API's structured-output schema). Grammar-constrained decoding
   has real per-token overhead, and such a schema exists to let the model
   tell us which passage supports which claim. That information doesn't need to be
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
# Confirmed against the model card (huggingface.co/Qwen/Qwen3.5-4B) — no
# "-Instruct" suffix; that repo id does not exist. This is the chat/
# instruction-tuned model directly (the base pretrain checkpoint is the
# separate "Qwen3.5-4B-Base"). It ships in hybrid-thinking mode by default,
# which is exactly what `enable_thinking: False` below turns off.
# `VOICE_RAG_VLLM_MODEL` overrides this if a different model is served.
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

SYSTEM_INSTRUCTION = (
    "Answer only using the passages given below, in the same language as the question. "
    "If they do not answer it, say so. Be brief — a sentence or two."
)


class LocalVllmGenerationService:
    """Talks to a local vLLM OpenAI-compatible server (`vllm serve`) over
    HTTP on localhost. Not a provider SDK, for the same reason
    the STT client isn't one: a single endpoint doesn't justify the
    dependency, and a raw client keeps retry/timeout/streaming behaviour
    visible and testable with httpx.MockTransport rather than hidden in a
    library."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        context_chunks: int | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int = 1,
        backoff_base_seconds: float = 0.2,
        total_budget_seconds: float = 3.0,
    ):
        self.base_url = (base_url or os.environ.get("VOICE_RAG_VLLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or os.environ.get("VOICE_RAG_VLLM_MODEL", DEFAULT_MODEL)
        self.context_chunks = context_chunks if context_chunks is not None else int(
            os.environ.get("VOICE_RAG_VLLM_CONTEXT_CHUNKS", "2")
        )
        # Decode time is linear in this number — the most direct latency lever
        # this service owns. Fitted to measurement rather than chosen round:
        # on the deployment, 20 tokens cost 190ms of a 163ms budget (200ms
        # target minus 37ms measured retrieval). Solving that against a fixed
        # per-request overhead of 40-60ms — HTTP round trip, queueing,
        # tokenisation and prefill, none of which shrink with fewer output
        # tokens — puts marginal cost at ~6.5-7.5ms per token. 14 tokens lands
        # inside budget across that whole range; 16 only does at the optimistic
        # end.
        #
        # This is the sharpest quality trade in the system. Hindi tokenises at
        # roughly 2-4 tokens per word, so 14 tokens is a short sentence, and
        # answers needing more will truncate. Raise it if answers read clipped
        # and accept the latency, or raise the whole budget instead.
        self.max_tokens = max_tokens if max_tokens is not None else int(
            os.environ.get("VOICE_RAG_VLLM_MAX_TOKENS", "14")
        )
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        # Deliberately small. A local model failing is not a network blip
        # that clears up on retry — it is the server still loading or out of
        # memory, so a short budget surfaces that as a fast extractive
        # fallback instead of a slow one.
        self.total_budget_seconds = total_budget_seconds
        # Per-attempt HTTP timeout defaults to the whole budget rather than a
        # fixed constant. A fixed 5s here silently capped the refinement
        # generator — constructed with a 45s budget precisely because nobody
        # waits on it — at 5s per attempt, and would bite hardest on the first
        # request after startup, which pays one-time CUDA warmup on top of
        # normal generation and is exactly the request a demo opens with.
        self._client = httpx.Client(timeout=timeout if timeout is not None else total_budget_seconds)

    def close(self) -> None:
        self._client.close()

    def is_ready(self, timeout: float = 2.0) -> bool:
        """A cheap, short-timeout readiness probe for /v1/health — not called
        on the request path. Unlike a hosted API (a third party's uptime,
        not this deployment's to report on), a broken local vLLM server means
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
        as the rest of the pipeline: bounded by a wall-clock deadline, not
        just an attempt count, and only transient failures are retried."""
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
                        if not line or not line.startswith("data:"):
                            continue
                        data = line.split(":", 1)[1].strip()
                        if data == "[DONE]":
                            break
                        # Tolerate chunks that carry no content: the opening
                        # delta announces only `role`, and a usage-summary
                        # chunk can carry an empty `choices` list entirely.
                        # Indexing [0] blindly turns either into an exception
                        # that the harness can only recover from by discarding
                        # a generation that actually succeeded.
                        try:
                            choices = json.loads(data).get("choices") or []
                            delta = choices[0].get("delta", {}).get("content") if choices else None
                        except (json.JSONDecodeError, AttributeError, IndexError):
                            logger.warning("vllm_stream_chunk_unparsed trace_id=%s", trace_id)
                            continue
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
        signal than model-declared per-claim citations: the NLI validator
        (guardrails/grounding.py) already scores a claim against a *list* of
        candidate evidence texts and picks the best-matching one
        (`best_evidence_index`) rather than trusting a claimed citation
        blindly — a model's own citations are a hint for *which* passage to
        check, not a substitute for checking. Handing the
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
