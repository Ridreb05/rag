"""Phase 8 — local LLM generation via llama.cpp (docs/09's "small quantized
local model" choice, docs/24 tech stack).

Model choice note — a real constraint discovered on this dev machine, not
assumed from the blueprint: the blueprint recommends a quantized 7-8B model
(Qwen2.5-7B/Llama-3.1-8B). Measured on this machine's 8GB VRAM GPU, with
BGE-M3 + the reranker warm (~2.2-3.5GB), a 4-bit 7B model (~4.5GB) leaves
little to no headroom for KV cache and risks OOM under concurrent load. This
uses Qwen2.5-3B-Instruct (Q4_K_M GGUF, ~2GB) instead for this specific
machine — the interface is model-agnostic, so swapping to a larger model on
a GPU with more headroom (production sizing per
docs/09-roadmap-and-summary.md) is a constructor argument change only.

llama.cpp is used (rather than transformers + bitsandbytes) specifically
because it has robust prebuilt CUDA wheels for Windows (verified: bitsandbytes'
Windows support is historically unreliable, and abetlen/llama-cpp-python
publishes prebuilt win_amd64 CUDA 13.0 wheels — see pyproject.toml's
llama-cpp-python-cu130 index) and native GBNF/JSON-schema constrained
decoding, which the harness's structured-output requirement (docs/03) needs
directly rather than via post-hoc parsing.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"


def _ensure_cuda_dlls_discoverable() -> None:
    """llama-cpp-python's CUDA backend (ggml-cuda.dll) dynamically links
    against cudart64_13.dll / cublas64_13.dll at load time. This machine has
    no system-wide CUDA Toolkit install — those DLLs only exist bundled
    inside torch's package directory, which isn't on Windows' DLL search
    path by default. Verified failure without this:
    ``FileNotFoundError: Could not find module '...\\llama_cpp\\lib\\llama.dll'
    (or one of its dependencies)`` — the error names llama.dll itself
    because Windows doesn't report which *dependency* was actually missing,
    which is what made this worth a real repro-and-fix rather than a guess.
    `os.add_dll_directory` (Python 3.8+, Windows-only) is the documented way
    to extend the search path for ctypes/CDLL loads; plain PATH mutation is
    not reliably honored under Windows' "safe DLL search mode."
    """
    if sys.platform != "win32":
        return
    try:
        import torch

        torch_lib_dir = Path(torch.__file__).parent / "lib"
        if torch_lib_dir.is_dir():
            os.add_dll_directory(str(torch_lib_dir))
    except ImportError:
        pass


class LocalLlmService:
    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = DEFAULT_FILENAME,
        n_gpu_layers: int = -1,  # -1 = offload all layers to GPU
        n_ctx: int = 4096,
    ):
        _ensure_cuda_dlls_discoverable()
        from llama_cpp import Llama

        logger.info("Loading local LLM %s/%s (n_gpu_layers=%s)", repo_id, filename, n_gpu_layers)
        self._llm = Llama.from_pretrained(
            repo_id=repo_id, filename=filename, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.2,
        json_schema: dict | None = None,
    ) -> str:
        response_format = None
        if json_schema is not None:
            response_format = {"type": "json_object", "schema": json_schema}
        out = self._llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature, response_format=response_format
        )
        return out["choices"][0]["message"]["content"]

    def chat_json(
        self, messages: list[dict[str, str]], json_schema: dict, max_tokens: int = 512, temperature: float = 0.2
    ) -> dict:
        raw = self.chat(messages, max_tokens=max_tokens, temperature=temperature, json_schema=json_schema)
        return json.loads(raw)
