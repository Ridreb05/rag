"""Sarvam STT + TTS REST clients (docs/02-architecture-and-retrieval.md#stt-selection).

Batch/synchronous REST endpoints only (files under 30s) — the streaming
WebSocket API (docs/02's actual chosen architecture for partial-transcript
speculative retrieval) is a separate, larger integration and is not
implemented here yet. This gives a real, testable STT/TTS round trip first.

Endpoint details below were verified against Sarvam's live docs on
2026-08-13 (docs.sarvam.ai), not assumed from the blueprint's prose:
- STT: POST https://api.sarvam.ai/speech-to-text, multipart form
  (file, model="saaras:v3", mode), header `api-subscription-key`,
  JSON response {request_id, transcript, language_code}.
- TTS: POST https://api.sarvam.ai/text-to-speech, JSON body
  (text, language_code, speaker, model="bulbul:v3", ...), same auth header,
  JSON response {request_id, audios: [base64 wav, ...]}.
"""

from __future__ import annotations

import base64
import logging

import httpx
from pydantic import BaseModel

from voice_rag.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.sarvam.ai"
STT_MODEL = "saaras:v3"
TTS_MODEL = "bulbul:v3"


class SttResult(BaseModel):
    request_id: str
    transcript: str
    language_code: str | None = None


class TtsResult(BaseModel):
    request_id: str
    audio_bytes: bytes  # decoded from the first entry in `audios`


def _api_key(explicit: str | None) -> str:
    key = explicit or settings.sarvam_api_key
    if not key:
        raise RuntimeError(
            "No Sarvam API key. Set SARVAM_API_KEY in .env, or pass api_key= explicitly."
        )
    return key


class SarvamSttClient:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self.api_key = _api_key(api_key)
        self._client = httpx.Client(timeout=timeout)

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: str = STT_MODEL,
        mode: str = "transcribe",
        content_type: str = "audio/wav",
        language_code: str | None = None,
    ) -> SttResult:
        form_data = {"model": model, "mode": mode}
        if language_code:
            form_data["language_code"] = language_code
        # Sarvam allowlists exact MIME types (e.g. "audio/webm") and rejects
        # codec parameters like "audio/webm;codecs=opus" that
        # MediaRecorder.mimeType sends by default in browsers.
        bare_content_type = content_type.split(";", 1)[0].strip()
        resp = self._client.post(
            f"{BASE_URL}/speech-to-text",
            headers={"api-subscription-key": self.api_key},
            data=form_data,
            files={"file": (filename, audio_bytes, bare_content_type)},
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("Sarvam STT rejected the request: %s %s", resp.status_code, resp.text)
            raise httpx.HTTPStatusError(f"{exc}: {resp.text}", request=exc.request, response=exc.response) from exc
        data = resp.json()
        return SttResult(request_id=data["request_id"], transcript=data["transcript"], language_code=data.get("language_code"))

    def transcribe_file(self, path: str, model: str = STT_MODEL, mode: str = "transcribe") -> SttResult:
        with open(path, "rb") as f:
            return self.transcribe_bytes(f.read(), filename=path, model=model, mode=mode)

    def close(self) -> None:
        self._client.close()


class SarvamTtsClient:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self.api_key = _api_key(api_key)
        self._client = httpx.Client(timeout=timeout)

    def synthesize(
        self,
        text: str,
        language_code: str = "hi-IN",
        speaker: str = "shubh",
        model: str = TTS_MODEL,
        pace: float = 1.0,
    ) -> TtsResult:
        resp = self._client.post(
            f"{BASE_URL}/text-to-speech",
            headers={"api-subscription-key": self.api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "language_code": language_code,
                "speaker": speaker,
                "model": model,
                "pace": pace,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        audio_bytes = base64.b64decode(data["audios"][0])
        return TtsResult(request_id=data["request_id"], audio_bytes=audio_bytes)

    def close(self) -> None:
        self._client.close()
