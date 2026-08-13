"""Real Sarvam API integration tests — network + a valid SARVAM_API_KEY
required. Excluded from the default run (see pyproject.toml addopts); run
explicitly with:

    uv run pytest -m slow tests/test_sarvam_integration.py
"""

import pytest

from voice_rag.settings import settings

pytestmark = pytest.mark.slow

requires_sarvam_key = pytest.mark.skipif(
    not settings.sarvam_api_key, reason="SARVAM_API_KEY not set in .env"
)


def _normalize_for_comparison(text: str) -> str:
    """Strip punctuation/whitespace differences that real ASR output
    legitimately introduces (e.g. inserted commas at prosodic pauses) —
    this is exactly why the architecture's query-normalization stage
    exists (docs/02-architecture-and-retrieval.md#query-understanding-layer),
    not something a round-trip test should treat as a failure."""
    import re

    return re.sub(r"[,।.!?\s]+", " ", text).strip()


@requires_sarvam_key
def test_tts_then_stt_round_trip_recovers_original_hindi_text():
    from voice_rag.stt.sarvam_client import SarvamSttClient, SarvamTtsClient

    tts = SarvamTtsClient()
    stt = SarvamSttClient()
    text = "मधुमेह एक पुरानी बीमारी है जो रक्त शर्करा को प्रभावित करती है।"

    tts_result = tts.synthesize(text, language_code="hi-IN")
    assert len(tts_result.audio_bytes) > 1000  # a real WAV, not an empty/error payload

    stt_result = stt.transcribe_bytes(tts_result.audio_bytes)
    assert stt_result.language_code == "hi-IN"
    # Real ASR output legitimately differs in punctuation from the source
    # text even on a clean synthesized round-trip (verified: Sarvam inserted
    # a comma at "है, जो" that wasn't in the original "है जो") — word content
    # is what must match, not exact punctuation.
    assert _normalize_for_comparison(stt_result.transcript) == _normalize_for_comparison(text)


