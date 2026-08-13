"""Pure-logic tests for the Sarvam client that need neither network nor a
real API key — kept out of the `slow`-marked integration test file so they
still run in the default fast suite.
"""

import pytest

from voice_rag.settings import settings
from voice_rag.stt.sarvam_client import _api_key


def test_api_key_guard_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.setattr(settings, "sarvam_api_key", None)
    with pytest.raises(RuntimeError, match="No Sarvam API key"):
        _api_key(None)


def test_api_key_guard_accepts_explicit_key(monkeypatch):
    monkeypatch.setattr(settings, "sarvam_api_key", None)
    assert _api_key("explicit-key") == "explicit-key"


def test_api_key_guard_falls_back_to_settings(monkeypatch):
    monkeypatch.setattr(settings, "sarvam_api_key", "from-settings")
    assert _api_key(None) == "from-settings"
