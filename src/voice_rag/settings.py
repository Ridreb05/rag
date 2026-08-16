"""Central settings, loaded from environment variables / .env.

.env is gitignored (see .gitignore) — never commit real keys. This module
is the only place that should read SARVAM_API_KEY etc. directly from the
environment; everything else takes the key as a constructor argument so it
stays mockable in tests.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    sarvam_api_key: str | None = None


settings = Settings()
