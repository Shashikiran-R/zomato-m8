"""
src/config.py
─────────────
Typed application configuration loaded from environment variables / .env file.
Uses pydantic-settings for strict validation and helpful error messages.

Usage:
    from src.config import settings

    print(settings.gemini_api_key)
    print(settings.max_candidates)
"""

import sys
import logging
from pathlib import Path
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Python version guard ────────────────────────────────────────────────────
if sys.version_info < (3, 10):
    import warnings
    warnings.warn(
        f"Python >= 3.10 is recommended. Current version: {sys.version}. "
        "Some type-hint syntax (e.g. `str | None`) requires 3.10+. "
        "Please upgrade your Python interpreter for full compatibility.",
        UserWarning,
        stacklevel=1,
    )

logger = logging.getLogger(__name__)

# ── Settings Model ──────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Central configuration for the AI-Powered Restaurant Recommendation System.

    All values are loaded from environment variables or the .env file at
    project root. Missing required fields raise a clear ValidationError at
    import time.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,      # GEMINI_API_KEY == gemini_api_key
        extra="ignore",            # silently ignore unknown env vars
    )

    # ── Required ────────────────────────────────────────────────────────────
    groq_api_key: str = Field(
        ...,
        description="Groq API key. Get one at https://console.groq.com/keys",
    )

    # ── Data ────────────────────────────────────────────────────────────────
    data_path: Path = Field(
        default=Path("data/processed/restaurants.parquet"),
        description="Path to the preprocessed restaurants Parquet file.",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq model identifier to use for recommendations.",
    )
    max_candidates: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum number of candidate restaurants sent to the LLM.",
    )

    # ── API ──────────────────────────────────────────────────────────────────
    api_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the FastAPI backend (used by the Streamlit UI).",
    )
    api_host: str = Field(
        default="0.0.0.0",
        description="Host address for the FastAPI server.",
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Port for the FastAPI server.",
    )

    # ── LLM Retry / Timeout ─────────────────────────────────────────────────
    llm_timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        description="Seconds to wait for a Gemini API response before timing out.",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Maximum number of retries for transient LLM API failures.",
    )

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("groq_api_key")
    @classmethod
    def api_key_must_not_be_placeholder(cls, v: str) -> str:
        """Ensure the API key has been set and is not the default placeholder."""
        placeholder = "your_groq_api_key_here"
        if not v or v.strip() == "" or v.strip() == placeholder:
            raise ValueError(
                "GROQ_API_KEY is not set or still contains the placeholder value. "
                "Please set a valid API key in your .env file."
            )
        return v.strip()

    @field_validator("log_level")
    @classmethod
    def log_level_must_be_valid(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {valid}. Got: '{v}'")
        return upper

    @model_validator(mode="after")
    def warn_if_data_path_missing(self) -> "Settings":
        """
        Emit a startup warning if the Parquet file does not yet exist.
        This is expected on a fresh clone before running data_ingestion.
        """
        if not self.data_path.exists():
            logger.warning(
                "Dataset not found at '%s'. "
                "Run `python -m src.data_ingestion` to generate it before starting the API.",
                self.data_path,
            )
        return self


# ── Singleton accessor ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    Use this throughout the application to avoid re-reading .env on every call:

        from src.config import get_settings
        settings = get_settings()
    """
    return Settings()


# ── Module-level convenience alias ──────────────────────────────────────────
# Allows `from src.config import settings` for simple use-cases.
# For dependency injection in FastAPI, prefer get_settings().
try:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logger.debug("Configuration loaded successfully: model=%s, max_candidates=%d",
                 settings.llm_model, settings.max_candidates)
except Exception as exc:  # noqa: BLE001
    # Config errors at import time should be loud but not crash unrelated tests.
    logger.error("Failed to load configuration: %s", exc)
    settings = None  # type: ignore[assignment]
    raise
