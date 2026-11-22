"""Centralised extraction settings, separate from the full application Settings so tests need no database."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field

from app.config import PROJECT_ROOT, Settings
from app.extraction.model_config import ModelRegistry


class ExtractionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Retries after the first attempt, for provider failures and for invalid responses together.
    max_retries: int = Field(default=2, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=1.0, ge=0)
    max_backoff_seconds: float = Field(default=30.0, ge=0)
    # Failed responses are kept for debugging, cut to this many characters.
    raw_response_max_chars: int = Field(default=20_000, ge=1_000)
    chars_per_token: float = Field(default=4.0, gt=0)
    # Share of the usable context window kept free, because token counts are estimates.
    context_safety_margin: float = Field(default=0.1, ge=0, le=0.5)
    # Neighbouring passages on each side of a passage that are offered as context.
    context_passages: int = Field(default=1, ge=0, le=10)
    # A run stops after this many failed extractions in a row (0 = never): a wrong key or a model
    # that is down would otherwise fail every remaining passage. Other failures never stop a run.
    max_consecutive_failures: int = Field(default=5, ge=0)
    # Units the model reports with less confidence than this are flagged for review.
    low_confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    # Earlier units of the same chapter offered as possible targets when extracting relationships.
    relationship_context_units: int = Field(default=10, ge=0, le=50)
    # Pin a prompt version per category (e.g. {"concept": "1.0.0"}); the latest is used otherwise.
    prompt_versions: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Settings) -> ExtractionSettings:
        return cls(
            max_retries=settings.extraction_max_retries,
            retry_backoff_seconds=settings.extraction_retry_backoff_seconds,
            max_backoff_seconds=settings.extraction_max_backoff_seconds,
            raw_response_max_chars=settings.extraction_raw_response_max_chars,
            chars_per_token=settings.passage_chars_per_token,
            context_safety_margin=settings.extraction_context_safety_margin,
            context_passages=settings.extraction_context_passages,
            max_consecutive_failures=settings.extraction_max_consecutive_failures,
            low_confidence_threshold=settings.extraction_low_confidence_threshold,
            relationship_context_units=settings.extraction_relationship_context_units,
            prompt_versions=settings.extraction_prompt_versions,
        )


def load_model_registry(settings: Settings) -> ModelRegistry:
    """Build the model registry from the application settings."""
    path = Path(settings.extraction_models_file)
    return ModelRegistry.from_file(
        path if path.is_absolute() else PROJECT_ROOT / path,
        default_id=settings.extraction_model_id,
        overrides=settings.extraction_model_overrides,
    )


def load_environment(env_file: Path | None = None) -> Mapping[str, str]:
    """Process environment, with values from .env filling in what is not exported (the environment wins)."""
    values = {k: v for k, v in dotenv_values(env_file or PROJECT_ROOT / ".env").items() if v is not None}
    return {**values, **os.environ}


