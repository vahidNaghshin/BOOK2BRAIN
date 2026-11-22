"""LLM model configuration and selection.

A configuration describes one model on one provider: where it runs, what it can do and how to call
it. It never holds a secret: `api_key_env` only *names* the environment variable that holds the key,
and a value that does not look like a variable name is rejected so a pasted key cannot end up in a
configuration file. Configurations are immutable; the execution metadata of an extraction stores a
snapshot of the one it used, so changing a configuration later does not rewrite history.
"""

from __future__ import annotations

import enum
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator

from app.extraction.errors import ModelConfigError

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DeploymentType(enum.StrEnum):
    LOCAL = "local"
    HOSTED = "hosted"


class ProviderKind(enum.StrEnum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    STORMY = "stormy"
    OPENAI_COMPATIBLE = "openai_compatible"  # any server that speaks the OpenAI chat-completions API


class StructuredOutputMode(enum.StrEnum):
    JSON_SCHEMA = "json_schema"  # the provider enforces a JSON schema
    JSON_MODE = "json_mode"  # the provider guarantees syntactically valid JSON only
    NONE = "none"  # prompt only; the response is parsed and validated here


_DEFAULT_BASE_URLS = {
    ProviderKind.OPENAI: "https://api.openai.com/v1",
    ProviderKind.OLLAMA: "http://localhost:11434/v1",
    ProviderKind.ANTHROPIC: "https://api.anthropic.com",
    ProviderKind.STORMY: "https://stormy-api.dolby.net",
}


class LlmModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    id: str
    provider: ProviderKind
    model: str = Field(min_length=1)
    model_version: str | None = None
    deployment: DeploymentType
    base_url: str | None = None
    api_key_env: str | None = None
    context_window: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=0.0, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_output_tokens: int = Field(default=4096, gt=0)
    structured_output: StructuredOutputMode = StructuredOutputMode.NONE
    # Some models (e.g. reasoning models) reject a temperature; the parameter is then not sent.
    supports_temperature: bool = True
    # Provider-specific generation parameters, sent as they are (minus `unsupported_params`).
    extra_params: dict[str, JsonValue] = Field(default_factory=dict)
    unsupported_params: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=60.0, gt=0)
    # Overrides the global retry budget for this model when set.
    max_retries: int | None = Field(default=None, ge=0, le=10)
    # Prices (USD per million tokens) used only to estimate cost; leave unset when unknown or local.
    input_cost_per_million_tokens: float | None = Field(default=None, ge=0)
    output_cost_per_million_tokens: float | None = Field(default=None, ge=0)
    active: bool = True
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError("id must be 1-100 letters, digits, '.', '_' or '-' and start with a letter or digit")
        return value

    @field_validator("api_key_env")
    @classmethod
    def _check_api_key_env(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_NAME.match(value):
            raise ValueError(
                "api_key_env must be the NAME of an environment variable (e.g. OPENAI_API_KEY), "
                "not the key itself. Keys never belong in configuration files."
            )
        return value

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str | None) -> str | None:
        if value is not None and not re.match(r"^https?://[^\s/]+", value):
            raise ValueError("base_url must be an http(s) URL")
        return value.rstrip("/") if value else value

    @model_validator(mode="after")
    def _check_consistency(self) -> LlmModelConfig:
        if self.provider == ProviderKind.OPENAI_COMPATIBLE and not self.base_url:
            raise ValueError("provider 'openai_compatible' requires base_url")
        if self.deployment == DeploymentType.LOCAL and not (self.base_url or self.provider == ProviderKind.OLLAMA):
            raise ValueError("a local model needs base_url (the address of the local service)")
        if self.deployment == DeploymentType.HOSTED and not self.api_key_env:
            raise ValueError("a hosted model needs api_key_env (the environment variable holding its API key)")
        if self.provider in (ProviderKind.ANTHROPIC, ProviderKind.STORMY):
            if self.structured_output != StructuredOutputMode.NONE:
                raise ValueError(f"provider '{self.provider}' has no JSON mode here; use structured_output 'none'")
            if self.deployment != DeploymentType.HOSTED:
                raise ValueError(f"provider '{self.provider}' is a hosted service")
        if self.context_window is not None and self.max_output_tokens >= self.context_window:
            raise ValueError("max_output_tokens must be smaller than context_window")
        return self

    @property
    def resolved_base_url(self) -> str:
        return self.base_url or _DEFAULT_BASE_URLS[self.provider]

    def generation_parameters(self) -> dict[str, Any]:
        """The generation parameters this configuration sends, as recorded in execution metadata."""
        params: dict[str, Any] = {"max_output_tokens": self.max_output_tokens}
        if self.supports_temperature and self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        params["structured_output"] = self.structured_output.value
        params.update(self.sendable_extra_params())
        return params

    def sendable_extra_params(self) -> dict[str, Any]:
        return {k: v for k, v in self.extra_params.items() if k not in self.unsupported_params}

    def read_api_key(self, environ: Mapping[str, str] | None = None) -> str | None:
        """The API key from the environment, or None when this model needs none."""
        if not self.api_key_env:
            return None
        value = (environ if environ is not None else os.environ).get(self.api_key_env, "").strip()
        if not value:
            raise ModelConfigError(
                f"Model '{self.id}' needs an API key: set the environment variable {self.api_key_env}."
            )
        return value

    def check_ready(self, environ: Mapping[str, str] | None = None) -> None:
        """Raise ModelConfigError unless extraction with this model can start."""
        if not self.active:
            raise ModelConfigError(f"Model configuration '{self.id}' is inactive.")
        self.read_api_key(environ)


class ModelRegistry:
    """The available model configurations and which one each extraction task uses."""

    def __init__(
        self,
        configs: Iterable[LlmModelConfig],
        default_id: str | None = None,
        overrides: Mapping[str, str] | None = None,
    ) -> None:
        self._configs: dict[str, LlmModelConfig] = {}
        for config in configs:
            if config.id in self._configs:
                raise ModelConfigError(f"Duplicate model configuration id '{config.id}'.")
            self._configs[config.id] = config
        self.default_id = default_id or None
        self.overrides = dict(overrides or {})
        for model_id in [self.default_id, *self.overrides.values()]:
            if model_id is not None and model_id not in self._configs:
                raise ModelConfigError(f"Selected model configuration '{model_id}' does not exist.")

    @classmethod
    def from_file(
        cls, path: Path, default_id: str | None = None, overrides: Mapping[str, str] | None = None
    ) -> ModelRegistry:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ModelConfigError(f"Model configuration file not found: {path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelConfigError(f"Model configuration file {path} cannot be read: {exc}") from exc
        entries = data.get("models") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            raise ModelConfigError(f"{path} must contain an object with a 'models' list.")
        configs = []
        for index, entry in enumerate(entries):
            try:
                configs.append(LlmModelConfig.model_validate(entry))
            except ValidationError as exc:
                label = entry.get("id", f"#{index + 1}") if isinstance(entry, dict) else f"#{index + 1}"
                details = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'config'}: {e['msg']}" for e in exc.errors())
                raise ModelConfigError(f"Invalid model configuration '{label}': {details}") from exc
        return cls(configs, default_id, overrides)

    def select(self, model_id: str) -> ModelRegistry:
        """A copy of this registry using another default model (category overrides are kept)."""
        return ModelRegistry(self._configs.values(), model_id, self.overrides)

    def get(self, model_id: str) -> LlmModelConfig:
        try:
            return self._configs[model_id]
        except KeyError:
            raise ModelConfigError(f"Unknown model configuration '{model_id}'.") from None

    def all(self) -> list[LlmModelConfig]:
        return list(self._configs.values())

    def active(self) -> list[LlmModelConfig]:
        return [config for config in self._configs.values() if config.active]

    def for_category(self, knowledge_type: str) -> LlmModelConfig:
        """The configuration for one knowledge category: its override, else the default."""
        model_id = self.overrides.get(str(knowledge_type)) or self.default_id
        if model_id is None:
            raise ModelConfigError(
                "No model selected for extraction: set EXTRACTION_MODEL_ID to one of the configured models."
            )
        return self.get(model_id)
