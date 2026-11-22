"""Helpers for the extraction tests: scripted fake providers, model configs and request builders.

No test here touches the network: a fake provider replaces the HTTP client.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from app.extraction.model_config import LlmModelConfig, ModelRegistry
from app.extraction.prompts import get_prompt
from app.extraction.providers import LlmProvider, LlmRequest, LlmResponse
from app.extraction.results import ExtractionRequest, PassageInput
from app.extraction.service import ExtractionService
from app.extraction.settings import ExtractionSettings
from app.knowledge.types import KnowledgeType
from tests.knowledge_helpers import BUILDERS

# The current prompt version, so tests do not need editing when a new version is released.
LATEST = get_prompt("concept").version


def make_config(**overrides: Any) -> LlmModelConfig:
    data: dict[str, Any] = {
        "id": "local-test",
        "provider": "ollama",
        "model": "test-model:1b",
        "deployment": "local",
        "context_window": 8192,
        "max_output_tokens": 1024,
        "structured_output": "json_mode",
    }
    return LlmModelConfig.model_validate(data | overrides)


def hosted_config(**overrides: Any) -> LlmModelConfig:
    return make_config(
        **{
            "id": "hosted-test",
            "provider": "openai",
            "model": "gpt-test",
            "deployment": "hosted",
            "api_key_env": "TEST_LLM_KEY",
            "context_window": 128_000,
            "structured_output": "json_schema",
        }
        | overrides
    )


class ScriptedProvider:
    """Answers with the next scripted item: a JSON string, an LlmResponse or an exception to raise."""

    def __init__(self, config: LlmModelConfig, script: list[Any]) -> None:
        self.config = config
        self.script = list(script)
        self.requests: list[LlmRequest] = []

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, LlmResponse):
            return step
        return LlmResponse(text=step, model=self.config.model, input_tokens=100, output_tokens=50, finish_reason="stop")


class Harness:
    def __init__(self, service: ExtractionService, provider: ScriptedProvider, sleeps: list[float]) -> None:
        self.service, self.provider, self.sleeps = service, provider, sleeps


def make_harness(script: list[Any], config: LlmModelConfig | None = None, **settings: Any) -> Harness:
    config = config or make_config()
    provider = ScriptedProvider(config, script)
    sleeps: list[float] = []
    service = ExtractionService(
        ModelRegistry([config], default_id=config.id),
        ExtractionSettings(retry_backoff_seconds=1.0, **settings),
        provider_factory=lambda _config: provider,
        environ={"TEST_LLM_KEY": "test-value"},
        sleep=sleeps.append,
    )
    return Harness(service, provider, sleeps)


def make_request(
    knowledge_type: str | KnowledgeType = "concept",
    text: str = "Entropy is a measure of the disorder of a system. Higher entropy means more ways to arrange the parts.",
    pages: tuple[int, int] | None = (12, 12),
    **kwargs: Any,
) -> ExtractionRequest:
    passage = PassageInput(
        id=uuid.uuid4(),
        text=text,
        page_start=pages[0] if pages else None,
        page_end=pages[1] if pages else None,
    )
    return ExtractionRequest(knowledge_type=knowledge_type, passage=passage, **kwargs)


def item(request: ExtractionRequest, content: dict[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
    knowledge_type = KnowledgeType(request.knowledge_type)
    entry: dict[str, Any] = {
        "content": content if content is not None else BUILDERS[knowledge_type](),
        "sources": [{"passage_id": str(request.passage.id), "role": "primary"}],
        "basis": "explicit",
        "confidence": 0.9,
        "is_uncertain": False,
    }
    return entry | fields


def response_json(*items: dict[str, Any], **extra: Any) -> str:
    return json.dumps({"items": list(items), "notes": None} | extra)
