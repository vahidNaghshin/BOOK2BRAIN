"""LLM providers behind one small interface.

The extraction service only knows `LlmProvider.complete(LlmRequest) -> LlmResponse`. Everything that
differs between providers (URL, headers, request body, response format, structured-output support,
which generation parameters are accepted, error mapping) lives here. A new provider is one class plus
an entry in PROVIDER_FACTORIES. Uses the standard library only; the HTTP transport can be replaced,
which is how tests run without network access.
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.extraction.errors import (
    ModelConfigError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from app.extraction.model_config import LlmModelConfig, ProviderKind, StructuredOutputMode

logger = logging.getLogger(__name__)

_ERROR_SNIPPET = 300


@dataclass(frozen=True)
class LlmMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class LlmRequest:
    system: str
    messages: list[LlmMessage]
    # Used when the model supports enforced structured output; otherwise ignored.
    json_schema: dict[str, Any] | None = None


@dataclass
class LlmResponse:
    text: str
    model: str  # the model identifier the provider reports having used
    model_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None  # "stop", or "length" when the output was cut off
    extra: dict[str, Any] = field(default_factory=dict)


class LlmProvider(Protocol):
    config: LlmModelConfig

    def complete(self, request: LlmRequest) -> LlmResponse: ...


@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: Mapping[str, str]
    body: str


Transport = Callable[[str, Mapping[str, str], bytes, float], HttpResult]


def urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> HttpResult:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - http(s) only, validated
            return HttpResult(response.status, dict(response.headers), response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, dict(exc.headers or {}), exc.read().decode("utf-8", "replace"))
    except (TimeoutError, socket.timeout) as exc:
        raise ProviderTimeoutError(f"The request timed out after {timeout:g} seconds.") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise ProviderTimeoutError(f"The request timed out after {timeout:g} seconds.") from exc
        raise ProviderConnectionError(f"The LLM service could not be reached: {exc.reason}") from exc


def _retry_after(headers: Mapping[str, str]) -> float | None:
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except ValueError:
                return None
    return None


def _snippet(body: str) -> str:
    return " ".join(body.split())[:_ERROR_SNIPPET]


def raise_for_status(result: HttpResult) -> None:
    """Map an unsuccessful HTTP answer onto the provider error types. Never includes request headers."""
    status = result.status
    if 200 <= status < 300:
        return
    detail = _snippet(result.body)
    if status == 429:
        raise ProviderRateLimitError(f"Rate limited by the LLM provider. {detail}", retry_after=_retry_after(result.headers))
    if status in (401, 403):
        raise ProviderAuthError(f"The LLM provider rejected the credentials (HTTP {status}).")
    if status in (408, 504):
        raise ProviderTimeoutError(f"The LLM provider timed out (HTTP {status}).")
    if status >= 500:
        raise ProviderConnectionError(f"The LLM provider failed (HTTP {status}). {detail}")
    raise ProviderRequestError(f"The LLM provider rejected the request (HTTP {status}). {detail}")


def _decode(result: HttpResult) -> dict[str, Any]:
    try:
        data = json.loads(result.body)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError("The LLM provider returned a response that is not JSON.") from exc
    if not isinstance(data, dict):
        raise ProviderResponseError("The LLM provider returned an unexpected response shape.")
    return data


class _HttpProvider:
    def __init__(self, config: LlmModelConfig, api_key: str | None, transport: Transport | None = None) -> None:
        self.config = config
        self._api_key = api_key
        self._transport = transport or urllib_transport

    def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", **headers}
        result = self._transport(url, headers, json.dumps(payload).encode("utf-8"), self.config.timeout_seconds)
        raise_for_status(result)
        return _decode(result)

    def __repr__(self) -> str:  # never reveal the key
        return f"{type(self).__name__}(config={self.config.id!r})"


class OpenAIChatProvider(_HttpProvider):
    """OpenAI, Ollama (its /v1 endpoint) and other OpenAI-compatible chat-completions servers."""

    def build_payload(self, request: LlmRequest) -> dict[str, Any]:
        config = self.config
        messages = [{"role": "system", "content": request.system}] + [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        payload: dict[str, Any] = {"model": config.model, "messages": messages}
        limit_key = "max_completion_tokens" if config.provider == ProviderKind.OPENAI else "max_tokens"
        payload[limit_key] = config.max_output_tokens
        if config.supports_temperature and config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.structured_output == StructuredOutputMode.JSON_SCHEMA and request.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "knowledge_extraction", "schema": request.json_schema},
            }
        elif config.structured_output in (StructuredOutputMode.JSON_SCHEMA, StructuredOutputMode.JSON_MODE):
            payload["response_format"] = {"type": "json_object"}
        payload.update(config.sendable_extra_params())
        return payload

    def complete(self, request: LlmRequest) -> LlmResponse:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        data = self._post(f"{self.config.resolved_base_url}/chat/completions", headers, self.build_payload(request))
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderResponseError("The LLM response has no message content.") from exc
        if not isinstance(text, str):
            raise ProviderResponseError("The LLM response content is not text.")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return LlmResponse(
            text=text,
            model=str(data.get("model") or self.config.model),
            model_version=data.get("system_fingerprint") or self.config.model_version,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
        )


class AnthropicProvider(_HttpProvider):
    """Anthropic Messages API. No JSON mode: the response is parsed and validated by the caller."""

    API_VERSION = "2023-06-01"

    def build_payload(self, request: LlmRequest) -> dict[str, Any]:
        config = self.config
        payload: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_output_tokens,
            "system": request.system,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if config.supports_temperature and config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        payload.update(config.sendable_extra_params())
        return payload

    def complete(self, request: LlmRequest) -> LlmResponse:
        headers = {"x-api-key": self._api_key or "", "anthropic-version": self.API_VERSION}
        data = self._post(f"{self.config.resolved_base_url}/v1/messages", headers, self.build_payload(request))
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ProviderResponseError("The LLM response has no content blocks.")
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        if not text:
            raise ProviderResponseError("The LLM response has no text content.")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        stop = data.get("stop_reason")
        return LlmResponse(
            text=text,
            model=str(data.get("model") or self.config.model),
            model_version=self.config.model_version,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            finish_reason="length" if stop == "max_tokens" else stop,
        )


class StormyProvider(_HttpProvider):
    """Stormy API: POST /api/v1/messages with a form field `message` holding JSON.

    The API has no system-prompt, JSON-mode or generation-parameter fields, so the system prompt is
    prepended to the user input and the answer is parsed and validated by the caller. Every call is a
    fresh conversation (no conversation_id), so the full prompt history is sent each time.
    """

    def build_payload(self, request: LlmRequest) -> dict[str, Any]:
        turns = [request.system] + [
            m.content if m.role == "user" else f"[Your previous answer]\n{m.content}" for m in request.messages
        ]
        return {"user_input": "\n\n".join(turns), "model": self.config.model, "streaming": False}

    def complete(self, request: LlmRequest) -> LlmResponse:
        form = urllib.parse.urlencode({"message": json.dumps(self.build_payload(request))}).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key or ''}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        result = self._transport(f"{self.config.resolved_base_url}/api/v1/messages/", headers, form, self.config.timeout_seconds)
        raise_for_status(result)
        data = _decode(result)
        text = data.get("message")
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseError("The Stormy response has no message text.")
        meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        return LlmResponse(
            text=text,
            model=str(meta.get("model") or self.config.model),
            model_version=self.config.model_version,
            input_tokens=meta.get("input_tokens"),
            output_tokens=meta.get("output_tokens"),
            finish_reason=None,
            extra={k: meta[k] for k in ("conversation_id", "message_id") if k in meta},
        )


ProviderFactory = Callable[[LlmModelConfig, str | None, Transport | None], LlmProvider]

PROVIDER_FACTORIES: dict[ProviderKind, ProviderFactory] = {
    ProviderKind.OPENAI: OpenAIChatProvider,
    ProviderKind.OLLAMA: OpenAIChatProvider,
    ProviderKind.OPENAI_COMPATIBLE: OpenAIChatProvider,
    ProviderKind.ANTHROPIC: AnthropicProvider,
    ProviderKind.STORMY: StormyProvider,
}


def create_provider(
    config: LlmModelConfig,
    environ: Mapping[str, str] | None = None,
    transport: Transport | None = None,
) -> LlmProvider:
    """Build the provider for a configuration, reading its API key from the environment."""
    factory = PROVIDER_FACTORIES.get(config.provider)
    if factory is None:
        raise ModelConfigError(f"No provider implementation for '{config.provider}'.")
    return factory(config, config.read_api_key(environ), transport)
