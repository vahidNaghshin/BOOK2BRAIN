"""Providers: request building per capability, response formats, error mapping. The transport is faked."""

import json

import pytest

from app.extraction.errors import (
    ModelConfigError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from app.extraction.providers import (
    AnthropicProvider,
    HttpResult,
    LlmMessage,
    LlmRequest,
    OpenAIChatProvider,
    create_provider,
)
from tests.extraction_helpers import hosted_config, make_config

REQUEST = LlmRequest(system="SYS", messages=[LlmMessage("user", "hello")], json_schema={"type": "object"})


class Recorder:
    def __init__(self, status=200, body=None, headers=None, error=None):
        self.result = HttpResult(status, headers or {}, body if isinstance(body, str) else json.dumps(body or {}))
        self.error, self.calls = error, []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, dict(headers), json.loads(body), timeout))
        if self.error:
            raise self.error
        return self.result


OPENAI_OK = {
    "model": "gpt-test-2025",
    "system_fingerprint": "fp_1",
    "choices": [{"message": {"content": '{"items": []}'}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
}


def test_openai_json_schema_request_and_response():
    transport = Recorder(body=OPENAI_OK)
    provider = OpenAIChatProvider(hosted_config(), "KEY", transport)
    response = provider.complete(REQUEST)
    url, headers, payload, timeout = transport.calls[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer KEY"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["max_completion_tokens"] == 1024 and payload["temperature"] == 0.0
    assert payload["messages"][0] == {"role": "system", "content": "SYS"}
    assert timeout == 60.0
    assert (response.text, response.model, response.model_version) == ('{"items": []}', "gpt-test-2025", "fp_1")
    assert (response.input_tokens, response.output_tokens, response.finish_reason) == (11, 7, "stop")


def test_json_mode_prompt_only_and_unsupported_parameters():
    cases = {"json_mode": {"type": "json_object"}, "none": None}
    for mode, expected in cases.items():
        transport = Recorder(body=OPENAI_OK)
        OpenAIChatProvider(make_config(structured_output=mode), None, transport).complete(REQUEST)
        _, headers, payload, _ = transport.calls[0]
        assert payload.get("response_format") == expected
        assert "Authorization" not in headers  # local model, no key
        assert "max_tokens" in payload  # compatible servers use max_tokens

    transport = Recorder(body=OPENAI_OK)
    config = make_config(supports_temperature=False, extra_params={"seed": 1, "bad": 2}, unsupported_params=["bad"])
    OpenAIChatProvider(config, None, transport).complete(REQUEST)
    payload = transport.calls[0][2]
    assert "temperature" not in payload and payload["seed"] == 1 and "bad" not in payload


def test_anthropic_request_and_response_format():
    config = make_config(
        id="claude", provider="anthropic", deployment="hosted", api_key_env="ANTHROPIC_API_KEY",
        structured_output="none", model="claude-test",
    )
    body = {
        "model": "claude-test-1",
        "content": [{"type": "text", "text": '{"items": []}'}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 5, "output_tokens": 9},
    }
    transport = Recorder(body=body)
    response = AnthropicProvider(config, "KEY", transport).complete(REQUEST)
    url, headers, payload, _ = transport.calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "KEY" and "anthropic-version" in headers
    assert payload["system"] == "SYS" and payload["max_tokens"] == 1024 and "response_format" not in payload
    assert (response.text, response.model, response.finish_reason, response.output_tokens) == (
        '{"items": []}', "claude-test-1", "length", 9,
    )


@pytest.mark.parametrize(
    "status, headers, error",
    [
        (429, {"Retry-After": "7"}, ProviderRateLimitError),
        (401, {}, ProviderAuthError),
        (403, {}, ProviderAuthError),
        (504, {}, ProviderTimeoutError),
        (500, {}, ProviderConnectionError),
        (400, {}, ProviderRequestError),
    ],
)
def test_http_errors_map_to_provider_errors(status, headers, error):
    provider = OpenAIChatProvider(hosted_config(), "KEY", Recorder(status=status, body="nope", headers=headers))
    with pytest.raises(error) as raised:
        provider.complete(REQUEST)
    if status == 429:
        assert raised.value.retry_after == 7.0
    assert "KEY" not in str(raised.value)
    assert raised.value.retryable == (error not in (ProviderAuthError, ProviderRequestError))


@pytest.mark.parametrize(
    "body", ["not json", "[]", {"choices": []}, {"choices": [{"message": {"content": None}}]}, {"nothing": 1}]
)
def test_malformed_provider_responses(body):
    provider = OpenAIChatProvider(hosted_config(), "KEY", Recorder(body=body))
    with pytest.raises(ProviderResponseError):
        provider.complete(REQUEST)


def test_transport_failures_propagate():
    provider = OpenAIChatProvider(hosted_config(), "KEY", Recorder(error=ProviderTimeoutError("slow")))
    with pytest.raises(ProviderTimeoutError):
        provider.complete(REQUEST)


def test_factory_reads_the_key_from_the_environment_and_hides_it():
    provider = create_provider(hosted_config(), {"TEST_LLM_KEY": "top-secret"}, Recorder(body=OPENAI_OK))
    assert isinstance(provider, OpenAIChatProvider)
    assert "top-secret" not in repr(provider)
    with pytest.raises(ModelConfigError, match="TEST_LLM_KEY"):
        create_provider(hosted_config(), {})
    assert isinstance(create_provider(make_config(), {}), OpenAIChatProvider)  # local: no key needed


def test_stormy_request_and_response_format():
    from urllib.parse import parse_qs

    from app.extraction.providers import StormyProvider

    config = make_config(
        id="stormy", provider="stormy", deployment="hosted", api_key_env="STORMY_API_TOKEN",
        structured_output="none", model="some-model",
    )
    body = {
        "message": '{"items": []}',
        "metadata": {"conversation_id": 3, "message_id": 4, "model": "some-model", "input_tokens": 8, "output_tokens": 2},
    }
    seen = []

    def transport(url, headers, data, timeout):
        seen.append((url, dict(headers), data))
        return HttpResult(200, {}, json.dumps(body))

    request = LlmRequest(
        system="SYS", messages=[LlmMessage("user", "hello"), LlmMessage("assistant", "bad"), LlmMessage("user", "fix")]
    )
    response = StormyProvider(config, "sai_dummy", transport).complete(request)
    url, headers, data = seen[0]
    assert url == "https://stormy-api.dolby.net/api/v1/messages/"
    assert headers["Authorization"] == "Bearer sai_dummy"
    payload = json.loads(parse_qs(data.decode())["message"][0])
    assert payload["model"] == "some-model" and payload["streaming"] is False
    assert payload["user_input"].startswith("SYS") and "hello" in payload["user_input"] and "fix" in payload["user_input"]
    assert (response.text, response.input_tokens, response.output_tokens) == ('{"items": []}', 8, 2)
    assert "sai_dummy" not in repr(StormyProvider(config, "sai_dummy", transport))
    with pytest.raises(ProviderResponseError):
        StormyProvider(config, "k", lambda *a: HttpResult(200, {}, "{}")).complete(request)
    with pytest.raises(ValueError, match="no JSON mode"):
        make_config(provider="stormy", deployment="hosted", api_key_env="STORMY_API_TOKEN", structured_output="json_mode")
