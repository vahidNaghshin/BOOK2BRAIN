"""Model configuration, registry and selection. No network, no database."""

import json

import pytest

from app.extraction.errors import ModelConfigError
from app.extraction.model_config import (
    DeploymentType,
    LlmModelConfig,
    ModelRegistry,
    ProviderKind,
    StructuredOutputMode,
)
from tests.extraction_helpers import hosted_config, make_config


def test_local_and_hosted_configurations_are_valid():
    local = make_config()
    hosted = hosted_config()
    assert local.deployment == DeploymentType.LOCAL and local.resolved_base_url == "http://localhost:11434/v1"
    assert hosted.deployment == DeploymentType.HOSTED and hosted.resolved_base_url == "https://api.openai.com/v1"
    assert local.read_api_key({}) is None


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"deployment": "hosted"}, "api_key_env"),
        ({"provider": "openai_compatible"}, "base_url"),
        ({"provider": "openai", "base_url": None}, "base_url"),
        ({"base_url": "ftp://host"}, "http(s)"),
        ({"temperature": 5}, "temperature"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"context_window": 512, "max_output_tokens": 1024}, "smaller than context_window"),
        ({"id": "has space"}, "id"),
        ({"model": ""}, "model"),
        ({"unknown_field": 1}, "unknown_field"),
    ],
)
def test_invalid_model_settings_are_rejected(overrides, fragment):
    with pytest.raises(ValueError, match=fragment):
        make_config(**overrides)


def test_a_pasted_api_key_is_rejected_instead_of_stored():
    with pytest.raises(ValueError, match="NAME of an environment variable"):
        hosted_config(api_key_env="sai_abcdefghijklmnop-1234")


def test_unsupported_structured_output_is_rejected_for_anthropic():
    with pytest.raises(ValueError, match="no JSON mode"):
        make_config(provider="anthropic", deployment="hosted", api_key_env="ANTHROPIC_API_KEY", structured_output="json_mode")
    ok = make_config(provider="anthropic", deployment="hosted", api_key_env="ANTHROPIC_API_KEY", structured_output="none")
    assert ok.structured_output == StructuredOutputMode.NONE


def test_api_key_comes_from_the_environment_only():
    config = hosted_config()
    assert config.read_api_key({"TEST_LLM_KEY": " secret-value "}) == "secret-value"
    with pytest.raises(ModelConfigError, match="TEST_LLM_KEY") as error:
        config.read_api_key({})
    assert "secret" not in str(error.value)
    assert "secret-value" not in json.dumps(config.model_dump(mode="json"))


def test_inactive_model_is_not_ready():
    config = make_config(active=False)
    with pytest.raises(ModelConfigError, match="inactive"):
        config.check_ready({})


def test_generation_parameters_reflect_capabilities():
    config = make_config(
        temperature=0.2, top_p=0.9, extra_params={"seed": 7, "mirostat": 2}, unsupported_params=["mirostat"]
    )
    assert config.generation_parameters() == {
        "max_output_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "structured_output": "json_mode",
        "seed": 7,
    }
    no_temperature = make_config(supports_temperature=False)
    assert "temperature" not in no_temperature.generation_parameters()


def test_configurations_are_immutable():
    with pytest.raises(ValueError):
        make_config().model = "other"


def test_registry_switches_between_models_and_categories():
    local, hosted = make_config(), hosted_config()
    registry = ModelRegistry([local, hosted], default_id="local-test", overrides={"equation": "hosted-test"})
    assert registry.for_category("concept").id == "local-test"
    assert registry.for_category("equation").id == "hosted-test"
    assert [c.id for c in registry.active()] == ["local-test", "hosted-test"]
    other = ModelRegistry([local, hosted], default_id="hosted-test")
    assert other.for_category("concept").id == "hosted-test"


def test_registry_rejects_bad_selection():
    with pytest.raises(ModelConfigError, match="does not exist"):
        ModelRegistry([make_config()], default_id="missing")
    with pytest.raises(ModelConfigError, match="Duplicate"):
        ModelRegistry([make_config(), make_config()])
    with pytest.raises(ModelConfigError, match="EXTRACTION_MODEL_ID"):
        ModelRegistry([make_config()]).for_category("concept")


def test_registry_loads_from_file_and_explains_invalid_entries(tmp_path):
    good = tmp_path / "models.json"
    good.write_text(json.dumps({"models": [make_config().model_dump(mode="json")]}))
    assert ModelRegistry.from_file(good, "local-test").get("local-test").provider == ProviderKind.OLLAMA

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"models": [{"id": "broken", "provider": "ollama", "model": "m", "deployment": "hosted"}]}))
    with pytest.raises(ModelConfigError, match="broken.*api_key_env"):
        ModelRegistry.from_file(bad)

    for content, message in [("not json", "cannot be read"), ('{"models": 3}', "'models' list")]:
        path = tmp_path / "x.json"
        path.write_text(content)
        with pytest.raises(ModelConfigError, match=message):
            ModelRegistry.from_file(path)
    with pytest.raises(ModelConfigError, match="not found"):
        ModelRegistry.from_file(tmp_path / "absent.json")


def test_shipped_model_file_is_valid():
    from app.config import PROJECT_ROOT

    registry = ModelRegistry.from_file(PROJECT_ROOT / "config" / "llm_models.json")
    assert registry.all()
    assert all(c.api_key_env is None or c.api_key_env.isupper() for c in registry.all())


def test_select_switches_the_default_model_without_changing_the_original():
    registry = ModelRegistry([make_config(), hosted_config()], default_id="local-test", overrides={"equation": "local-test"})
    switched = registry.select("hosted-test")
    assert switched.for_category("concept").id == "hosted-test" and registry.for_category("concept").id == "local-test"
    assert switched.overrides == registry.overrides
    with pytest.raises(ModelConfigError, match="does not exist"):
        registry.select("nope")
