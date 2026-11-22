"""The connection-test command line, with a fake provider (no network)."""

import json

from app.extraction import check
from app.extraction.errors import ProviderTimeoutError
from app.extraction.providers import LlmResponse
from tests.extraction_helpers import ScriptedProvider, hosted_config, make_config

SECRET = "sai_dummy_secret_value"


def test_connection_ok_reports_reply_and_tokens_without_the_key():
    config = hosted_config()
    provider = ScriptedProvider(config, [LlmResponse(text="pong", model="gpt-test", input_tokens=5, output_tokens=1)])
    ok, detail = check.test_connection(config, {"TEST_LLM_KEY": SECRET}, provider)
    assert ok and "pong" in detail and "5/1" in detail and SECRET not in detail


def test_connection_failures_are_reported_not_raised():
    config = hosted_config()
    ok, detail = check.test_connection(config, {"TEST_LLM_KEY": SECRET}, ScriptedProvider(config, [ProviderTimeoutError("slow")]))
    assert not ok and "ProviderTimeoutError" in detail
    ok, detail = check.test_connection(config, {})
    assert not ok and "TEST_LLM_KEY" in detail


def test_status_and_listing(tmp_path, capsys):
    models = tmp_path / "m.json"
    models.write_text(json.dumps({"models": [
        hosted_config().model_dump(mode="json"), make_config(active=False).model_dump(mode="json"),
    ]}))
    assert check.main(["--models-file", str(models)], environ={}) == 0
    out = capsys.readouterr().out
    assert "hosted-test" in out and "not ready" in out and "inactive" in out
    assert check.main(["--models-file", str(models)], environ={"TEST_LLM_KEY": SECRET}) == 0
    assert "ready" in capsys.readouterr().out


def test_unknown_model_is_a_clean_error(tmp_path, capsys):
    models = tmp_path / "m.json"
    models.write_text(json.dumps({"models": [make_config().model_dump(mode="json")]}))
    assert check.main(["--models-file", str(models), "--model", "nope"], environ={}) == 2
    assert "Unknown model" in capsys.readouterr().err


def test_dotenv_values_fill_in_but_the_environment_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TEST_LLM_KEY=from-file\nOTHER=file\n")
    monkeypatch.setenv("OTHER", "process")
    loaded = check.load_environment(env)
    assert loaded["TEST_LLM_KEY"] == "from-file" and loaded["OTHER"] == "process"
