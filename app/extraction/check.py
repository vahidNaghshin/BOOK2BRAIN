"""Command line for listing models and testing LLM connections (debugging aid).

    python -m app.extraction.check                       list models and whether each is ready
    python -m app.extraction.check --model ID            test the connection to one model
    python -m app.extraction.check --all                 test every active model that is ready
    python -m app.extraction.check --model ID --extract  also run a real concept extraction on a sample passage

API keys are read from the environment or from .env, never from the models file, and are never printed.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

from app.config import PROJECT_ROOT
from app.extraction.errors import ExtractionError
from app.extraction.model_config import LlmModelConfig, ModelRegistry
from app.extraction.providers import LlmMessage, LlmProvider, LlmRequest, create_provider
from app.extraction.results import ExtractionRequest, PassageInput
from app.extraction.service import ExtractionService
from app.extraction.settings import ExtractionSettings, load_environment

DEFAULT_MODELS_FILE = PROJECT_ROOT / "config" / "llm_models.json"
PING = "Reply with the single word: pong"
SAMPLE = (
    "Entropy is a measure of the disorder of a system. A system with higher entropy has more ways "
    "of arranging its parts, so isolated systems tend to move toward higher entropy."
)


def test_connection(config: LlmModelConfig, environ: Mapping[str, str], provider: LlmProvider | None = None) -> tuple[bool, str]:
    """Send a one-word prompt. Returns (ok, one-line description); never raises and never includes a key."""
    try:
        config.check_ready(environ)
        provider = provider or create_provider(config, environ)
        started = time.perf_counter()
        response = provider.complete(LlmRequest(system="You are a connectivity test.", messages=[LlmMessage("user", PING)]))
    except ExtractionError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    reply = " ".join(response.text.split())[:60]
    return True, f"{elapsed:.1f}s, reply={reply!r}, model={response.model}, tokens={response.input_tokens}/{response.output_tokens}"


def status(config: LlmModelConfig, environ: Mapping[str, str]) -> str:
    if not config.active:
        return "inactive"
    try:
        config.check_ready(environ)
    except ExtractionError as exc:
        return f"not ready: {exc}"
    return "ready"


def run_extraction(registry: ModelRegistry, config: LlmModelConfig, environ: Mapping[str, str]) -> bool:
    service = ExtractionService(registry.select(config.id), ExtractionSettings(), environ=environ)
    passage = PassageInput(id=uuid.uuid4(), text=SAMPLE, page_start=1, page_end=1)
    result = service.extract(ExtractionRequest(knowledge_type="concept", passage=passage))
    run = result.execution
    print(f"  status={result.status.value} items={len(result.items)} attempts={run.attempt_count} prompt={run.prompt_identifier}")
    for found in result.items:
        print(f"  concept: {found.content.name} - {found.content.definition[:80]}")
    for issue in result.issues[:5]:
        print(f"  issue: [{issue.code}] {issue.message}")
    if result.failure:
        print(f"  failure: {result.failure.code}: {result.failure.message}")
    return result.status.value != "failed"


def main(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List models and test LLM connections.")
    parser.add_argument("--models-file", type=Path, default=DEFAULT_MODELS_FILE)
    parser.add_argument("--model", help="model configuration id to test")
    parser.add_argument("--all", action="store_true", help="test every active, ready model")
    parser.add_argument("--extract", action="store_true", help="also run a sample concept extraction")
    args = parser.parse_args(argv)
    environ = load_environment() if environ is None else environ

    try:
        registry = ModelRegistry.from_file(args.models_file)
        if args.model:
            targets = [registry.get(args.model)]
        elif args.all:
            targets = [c for c in registry.active() if status(c, environ) == "ready"]
        else:
            targets = []
    except ExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not targets:
        print(f"{'ID':<34}{'PROVIDER':<18}{'MODEL':<34}STATUS")
        for c in registry.all():
            print(f"{c.id:<34}{c.provider.value:<18}{c.model:<34}{status(c, environ)}")
        print("\nTest one with --model ID, or all ready models with --all.")
        return 0

    failures = 0
    for config in targets:
        print(f"{config.id} ({config.provider.value}, {config.model})")
        ok, detail = test_connection(config, environ)
        print(f"  connection: {'OK' if ok else 'FAILED'} - {detail}")
        if ok and args.extract:
            ok = run_extraction(registry, config, environ)
        failures += not ok
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
