import io
import json
import urllib.error

import pytest

from app.config import Settings
from app.structure.detector import StructureDetector
from app.structure.llm import (
    LlmCandidate,
    LlmDecision,
    LlmUnavailableError,
    OpenAICompatibleClassifier,
    classifier_from_settings,
)
from app.structure.models import Severity
from tests.structure_helpers import DocBuilder, numbered_textbook, plain_text_book


class FakeClassifier:
    """Answers from a table keyed by candidate text; records everything it was asked."""

    def __init__(self, answers: dict[str, dict] | None = None, batch_size: int = 15, fail: bool = False) -> None:
        self.answers = answers or {}
        self.batch_size = batch_size
        self.fail = fail
        self.batches: list[list[LlmCandidate]] = []

    def classify(self, candidates: list[LlmCandidate]) -> list[LlmDecision]:
        self.batches.append(candidates)
        if self.fail:
            raise LlmUnavailableError("service is down")
        decisions = []
        for candidate in candidates:
            answer = self.answers.get(candidate.text)
            if answer is not None:
                decisions.append(
                    LlmDecision(candidate_id=candidate.candidate_id, original_text=candidate.text, **answer)
                )
        return decisions

    @property
    def asked(self) -> list[str]:
        return [c.text for batch in self.batches for c in batch]


def settings(**overrides) -> Settings:
    return Settings(database_url="postgresql://u:p@localhost/x", _env_file=None, **overrides)


def titled(result, title):
    return next((n for n in result.nodes if n.title == title), None)


SECTION = {"normalized_title": "Measures of Centre", "classification": "section", "level": 2, "confidence": 0.92, "reasoning": "Short title followed by body text."}


def test_llm_is_not_called_when_deterministic_methods_are_confident() -> None:
    classifier = FakeClassifier()

    StructureDetector(classifier=classifier).detect(numbered_textbook().build())

    assert classifier.batches == []


def test_llm_only_sees_ambiguous_candidates() -> None:
    classifier = FakeClassifier()

    StructureDetector(classifier=classifier).detect(plain_text_book().build())

    assert sorted(classifier.asked) == ["Conditional Probability", "Measures of Centre"]  # not the chapters


def test_llm_classification_is_applied_and_recorded() -> None:
    classifier = FakeClassifier({
        "Measures of Centre": SECTION,
        "Conditional Probability": {**SECTION, "normalized_title": "Conditional Probability", "level": 2},
    })

    result = StructureDetector(classifier=classifier).detect(plain_text_book().build())

    section = titled(result, "Measures of Centre")
    assert section.node_type == "section"
    assert section.confidence == 0.92
    assert section.metadata["method"] == "llm"
    assert section.metadata["llm"]["reasoning"] == "Short title followed by body text."
    assert section.metadata["status"] == "accepted"
    assert "ambiguous_heading" not in {i.code for i in result.issues}


def test_llm_can_say_a_candidate_is_not_a_heading_and_the_text_is_kept() -> None:
    classifier = FakeClassifier({
        "Measures of Centre": {"classification": "not_heading", "confidence": 0.9, "reasoning": "A caption-like line."},
    })

    result = StructureDetector(classifier=classifier).detect(plain_text_book().build())

    assert titled(result, "Measures of Centre") is None
    assert any(p.text == "Measures of Centre" for p in result.paragraphs)
    assert "heading_rejected_by_llm" in {i.code for i in result.issues}


def test_low_confidence_llm_answers_keep_the_warning() -> None:
    classifier = FakeClassifier({"Measures of Centre": {**SECTION, "confidence": 0.5}})

    result = StructureDetector(classifier=classifier).detect(plain_text_book().build())

    node = titled(result, "Measures of Centre")
    assert node.node_type == "section" and node.confidence == 0.5
    assert node.metadata["status"] == "ambiguous"
    assert any(i.entity_id == node.id and i.code == "ambiguous_heading" for i in result.issues)


def test_llm_cannot_invent_headings_or_rewrite_titles() -> None:
    classifier = FakeClassifier({"Measures of Centre": {**SECTION, "normalized_title": "A Completely Different Title"}})

    result = StructureDetector(classifier=classifier).detect(plain_text_book().build())

    assert titled(result, "A Completely Different Title") is None
    assert titled(result, "Measures of Centre") is not None  # the extracted text wins


def test_llm_answers_about_unknown_candidates_or_classes_are_ignored_with_warnings() -> None:
    class Rogue(FakeClassifier):
        def classify(self, candidates):
            return [
                LlmDecision(candidate_id=99, classification="chapter", confidence=0.9),
                LlmDecision(candidate_id=0, classification="banana", confidence=0.9),
            ]

    result = StructureDetector(classifier=Rogue()).detect(plain_text_book().build())

    codes = {i.code for i in result.issues}
    assert {"llm_unknown_candidate", "llm_invalid_classification"} <= codes
    assert titled(result, "Measures of Centre").node_type == "other_heading"


def test_llm_failure_is_reported_and_leaves_deterministic_results_intact() -> None:
    baseline = StructureDetector().detect(plain_text_book().build())

    result = StructureDetector(classifier=FakeClassifier(fail=True)).detect(plain_text_book().build())

    failure = [i for i in result.issues if i.code == "llm_classification_failed"]
    assert failure and failure[0].severity is Severity.WARNING
    assert [(n.node_type, n.title) for n in result.nodes] == [(n.node_type, n.title) for n in baseline.nodes]


def test_candidates_are_sent_in_batches_and_capped() -> None:
    b = DocBuilder()
    b.title("Many").new_page().heading("1. Start").prose(80, 0)
    for i in range(6):
        b.plain(f"Odd Heading {i}", 18.0).prose(80, i + 1)
    classifier = FakeClassifier(batch_size=2)

    result = StructureDetector(classifier=classifier, llm_max_candidates=4).detect(b.build())

    assert [len(batch) for batch in classifier.batches] == [2, 2]
    assert "llm_candidate_limit" in {i.code for i in result.issues}


def test_classifier_request_carries_context_for_the_candidate() -> None:
    classifier = FakeClassifier()

    StructureDetector(classifier=classifier).detect(plain_text_book().build())

    request = next(c for batch in classifier.batches for c in batch if c.text == "Measures of Centre")
    assert request.text_after.startswith("The ")
    assert request.previous_heading == "Describing Data"
    assert request.page == 2
    assert request.signals and request.deterministic_score < 0.7


# -- configuration and the HTTP client ----------------------------------------------------------


def test_llm_is_off_by_default() -> None:
    assert classifier_from_settings(settings()) is None


def test_enabled_llm_requires_a_model_and_a_web_url() -> None:
    with pytest.raises(LlmUnavailableError, match="MODEL"):
        classifier_from_settings(settings(structure_llm_enabled=True))
    with pytest.raises(LlmUnavailableError, match="http"):
        classifier_from_settings(settings(structure_llm_enabled=True, structure_llm_model="m", structure_llm_base_url="file:///etc/passwd"))


def test_enabled_llm_builds_a_client_without_leaking_the_key() -> None:
    client = classifier_from_settings(
        settings(structure_llm_enabled=True, structure_llm_model="gpt-x", structure_llm_api_key="sk-secret")
    )

    assert isinstance(client, OpenAICompatibleClassifier)
    assert client.model == "gpt-x" and client.api_key == "sk-secret"
    assert "sk-secret" not in repr(settings(structure_llm_api_key="sk-secret"))


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def candidate(i=0) -> LlmCandidate:
    return LlmCandidate(candidate_id=i, text="Some Heading", deterministic_score=0.5)


def client() -> OpenAICompatibleClassifier:
    return OpenAICompatibleClassifier(base_url="https://llm.example/v1", model="m", api_key="k")


def test_http_client_parses_a_chat_completion(monkeypatch) -> None:
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.data)
        content = json.dumps({"results": [{"candidate_id": 0, "classification": "section", "confidence": 0.8, "level": 2}]})
        return FakeResponse(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    decisions = client().classify([candidate()])

    assert decisions[0].classification == "section" and decisions[0].level == 2
    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert json.loads(seen["body"]["messages"][1]["content"])["candidates"][0]["text"] == "Some Heading"


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.HTTPError("https://x", 500, "boom", {}, None),
        urllib.error.URLError("no route"),
        TimeoutError("slow"),
    ],
)
def test_http_failures_become_llm_unavailable(monkeypatch, failure) -> None:
    def fake_urlopen(request, timeout):
        raise failure

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(LlmUnavailableError):
        client().classify([candidate()])


def test_malformed_responses_are_rejected(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse(b'{"nothing": true}'))

    with pytest.raises(LlmUnavailableError, match="unexpected"):
        client().classify([candidate()])


def test_individual_malformed_results_are_skipped(monkeypatch) -> None:
    content = json.dumps({"results": [{"candidate_id": 0, "classification": "section", "confidence": 7}, {"candidate_id": 1, "classification": "chapter", "confidence": 0.9}]})
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse(body))

    decisions = client().classify([candidate(0), candidate(1)])

    assert [d.candidate_id for d in decisions] == [1]
