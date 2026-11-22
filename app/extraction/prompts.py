"""Versioned prompt templates: lookup and rendering.

Wording lives in app.extraction.prompt_texts. A template is identified by name and version
(`concept_extraction@1.0.0`); results record both, so an extraction can always be traced to the exact
prompt that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.extraction import prompt_texts as texts
from app.extraction.results import ExtractionRequest, PassageInput, SourceInfo, ValidationIssue
from app.knowledge.content import CONTENT_MODELS
from app.knowledge.types import KnowledgeType

_MAX_REPORTED_ISSUES = 25


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    knowledge_type: KnowledgeType
    purpose: str
    identify: tuple[str, ...]
    skeleton: dict[str, Any]
    missing_information: str
    terminology: str
    rules: str = texts.GENERAL_RULES
    user_intro: str = texts.USER_INTRO_1_0
    include_source: bool = False

    @property
    def identifier(self) -> str:
        return f"{self.name}@{self.version}"

    def render_system(self) -> str:
        identify = "\n".join(f"- {line}" for line in self.identify)
        return (
            f"{self.rules}\n"
            f"TASK ({self.knowledge_type.value}): {self.purpose}\n\n"
            f"Identify:\n{identify}\n\n"
            f"{texts.ENVELOPE}\n"
            f'The "content" object for this category:\n{json.dumps(self.skeleton, indent=2, ensure_ascii=False)}\n\n'
            f"Missing information and incomplete items: {self.missing_information}\n"
            f"Terminology: {self.terminology}\n"
        )

    def render_user(self, request: ExtractionRequest, *, context_before=None, context_after=None) -> str:
        before = request.context_before if context_before is None else context_before
        after = request.context_after if context_after is None else context_after
        parts = [_passage_tag(p, "context") for p in before]
        parts.append(_passage_tag(request.passage, "target"))
        parts.extend(_passage_tag(p, "context") for p in after)
        intro = self.user_intro.format(category=self.knowledge_type.value)
        if self.include_source and request.source is not None:
            intro = f"{texts.SOURCE_LINE.format(book=_describe_book(request.source))}\n\n{intro}"
        return intro + "\n\n" + "\n\n".join(parts)

    def output_schema(self) -> dict[str, Any]:
        """JSON schema of the full response, for providers that enforce structured output."""
        content = CONTENT_MODELS[self.knowledge_type].model_json_schema()
        definitions = content.pop("$defs", {})
        item = {
            "type": "object",
            "properties": {
                "content": content,
                "sources": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "passage_id": {"type": "string"},
                            "quote": {"type": ["string", "null"]},
                            "role": {"type": "string", "enum": ["primary", "supporting"]},
                        },
                        "required": ["passage_id"],
                    },
                },
                "page_start": {"type": ["integer", "null"]},
                "page_end": {"type": ["integer", "null"]},
                "basis": {"type": "string", "enum": ["explicit", "inferred", "mixed"]},
                "confidence": {"type": ["number", "null"]},
                "is_uncertain": {"type": "boolean"},
                "uncertainty_notes": {"type": ["string", "null"]},
            },
            "required": ["content", "sources", "basis"],
        }
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": item}, "notes": {"type": ["string", "null"]}},
            "required": ["items"],
        }
        if definitions:
            schema["$defs"] = definitions
        return schema


def _passage_tag(passage: PassageInput, role: str) -> str:
    attributes = [f'id="{passage.id}"', f'role="{role}"']
    if passage.page_start is not None:
        last = passage.last_page
        attributes.append(f'pages="{passage.page_start}"' if last == passage.page_start else f'pages="{passage.page_start}-{last}"')
    else:
        attributes.append('pages="unknown"')
    if passage.chapter_title:
        attributes.append(f'chapter="{_attr(passage.chapter_title)}"')
    if passage.section_title:
        attributes.append(f'section="{_attr(passage.section_title)}"')
    return f"<passage {' '.join(attributes)}>\n{passage.text}\n</passage>"


def _describe_book(source: SourceInfo) -> str:
    text = f'"{_attr(source.title)}"'
    if source.subtitle:
        text += f" ({_attr(source.subtitle)})"
    if source.author:
        text += f" by {_attr(source.author)}"
    details = ", ".join(str(v) for v in (source.publisher, source.year) if v)
    return f"{text} [{_attr(details)}]" if details else text


def _attr(value: str) -> str:
    return value.replace('"', "'").replace("<", "(").replace(">", ")")


def render_correction(issues: list[ValidationIssue], *, truncated: bool = False) -> str:
    """The follow-up message that asks the model to fix specific problems."""
    errors = [i for i in issues if i.severity == "error"] or issues
    lines = [
        f"- {i.message}" + (f" (at {i.path})" if i.path else "") for i in errors[:_MAX_REPORTED_ISSUES]
    ]
    if len(errors) > _MAX_REPORTED_ISSUES:
        lines.append(f"- ... and {len(errors) - _MAX_REPORTED_ISSUES} more problems")
    text = f"{texts.CORRECTION_INTRO}\n\nProblems:\n" + "\n".join(lines)
    if truncated:
        text += "\n\nYour answer was cut off because it was too long. Return fewer or shorter items."
    return text


def _build_registry() -> dict[KnowledgeType, dict[str, PromptTemplate]]:
    registry: dict[KnowledgeType, dict[str, PromptTemplate]] = {}
    for key, spec in texts.CATEGORIES.items():
        knowledge_type = KnowledgeType(key)
        common = {
            "knowledge_type": knowledge_type,
            "purpose": spec["purpose"],
            "identify": tuple(spec["identify"]),
            "skeleton": spec["skeleton"],
            "missing_information": spec["missing"],
            "terminology": spec["terminology"],
        }
        registry[knowledge_type] = {
            texts.FIRST_VERSION: PromptTemplate(name=f"{key}_extraction", version=texts.FIRST_VERSION, **common),
            texts.SECOND_VERSION: PromptTemplate(
                name=f"{key}_extraction",
                version=texts.SECOND_VERSION,
                rules=texts.GENERAL_RULES + texts.CONTEXT_RULES,
                user_intro=texts.USER_INTRO_1_1,
                **common,
            ),
            texts.PROMPT_VERSION: PromptTemplate(
                name=f"{key}_extraction",
                version=texts.PROMPT_VERSION,
                rules=texts.GENERAL_RULES + texts.CONTEXT_RULES,
                user_intro=texts.USER_INTRO_1_1,
                include_source=True,
                **common,
            ),
        }
    return registry


PROMPTS = _build_registry()


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) if part.isdigit() else 0 for part in version.split("."))


def get_prompt(knowledge_type: KnowledgeType | str, version: str | None = None) -> PromptTemplate:
    """The template for a category: the given version, or the latest one."""
    versions = PROMPTS[KnowledgeType(knowledge_type)]
    if version is None:
        return versions[max(versions, key=_version_key)]
    try:
        return versions[version]
    except KeyError:
        known = ", ".join(sorted(versions, key=_version_key))
        raise KeyError(f"No {knowledge_type} prompt version '{version}' (available: {known})") from None
