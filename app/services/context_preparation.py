"""Prepares what the model is shown for one passage: the passage, its neighbours and where it comes from.

The primary passage is always sent whole. Neighbouring passages are reduced first, furthest first,
when the prompt would not fit the selected model's context window, and the reduction is recorded
(ContextInfo) so it can be seen later. Nothing is ever truncated silently: if the passage alone does
not fit, the request is returned unchanged and marked as not fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.extraction.context import describe, fit_context
from app.extraction.model_config import LlmModelConfig
from app.extraction.prompts import get_prompt
from app.extraction.results import ContextInfo, ExtractionRequest, PassageInput, SourceInfo
from app.extraction.settings import ExtractionSettings
from app.knowledge.types import KnowledgeType
from app.repositories.passage_repository import PassageRepository
from database.models import Book, Passage


@dataclass(frozen=True)
class PreparedContext:
    request: ExtractionRequest
    info: ContextInfo
    fits: bool


def source_info(book: Book) -> SourceInfo:
    return SourceInfo(
        title=book.title,
        subtitle=book.subtitle,
        author=book.author,
        publisher=book.publisher,
        year=book.publication_year,
        language=book.language,
    )


class ContextPreparer:
    def __init__(self, passages: PassageRepository, settings: ExtractionSettings, *, neighbours: int | None = None) -> None:
        self.passages = passages
        self.settings = settings
        self.neighbours = settings.context_passages if neighbours is None else neighbours

    def passage_input(self, passage: Passage) -> PassageInput:
        located = self.passages.source_of(passage)
        return PassageInput(
            id=passage.id,
            text=passage.text,
            page_start=passage.page_start,
            page_end=passage.page_end,
            chapter_title=located.chapter.title if located.chapter else None,
            section_title=located.section.title if located.section else None,
        )

    def prepare(
        self, passage: Passage, category: KnowledgeType, config: LlmModelConfig, source: SourceInfo | None = None
    ) -> PreparedContext:
        before, after = self.passages.neighbours(passage, self.neighbours)
        request = ExtractionRequest(
            knowledge_type=category,
            passage=self.passage_input(passage),
            context_before=[self.passage_input(p) for p in before],
            context_after=[self.passage_input(p) for p in after],
            source=source,
        )
        try:
            prompt = get_prompt(category, self.settings.prompt_versions.get(category.value))
        except KeyError:  # the extraction service reports an unknown prompt version itself
            info = ContextInfo(requested=len(before) + len(after), sent=len(before) + len(after))
            return PreparedContext(request.model_copy(update={"context": info}), info, True)
        fit = fit_context(
            request,
            prompt,
            config,
            chars_per_token=self.settings.chars_per_token,
            safety_margin=self.settings.context_safety_margin,
        )
        info = describe(request, fit)
        return PreparedContext(fit.request.model_copy(update={"context": info}), info, fit.fits)
