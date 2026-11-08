"""Test doubles for OCR and parsers."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from app.parsing.base import DocumentParser, Extraction, ProgressCallback
from app.parsing.errors import OcrPageError, OcrUnavailableError, ParserError
from app.parsing.models import ContentBlock, ParseStage
from app.parsing.ocr_service import OcrEngine, OcrPageResult, OcrParagraph


def ocr_result(*paragraphs: str, confidence: float = 90.0) -> OcrPageResult:
    return OcrPageResult(
        paragraphs=[OcrParagraph(text=text, confidence=confidence) for text in paragraphs],
        mean_confidence=confidence if paragraphs else None,
    )


class FakeOcrEngine(OcrEngine):
    """Returns queued results (or raises queued exceptions) one per page, in order."""

    name = "fake"

    def __init__(
        self,
        results: list[OcrPageResult | Exception] | None = None,
        *,
        unavailable: str | None = None,
    ) -> None:
        self.results = list(results or [])
        self.unavailable = unavailable
        self.calls = 0
        self.images: list[Image.Image] = []

    def check_available(self, language: str) -> None:
        if self.unavailable:
            raise OcrUnavailableError(self.unavailable)

    def recognize(self, image: Image.Image, *, language: str, timeout: int) -> OcrPageResult:
        self.calls += 1
        self.images.append(image)
        outcome = self.results.pop(0) if self.results else ocr_result()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class StubParser(DocumentParser):
    """A parser with scripted behaviour, for testing the processing workflow."""

    file_format = "pdf"
    extraction_method = "stub"

    def __init__(
        self,
        blocks: list[ContentBlock] | None = None,
        *,
        page_count: int | None = None,
        error: ParserError | None = None,
        exception: BaseException | None = None,
        observer: Callable[[Path], None] | None = None,
        stages: list[tuple[ParseStage, int, int]] | None = None,
    ) -> None:
        self.blocks = blocks if blocks is not None else []
        self.page_count = page_count
        self.error = error
        self.exception = exception
        self.observer = observer
        self.stages = stages or []

    def _extract(self, path: Path, on_progress: ProgressCallback) -> Extraction:
        if self.observer:
            self.observer(path)
        for stage, current, total in self.stages:
            on_progress(stage, current, total)
        if self.exception:
            raise self.exception
        if self.error:
            raise self.error
        return Extraction(
            blocks=[block.model_copy() for block in self.blocks], page_count=self.page_count
        )


class StubParserFactory:
    def __init__(self, parser: DocumentParser) -> None:
        self.parser = parser
        self.requested: list[str] = []

    def get_parser(self, file_format: str) -> DocumentParser:
        self.requested.append(file_format)
        return self.parser


def page_error(message: str = "boom") -> OcrPageError:
    return OcrPageError(message)


def block(text: str, **kwargs: Any) -> ContentBlock:
    return ContentBlock(text=text, **kwargs)
