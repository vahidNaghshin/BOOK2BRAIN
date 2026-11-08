"""Test doubles for OCR and parsers."""

from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from app.parsing.base import DocumentParser, Extraction, ProgressCallback
from app.parsing.equation.equation_ocr import (
    MathOcrEngine,
    MathOcrError,
    MathOcrResult,
    MathOcrService,
    MathOcrUnavailableError,
)
from app.parsing.errors import OcrPageError, OcrUnavailableError, ParserError
from app.parsing.models import ContentBlock, ParseStage
from app.parsing.ocr_service import OcrEngine, OcrPageResult, OcrParagraph


class FakeMathEngine(MathOcrEngine):
    """Scripted math OCR: each call takes the next outcome (a LaTeX string, MathOcrResult or error)."""

    name = "fake"

    def __init__(self, outcomes: Sequence[Any] = (), *, unavailable: str | None = None, default: str = "x = 1") -> None:
        self.outcomes = list(outcomes)
        self.unavailable = unavailable
        self.default = default
        self.calls = 0

    def check_available(self) -> None:
        if self.unavailable:
            raise MathOcrUnavailableError(self.unavailable)

    def recognize(self, image: Image.Image) -> MathOcrResult:
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else self.default
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if isinstance(outcome, MathOcrResult) else MathOcrResult(latex=outcome)


def math_service(*outcomes: Any, **kwargs: Any) -> MathOcrService:
    return MathOcrService(FakeMathEngine(outcomes, **kwargs))


def formula_item(page: int, box: tuple[float, float, float, float], text: str, orig: str) -> Any:
    """A stand-in for a Docling formula item. `box` is (l, t, r, b) in BOTTOMLEFT coordinates."""
    left, top, right, bottom = box
    return SimpleNamespace(
        label=SimpleNamespace(value="formula"),
        text=text,
        orig=orig,
        prov=[SimpleNamespace(page_no=page, bbox=SimpleNamespace(l=left, t=top, r=right, b=bottom))],
    )


class FakeFormulaConverter:
    """Docling converter double for formula enrichment. Records the page ranges it was asked for."""

    def __init__(self, items: Sequence[Any] = (), error: Exception | None = None) -> None:
        self.items = list(items)
        self.error = error
        self.ranges: list[tuple[int, int]] = []

    def factory(self, device: str, dtype: str) -> "FakeFormulaConverter":
        return self

    def convert(self, path: str, page_range: tuple[int, int]) -> Any:
        self.ranges.append(page_range)
        if self.error:
            raise self.error
        first, last = page_range
        wanted = [item for item in self.items if first <= item.prov[0].page_no <= last]
        return SimpleNamespace(document=SimpleNamespace(iterate_items=lambda: [(item, 0) for item in wanted]))


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
