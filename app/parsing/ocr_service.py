"""OCR as a separate, replaceable service.

The rest of the application only depends on `OcrEngine` and `OcrService`. Tesseract is
one implementation; another engine can be added by implementing `OcrEngine`.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import ClassVar

from PIL import Image
from pydantic import BaseModel, Field

from app.parsing.errors import OcrPageError, OcrUnavailableError
from app.parsing.text_normalizer import join_wrapped_lines

logger = logging.getLogger(__name__)


class OcrParagraph(BaseModel):
    text: str
    confidence: float | None = None  # 0-100, mean of word confidences


class OcrPageResult(BaseModel):
    paragraphs: list[OcrParagraph] = Field(default_factory=list)
    mean_confidence: float | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs)


class OcrEngine(ABC):
    name: ClassVar[str]

    @abstractmethod
    def check_available(self, language: str) -> None:
        """Raise OcrUnavailableError, with an actionable message, if the engine can't be used."""

    @abstractmethod
    def recognize(self, image: Image.Image, *, language: str, timeout: int) -> OcrPageResult:
        """OCR one page image. Raise OcrPageError if this page fails."""


class TesseractOcrEngine(OcrEngine):
    name = "tesseract"

    def __init__(self, *, dehyphenate: bool = True) -> None:
        self.dehyphenate = dehyphenate

    def check_available(self, language: str) -> None:
        import pytesseract

        try:
            installed = set(pytesseract.get_languages(config=""))
        except pytesseract.TesseractNotFoundError as exc:
            raise OcrUnavailableError(
                "Tesseract OCR is not installed or not on PATH, but this document contains "
                "scanned pages that need it. Install it (macOS: `brew install tesseract`; "
                "Debian/Ubuntu: `sudo apt install tesseract-ocr`; Windows: the UB Mannheim "
                "installer) and retry, or set OCR_ENABLED=false to skip scanned pages."
            ) from exc
        except (pytesseract.TesseractError, OSError) as exc:
            raise OcrUnavailableError(f"Tesseract could not be started: {exc}") from exc

        missing = [lang for lang in language.split("+") if lang and lang not in installed]
        if missing:
            available = ", ".join(sorted(installed - {"osd"})) or "none"
            raise OcrUnavailableError(
                f"Tesseract language data not installed for: {', '.join(missing)} "
                f"(installed: {available}). Install it (macOS: `brew install tesseract-lang`; "
                "Debian/Ubuntu: `sudo apt install tesseract-ocr-<lang>`) or change OCR_LANGUAGE."
            )

    def recognize(self, image: Image.Image, *, language: str, timeout: int) -> OcrPageResult:
        import pytesseract

        try:
            data = pytesseract.image_to_data(
                image, lang=language, output_type=pytesseract.Output.DICT, timeout=timeout
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise OcrUnavailableError("Tesseract OCR disappeared while processing.") from exc
        except (pytesseract.TesseractError, RuntimeError, OSError) as exc:
            raise OcrPageError(f"Tesseract failed on this page: {exc}") from exc
        return _paragraphs_from_tesseract(data, dehyphenate=self.dehyphenate)


def _paragraphs_from_tesseract(data: dict[str, list], *, dehyphenate: bool = True) -> OcrPageResult:
    """Group Tesseract's word table into paragraphs of unwrapped lines."""
    # (block, paragraph) -> line -> [(word, confidence)]
    grouped: dict[tuple[int, int], dict[int, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, word in enumerate(data["text"]):
        word = str(word).strip()
        confidence = float(data["conf"][index])
        if not word or confidence < 0:
            continue
        key = (int(data["block_num"][index]), int(data["par_num"][index]))
        grouped[key][int(data["line_num"][index])].append((word, confidence))

    paragraphs: list[OcrParagraph] = []
    all_confidences: list[float] = []
    for key in sorted(grouped):
        lines = grouped[key]
        line_texts = [" ".join(word for word, _ in lines[number]) for number in sorted(lines)]
        confidences = [conf for words in lines.values() for _, conf in words]
        text = join_wrapped_lines(line_texts, dehyphenate=dehyphenate)
        if text:
            paragraphs.append(
                OcrParagraph(text=text, confidence=round(sum(confidences) / len(confidences), 1))
            )
            all_confidences.extend(confidences)

    mean = round(sum(all_confidences) / len(all_confidences), 1) if all_confidences else None
    return OcrPageResult(paragraphs=paragraphs, mean_confidence=mean)


class OcrService:
    """Applies one OCR engine with the configured language and timeout."""

    def __init__(self, engine: OcrEngine, *, language: str, timeout_seconds: int) -> None:
        self.engine = engine
        self.language = language
        self.timeout_seconds = timeout_seconds

    @property
    def engine_name(self) -> str:
        return self.engine.name

    def ensure_available(self) -> None:
        self.engine.check_available(self.language)

    def recognize(self, image: Image.Image) -> OcrPageResult:
        return self.engine.recognize(image, language=self.language, timeout=self.timeout_seconds)
