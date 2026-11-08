import shutil

import pytest
from PIL import Image

from app.parsing import ocr_service
from app.parsing.errors import OcrPageError, OcrUnavailableError
from app.parsing.ocr_service import (
    OcrService,
    TesseractOcrEngine,
    _paragraphs_from_tesseract,
)
from tests.factories import _page_image
from tests.fakes import FakeOcrEngine, ocr_result


def word_table(words: list[tuple[str, float, int, int, int]]) -> dict[str, list]:
    """Build a Tesseract-style word table from (text, conf, block, paragraph, line) tuples."""
    return {
        "text": [w[0] for w in words],
        "conf": [w[1] for w in words],
        "block_num": [w[2] for w in words],
        "par_num": [w[3] for w in words],
        "line_num": [w[4] for w in words],
    }


def test_word_table_is_grouped_into_paragraphs_of_unwrapped_lines() -> None:
    data = word_table(
        [
            ("", -1, 1, 0, 0),  # structural row, ignored
            ("The", 96.0, 1, 1, 1),
            ("quick", 94.0, 1, 1, 1),
            ("brown", 90.0, 1, 1, 2),  # wrapped second line of the same paragraph
            ("fox.", 92.0, 1, 1, 2),
            ("Second", 80.0, 2, 1, 1),
            ("paragraph", 80.0, 2, 1, 1),
        ]
    )

    result = _paragraphs_from_tesseract(data)

    assert [p.text for p in result.paragraphs] == ["The quick brown fox.", "Second paragraph"]
    assert result.paragraphs[0].confidence == 93.0
    assert result.mean_confidence == pytest.approx(88.7, abs=0.1)
    assert result.text == "The quick brown fox.\n\nSecond paragraph"


def test_word_table_repairs_hyphenated_line_breaks_when_enabled() -> None:
    data = word_table([("an", 90, 1, 1, 1), ("exam-", 90, 1, 1, 1), ("ple", 90, 1, 1, 2)])

    assert _paragraphs_from_tesseract(data).text == "an example"
    assert _paragraphs_from_tesseract(data, dehyphenate=False).text == "an exam- ple"


def test_empty_word_table_gives_empty_result() -> None:
    result = _paragraphs_from_tesseract(word_table([("", -1, 1, 0, 0), ("  ", 50, 1, 1, 1)]))

    assert result.paragraphs == [] and result.mean_confidence is None


def test_service_delegates_to_the_engine_with_configured_language() -> None:
    engine = FakeOcrEngine([ocr_result("hello")])
    service = OcrService(engine, language="eng+deu", timeout_seconds=30)

    result = service.recognize(Image.new("RGB", (10, 10)))

    assert result.text == "hello"
    assert service.engine_name == "fake" and service.language == "eng+deu"
    assert engine.calls == 1


def test_another_engine_can_be_plugged_in_without_changing_the_service() -> None:
    class UpperEngine(FakeOcrEngine):
        name = "upper"

        def recognize(self, image, *, language, timeout):  # type: ignore[no-untyped-def]
            return ocr_result(f"{language}:ok")

    assert OcrService(UpperEngine(), language="fra", timeout_seconds=5).recognize(
        Image.new("RGB", (1, 1))
    ).text == "fra:ok"


def test_missing_tesseract_gives_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import pytesseract

    def not_found(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "get_languages", not_found)

    with pytest.raises(OcrUnavailableError) as exc_info:
        TesseractOcrEngine().check_available("eng")

    message = str(exc_info.value)
    assert "not installed" in message and "brew install tesseract" in message
    assert "apt install tesseract-ocr" in message and "OCR_ENABLED" in message


def test_missing_language_data_gives_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import pytesseract

    monkeypatch.setattr(pytesseract, "get_languages", lambda config="": ["eng", "osd"])

    TesseractOcrEngine().check_available("eng")
    with pytest.raises(OcrUnavailableError) as exc_info:
        TesseractOcrEngine().check_available("eng+deu")

    assert "deu" in str(exc_info.value) and "OCR_LANGUAGE" in str(exc_info.value)


def test_page_failure_is_reported_as_page_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import pytesseract

    def timeout(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(pytesseract, "image_to_data", timeout)

    with pytest.raises(OcrPageError, match="timeout"):
        TesseractOcrEngine().recognize(Image.new("RGB", (5, 5)), language="eng", timeout=1)


def test_tesseract_disappearing_mid_run_is_reported_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pytesseract

    def gone(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_data", gone)

    with pytest.raises(OcrUnavailableError):
        TesseractOcrEngine().recognize(Image.new("RGB", (5, 5)), language="eng", timeout=1)


@pytest.mark.ocr
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract is not installed")
def test_real_tesseract_reads_a_rendered_page() -> None:
    """Optional integration test: runs only when Tesseract is installed."""
    import io

    image = Image.open(io.BytesIO(_page_image(["Chapter Two: The Scan", "", "Tesseract reads this."])))
    service = OcrService(TesseractOcrEngine(), language="eng", timeout_seconds=60)

    service.ensure_available()
    result = service.recognize(image)

    assert "Chapter Two" in result.text and "Tesseract reads this" in result.text
    assert result.mean_confidence is not None and result.mean_confidence > 60
    assert ocr_service.TesseractOcrEngine.name == "tesseract"
