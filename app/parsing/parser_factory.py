from app.config import Settings
from app.parsing.base import DocumentParser
from app.parsing.docx_parser import DocxParser
from app.parsing.epub_parser import EpubParser
from app.parsing.errors import UnsupportedDocumentError
from app.parsing.ocr_service import OcrEngine, OcrService, TesseractOcrEngine
from app.parsing.pdf_parser import PdfParser


class ParserFactory:
    """Picks the parser for a file format and wires in OCR according to the settings."""

    def __init__(self, settings: Settings, ocr_engine: OcrEngine | None = None) -> None:
        self.settings = settings
        self._ocr_engine = ocr_engine

    def _ocr_service(self) -> OcrService | None:
        if not self.settings.ocr_enabled:
            return None
        engine = self._ocr_engine or TesseractOcrEngine(dehyphenate=self.settings.ocr_dehyphenate)
        return OcrService(
            engine,
            language=self.settings.ocr_language,
            timeout_seconds=self.settings.ocr_page_timeout_seconds,
        )

    def get_parser(self, file_format: str) -> DocumentParser:
        if file_format == "pdf":
            return PdfParser(self.settings, self._ocr_service())
        if file_format == "epub":
            return EpubParser()
        if file_format == "docx":
            return DocxParser()
        raise UnsupportedDocumentError(f"No parser is available for '{file_format}' files.")
