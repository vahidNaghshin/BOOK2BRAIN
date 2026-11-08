"""Shared Docling converters. Imported lazily: Docling pulls in torch and is slow to import."""

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_pdf_converter() -> Any:
    """A PDF converter with Docling's own OCR off: OCR is a separate, replaceable service."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


@lru_cache(maxsize=1)
def get_docx_converter() -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter

    return DocumentConverter(allowed_formats=[InputFormat.DOCX])
