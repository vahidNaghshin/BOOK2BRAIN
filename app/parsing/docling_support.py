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


@lru_cache(maxsize=2)
def get_formula_converter(device: str, dtype: str) -> Any:
    """The same PDF pipeline as `get_pdf_converter`, plus Docling's formula enrichment.

    Everything else (layout, tables) is unchanged. Docling's default device/dtype for the formula
    model was ~13x slower than cpu/float32 in testing, so both are explicit and configurable.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        do_formula_enrichment=True,
        accelerator_options=AcceleratorOptions(device=device),
    )
    for config in options.code_formula_options.model_spec.engine_overrides.values():
        config.extra_config["torch_dtype"] = dtype
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


@lru_cache(maxsize=1)
def get_docx_converter() -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter

    return DocumentConverter(allowed_formats=[InputFormat.DOCX])
