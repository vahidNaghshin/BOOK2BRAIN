import uuid
from pathlib import Path

import pytest

from app.parsing.docx_parser import DocxParser
from app.parsing.models import BlockType, ExtractionStatus
from tests.factories import make_docx_document, make_docx_with_broken_xml


def parse(path: Path):
    return DocxParser().parse(path, document_id=uuid.uuid4(), original_filename=path.name)


@pytest.fixture
def write(tmp_path: Path):
    def _write(data: bytes, name: str = "doc.docx") -> Path:
        path = tmp_path / name
        path.write_bytes(data)
        return path

    return _write


def test_paragraphs_are_extracted_in_order(write) -> None:
    result = parse(
        write(make_docx_document(heading=None, paragraphs=["Alpha paragraph.", "Beta paragraph.", "Gamma paragraph."]))
    )

    assert result.status is ExtractionStatus.SUCCESS
    assert [b.text for b in result.blocks] == ["Alpha paragraph.", "Beta paragraph.", "Gamma paragraph."]
    assert all(b.block_type is BlockType.PARAGRAPH for b in result.blocks)
    assert [b.sequence_number for b in result.blocks] == [0, 1, 2]


def test_headings_are_extracted_with_levels_and_become_section_titles(write) -> None:
    result = parse(write(make_docx_document(heading="Big Heading", paragraphs=["Body text under the heading."])))

    heading, body = result.blocks
    assert heading.block_type is BlockType.HEADING and heading.text == "Big Heading"
    assert heading.metadata == {"heading_level": 1}
    assert body.section_title == "Big Heading"


def test_tables_are_extracted_row_by_row(write) -> None:
    table = [["Name", "Value"], ["Alpha", "42"], ["Beta", "7"]]

    result = parse(write(make_docx_document(heading=None, paragraphs=["Before."], table=table, closing="After.")))

    kinds = [b.block_type for b in result.blocks]
    assert kinds == [BlockType.PARAGRAPH, BlockType.TABLE, BlockType.PARAGRAPH]
    table_block = result.blocks[1]
    assert table_block.text == "Name | Value\nAlpha | 42\nBeta | 7"
    assert table_block.metadata == {"rows": 3, "columns": 2}


def test_list_content_is_extracted_not_discarded(write) -> None:
    result = parse(
        write(make_docx_document(heading=None, paragraphs=["Intro."], bullets=["Bullet one", "Bullet two"], numbered=["Step one"]))
    )

    lists = [b for b in result.blocks if b.block_type is BlockType.LIST]
    assert [b.text for b in lists] == ["Bullet one", "Bullet two", "Step one"]
    assert lists[0].metadata == {"enumerated": False}
    assert lists[2].metadata["enumerated"] is True


def test_document_order_is_preserved_across_block_types(write) -> None:
    result = parse(
        write(
            make_docx_document(
                heading="Title",
                paragraphs=["Para."],
                bullets=["Item"],
                table=[["a", "b"]],
                closing="Closing paragraph.",
            )
        )
    )

    assert [b.block_type.value for b in result.blocks] == ["heading", "paragraph", "list", "table", "paragraph"]
    assert [b.sequence_number for b in result.blocks] == [0, 1, 2, 3, 4]


def test_page_numbers_are_never_invented(write) -> None:
    result = parse(write(make_docx_document(table=[["a", "b"]], bullets=["x"])))

    assert result.page_count is None
    assert all(b.page_number is None for b in result.blocks)
    assert all("page" not in (b.source_location or {}) for b in result.blocks)


def test_source_location_is_kept_for_every_block(write) -> None:
    result = parse(write(make_docx_document(bullets=["Item"])))

    for block in result.blocks:
        assert block.source_location["docling_ref"].startswith("#/")
        assert isinstance(block.source_location["element_index"], int)


def test_result_records_extraction_metadata(write) -> None:
    path = write(make_docx_document(paragraphs=["Some text content here."]), "named.docx")

    result = parse(path)

    assert result.file_format == "docx" and result.extraction_method == "docling"
    assert result.original_filename == "named.docx"
    assert result.character_count == sum(len(b.text) for b in result.blocks) > 0
    assert result.extracted_at is not None and result.errors == []


def test_empty_document_fails_gracefully(write) -> None:
    result = parse(write(make_docx_document(heading=None, paragraphs=[])))

    assert result.status is ExtractionStatus.FAILED
    assert [e.code for e in result.errors] == ["empty_document"]
    assert result.blocks == []


@pytest.mark.parametrize(
    "data",
    [b"PK\x03\x04 this is not really a docx", b"plain text, not even a zip", make_docx_with_broken_xml()],
    ids=["fake-zip", "not-a-zip", "broken-xml"],
)
def test_malformed_documents_fail_gracefully_without_raising(write, data: bytes) -> None:
    result = parse(write(data))

    assert result.status is ExtractionStatus.FAILED
    assert result.errors and result.errors[0].code in {"corrupt_document", "empty_document"}
    assert result.blocks == []


def test_progress_is_reported(write) -> None:
    events: list[tuple[str, int, int]] = []

    DocxParser().parse(
        write(make_docx_document()),
        document_id=uuid.uuid4(),
        original_filename="x.docx",
        on_progress=lambda stage, cur, total: events.append((stage.value, cur, total)),
    )

    assert events[0] == ("parsing", 0, 1) and events[-1] == ("parsing", 1, 1)
