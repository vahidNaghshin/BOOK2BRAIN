"""Integrity validation: relationships between the saved records, read back from the database."""

import io
import uuid

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from app.config import Settings
from app.schemas import BookMetadata
from app.services.document_service import DocumentService
from app.services.integrity_service import IntegrityValidator
from app.services.pipeline_service import ProcessingPipeline
from app.structure.models import Severity
from database.models import (
    ExtractedBlock,
    Paragraph,
    Passage,
    PassageParagraph,
    ProcessingRun,
    StructuralNode,
    StructureAnalysis,
)
from tests.factories import make_pdf
from tests.fakes import StubParser, StubParserFactory
from tests.structure_helpers import content_blocks, numbered_textbook


@pytest.fixture
def settings(test_database_url, tmp_path) -> Settings:
    return Settings(
        database_url=test_database_url, storage_root=tmp_path / "storage", max_upload_size_mb=5,
        equation_extraction_enabled=False, _env_file=None,
    )


def processed(session: Session, service: DocumentService, storage, settings, seed: str):
    document = service.upload(
        file=io.BytesIO(make_pdf(seed)), filename=f"{seed}.pdf", metadata=BookMetadata(title=f"Book {seed}")
    ).document
    blocks = numbered_textbook().build()
    factory = StubParserFactory(StubParser(content_blocks(blocks), page_count=blocks.page_count))
    run = ProcessingPipeline(session, storage, settings, factory).process_document(document.id)
    assert run.status == "completed"
    return document, run.id


def codes(session: Session, storage, document, run_id=None, severity=None) -> set[str]:
    session.commit()
    issues = IntegrityValidator(session, storage).validate(document.id, run_id)
    return {i.code for i in issues if severity is None or i.severity is severity}


@pytest.fixture
def book(session, document_service, storage, settings):
    return processed(session, document_service, storage, settings, "i1")


def test_a_correctly_processed_document_has_no_errors(session, storage, book) -> None:
    document, run_id = book

    assert codes(session, storage, document, run_id, Severity.ERROR) == set()
    assert codes(session, storage, document, run_id, Severity.WARNING) == set()


def test_the_run_must_belong_to_the_document(session, document_service, storage, settings, book) -> None:
    document, _ = book
    _, other_run = processed(session, document_service, storage, settings, "i2")

    assert "run_wrong_document" in codes(session, storage, document, other_run)


def test_the_original_file_must_still_be_stored(session, storage, book) -> None:
    document, run_id = book
    storage.delete(document.file_path)

    assert "source_file_missing" in codes(session, storage, document, run_id)


def test_extracted_text_must_exist(session, storage, book) -> None:
    document, run_id = book
    session.execute(delete(ExtractedBlock).where(ExtractedBlock.document_id == document.id))

    assert "no_extracted_text" in codes(session, storage, document, run_id)


def test_block_pages_must_be_within_the_document(session, storage, book) -> None:
    document, run_id = book
    session.execute(update(ExtractedBlock).where(ExtractedBlock.document_id == document.id, ExtractedBlock.sequence_number == 3).values(page_number=500))

    assert "block_page_out_of_range" in codes(session, storage, document, run_id)


def test_node_relationships_are_checked(session, storage, book) -> None:
    document, run_id = book
    section = session.scalar(select(StructuralNode).where(StructuralNode.document_id == document.id, StructuralNode.title == "Background"))
    session.execute(update(StructuralNode).where(StructuralNode.id == section.id).values(level=7))

    assert "level_inconsistent" in codes(session, storage, document, run_id, Severity.ERROR)


def test_nodes_may_not_have_a_parent_from_another_document(session, document_service, storage, settings, book) -> None:
    document, run_id = book
    other, _ = processed(session, document_service, storage, settings, "i3")
    foreign = session.scalar(select(StructuralNode).where(StructuralNode.document_id == other.id, StructuralNode.title == "Models"))
    section = session.scalar(select(StructuralNode).where(StructuralNode.document_id == document.id, StructuralNode.title == "Background"))
    session.execute(update(StructuralNode).where(StructuralNode.id == section.id).values(parent_id=foreign.id))

    assert "parent_in_other_document" in codes(session, storage, document, run_id)


def test_node_page_references_are_checked(session, storage, book) -> None:
    document, run_id = book
    session.execute(update(StructuralNode).where(StructuralNode.document_id == document.id, StructuralNode.title == "Models").values(page_end=999))

    assert "node_page_out_of_range" in codes(session, storage, document, run_id)


def test_duplicate_nodes_are_flagged(session, storage, book) -> None:
    document, run_id = book
    original = session.scalar(select(StructuralNode).where(StructuralNode.document_id == document.id, StructuralNode.title == "Background"))
    session.add(
        StructuralNode(
            book_id=document.book_id, document_id=document.id, parent_id=original.parent_id, node_type=original.node_type,
            title=original.title, number=original.number, level=original.level, sequence=999,
        )
    )

    assert "duplicate_nodes" in codes(session, storage, document, run_id, Severity.WARNING)
    assert "node_sequence_gap" in codes(session, storage, document, run_id, Severity.ERROR)


def test_passages_must_belong_to_nodes_of_the_same_document(session, document_service, storage, settings, book) -> None:
    document, run_id = book
    other, _ = processed(session, document_service, storage, settings, "i4")
    foreign_node = session.scalar(select(StructuralNode).where(StructuralNode.document_id == other.id, StructuralNode.title == "Models"))
    passage = session.scalar(select(Passage).where(Passage.document_id == document.id).limit(1))
    session.execute(update(Passage).where(Passage.id == passage.id).values(structural_node_id=foreign_node.id, sequence=99))

    assert "passage_in_other_document" in codes(session, storage, document, run_id)


def test_passage_text_tokens_and_order_are_checked(session, storage, book) -> None:
    document, run_id = book
    first, second = session.scalars(select(Passage).where(Passage.document_id == document.id).order_by(Passage.document_sequence).limit(2))
    session.execute(update(Passage).where(Passage.id == first.id).values(text="   "))
    session.execute(update(Passage).where(Passage.id == second.id).values(token_count=0, document_sequence=500))

    found = codes(session, storage, document, run_id, Severity.ERROR)

    assert {"empty_passage", "passage_without_tokens", "passage_order_gap"} <= found


def test_passage_sequence_inside_a_section_must_be_consecutive(session, storage, book) -> None:
    document, run_id = book
    passage = session.scalar(select(Passage).where(Passage.document_id == document.id).limit(1))
    session.execute(update(Passage).where(Passage.id == passage.id).values(sequence=5))

    assert "passage_sequence_gap" in codes(session, storage, document, run_id)


def test_passage_pages_must_be_within_the_document(session, storage, book) -> None:
    document, run_id = book
    passage = session.scalar(select(Passage).where(Passage.document_id == document.id).limit(1))
    session.execute(update(Passage).where(Passage.id == passage.id).values(page_start=1, page_end=400))

    assert "passage_page_out_of_range" in codes(session, storage, document, run_id)


def test_passages_must_link_to_paragraphs_of_their_own_section(session, storage, book) -> None:
    document, run_id = book
    passage = session.scalar(select(Passage).where(Passage.document_id == document.id).limit(1))
    stray = session.scalar(select(Paragraph).where(Paragraph.document_id == document.id, Paragraph.structural_node_id != passage.structural_node_id).limit(1))
    session.execute(update(PassageParagraph).where(PassageParagraph.passage_id == passage.id, PassageParagraph.position == 0).values(paragraph_id=stray.id))
    unlinked = session.scalar(select(Passage).where(Passage.document_id == document.id, Passage.id != passage.id).limit(1))
    session.execute(delete(PassageParagraph).where(PassageParagraph.passage_id == unlinked.id))

    found = codes(session, storage, document, run_id)

    assert {"passage_paragraph_mismatch", "passage_without_paragraphs"} <= found


def test_text_without_passages_is_an_error(session, storage, book) -> None:
    document, run_id = book
    node_id = session.scalar(select(Passage.structural_node_id).where(Passage.document_id == document.id).limit(1))
    session.execute(delete(Passage).where(Passage.structural_node_id == node_id))

    assert "text_without_passages" in codes(session, storage, document, run_id)


def test_the_structure_steps_own_errors_fail_validation(session, storage, book) -> None:
    document, run_id = book
    analysis = session.scalar(select(StructureAnalysis).where(StructureAnalysis.document_id == document.id))
    analysis.structure_issues = [*(analysis.structure_issues or []), {"severity": "error", "code": "content_missing", "message": "x"}]

    assert "structure_validation_failed" in codes(session, storage, document, run_id)


def test_missing_structure_analysis_is_an_error(session, storage, book) -> None:
    document, run_id = book
    session.execute(delete(StructureAnalysis).where(StructureAnalysis.document_id == document.id))

    assert "structure_not_completed" in codes(session, storage, document, run_id)


def test_counts_of_different_kinds_are_not_expected_to_match(session, storage, book) -> None:
    document, run_id = book
    blocks = session.scalar(select(func.count()).select_from(ExtractedBlock).where(ExtractedBlock.document_id == document.id))
    passages = session.scalar(select(func.count()).select_from(Passage).where(Passage.document_id == document.id))

    assert blocks != passages
    assert codes(session, storage, document, run_id, Severity.ERROR) == set()


def test_unknown_documents_are_reported(session, storage) -> None:
    issues = IntegrityValidator(session, storage).validate(uuid.uuid4())

    assert [i.code for i in issues] == ["document_missing"]


def test_validation_works_without_a_run(session, storage, book) -> None:
    document, _ = book

    assert codes(session, storage, document, None, Severity.ERROR) == set()
    assert session.get(ProcessingRun, book[1]) is not None
