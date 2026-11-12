import sys
import uuid

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.services.errors import (
    DocumentNotFoundError,
    StructureNotAllowedError,
    StructureServiceError,
)
from app.services.structure_service import StructureService
from app.structure.segmentation import SegmentationConfig
from database.models import (
    Book,
    Document,
    Equation,
    ExtractedBlock,
    Paragraph,
    Passage,
    PassageParagraph,
    ProcessingRun,
    ProcessingRunStatus,
    ProcessingStatus,
    StructuralNode,
    StructureAnalysis,
)
from tests.structure_helpers import DocBuilder, numbered_textbook, persist_document, plain_text_book
from tests.test_structure_llm import FakeClassifier


def service(session: Session, settings: Settings, **kwargs) -> StructureService:
    return StructureService(session, settings, **kwargs)


def count(session: Session, model) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


@pytest.fixture
def document(session: Session) -> Document:
    return persist_document(session, numbered_textbook().build())


def snapshot(session: Session, document_id) -> dict:
    nodes = session.execute(
        select(StructuralNode.sequence, StructuralNode.node_type, StructuralNode.title, StructuralNode.level)
        .where(StructuralNode.document_id == document_id)
        .order_by(StructuralNode.sequence)
    ).all()
    return {
        "nodes": [tuple(row) for row in nodes],
        "paragraphs": count(session, Paragraph),
        "passages": count(session, Passage),
    }


# -- detection and persistence -----------------------------------------------------------------


def test_detect_structure_saves_the_hierarchy_paragraphs_and_passages(
    session: Session, settings: Settings, document: Document
) -> None:
    analysis = service(session, settings).detect_structure(document.id)

    assert analysis.structure_status == "completed" and analysis.passage_status == "completed"
    nodes = list(
        session.scalars(
            select(StructuralNode).where(StructuralNode.document_id == document.id).order_by(StructuralNode.sequence)
        )
    )
    assert [n.sequence for n in nodes] == list(range(len(nodes)))
    assert all(n.book_id == document.book_id and n.document_id == document.id for n in nodes)
    by_id = {n.id: n for n in nodes}
    for node in nodes:
        if node.parent_id:
            assert node.level == by_id[node.parent_id].level + 1
    chapter = next(n for n in nodes if n.title == "Models")
    assert (chapter.node_type, chapter.number, chapter.original_title) == ("chapter", "2", "2. Models")
    assert (chapter.page_start, chapter.page_end) == (6, 6)
    assert chapter.confidence and chapter.confidence > 0.9
    assert chapter.created_at and chapter.updated_at

    paragraphs = list(session.scalars(select(Paragraph).where(Paragraph.document_id == document.id)))
    assert paragraphs and all(p.structural_node_id in by_id and p.book_id == document.book_id for p in paragraphs)
    passages = list(session.scalars(select(Passage).where(Passage.document_id == document.id)))
    assert passages and all(p.structural_node_id in by_id and p.token_count > 0 for p in passages)
    paragraph_ids = {p.id for p in paragraphs}
    links = list(session.scalars(select(PassageParagraph)))
    assert links and all(link.paragraph_id in paragraph_ids for link in links)


def test_analysis_record_holds_summary_and_findings(session: Session, settings: Settings, document: Document) -> None:
    analysis = service(session, settings).detect_structure(document.id)

    summary = analysis.structure_summary
    assert summary["node_count"] == 16 and summary["node_types"]["chapter"] == 3
    assert summary["detected_metadata"]["title"] == "Foundations of Learning"
    assert analysis.passage_config["target_tokens"] == 750
    assert analysis.passage_summary["passage_count"] == count(session, Passage)
    assert analysis.structure_detected_at and analysis.passage_generated_at


def test_repeated_processing_creates_no_duplicate_records(session: Session, settings: Settings, document: Document) -> None:
    svc = service(session, settings)
    svc.detect_structure(document.id)
    first = snapshot(session, document.id)
    first_ids = set(session.scalars(select(StructuralNode.id)))

    for _ in range(3):
        svc.detect_structure(document.id)

    assert snapshot(session, document.id) == first
    assert count(session, StructureAnalysis) == 1
    assert count(session, StructuralNode) == len(first["nodes"])
    assert set(session.scalars(select(StructuralNode.id))).isdisjoint(first_ids)  # replaced, not appended
    assert count(session, PassageParagraph) == session.scalar(text("select count(*) from passage_paragraphs"))


def test_the_book_text_is_read_from_the_database_not_parsed_again(
    session: Session, settings: Settings, document: Document
) -> None:
    # The document's file does not exist in storage: only the stored blocks are used.
    assert not (settings.storage_root / document.file_path).exists()

    service(session, settings).detect_structure(document.id)

    assert count(session, StructuralNode) > 0


def test_passage_regeneration_does_not_touch_structure(
    session: Session, settings: Settings, document: Document, monkeypatch
) -> None:
    svc = service(session, settings)
    first = svc.detect_structure(document.id)
    node_ids = set(session.scalars(select(StructuralNode.id)))
    paragraph_ids = set(session.scalars(select(Paragraph.id)))
    passages_before = count(session, Passage)

    def forbidden(*args, **kwargs):
        raise AssertionError("structure detection must not run")

    monkeypatch.setattr("app.services.structure_service.StructureDetector.detect", forbidden)
    config = SegmentationConfig(target_tokens=60, min_tokens=30, max_tokens=90)
    second = svc.regenerate_passages(document.id, config)

    assert set(session.scalars(select(StructuralNode.id))) == node_ids
    assert set(session.scalars(select(Paragraph.id))) == paragraph_ids
    assert count(session, Passage) > passages_before
    assert second.passage_config["target_tokens"] == 60
    assert second.structure_detected_at == first.structure_detected_at
    assert not [i for i in second.passage_issues if i.severity == "error"]


def test_regenerating_passages_twice_gives_the_same_passages(session: Session, settings: Settings, document: Document) -> None:
    svc = service(session, settings)
    svc.detect_structure(document.id)
    config = SegmentationConfig(target_tokens=120, min_tokens=60, max_tokens=200)
    query = select(Passage.sequence, Passage.document_sequence, Passage.text).order_by(Passage.document_sequence)

    svc.regenerate_passages(document.id, config)
    first = list(session.execute(query))
    svc.regenerate_passages(document.id, config)
    second = list(session.execute(query))

    assert first == second


def test_default_passage_sizes_come_from_application_settings(session: Session, settings: Settings, document: Document) -> None:
    custom = settings.model_copy(
        update={"passage_target_tokens": 100, "passage_min_tokens": 50, "passage_max_tokens": 150}
    )

    analysis = service(session, custom).detect_structure(document.id)

    assert analysis.passage_config["target_tokens"] == 100
    assert analysis.passage_summary["max_tokens"] <= 150 + 30


# -- guards ------------------------------------------------------------------------------------


def test_unprocessed_documents_cannot_be_analysed(session: Session, settings: Settings, document: Document) -> None:
    document.processing_status = ProcessingStatus.UPLOADED.value
    session.commit()

    with pytest.raises(StructureNotAllowedError, match="processed"):
        service(session, settings).detect_structure(document.id)
    assert count(session, StructuralNode) == 0


def test_unknown_documents_are_reported(session: Session, settings: Settings) -> None:
    with pytest.raises(DocumentNotFoundError):
        service(session, settings).detect_structure(uuid.uuid4())
    with pytest.raises(DocumentNotFoundError):
        service(session, settings).get_overview(uuid.uuid4())


def test_passages_need_the_structure_first(session: Session, settings: Settings, document: Document) -> None:
    with pytest.raises(StructureNotAllowedError, match="structure"):
        service(session, settings).regenerate_passages(document.id)


def test_a_document_without_text_is_reported(session: Session, settings: Settings, document: Document) -> None:
    session.execute(ExtractedBlock.__table__.delete().where(ExtractedBlock.document_id == document.id))
    session.commit()

    with pytest.raises(StructureServiceError, match="no extracted text"):
        service(session, settings).detect_structure(document.id)


def test_a_failure_keeps_the_previous_structure_and_is_recorded(
    session: Session, settings: Settings, document: Document, monkeypatch
) -> None:
    svc = service(session, settings)
    svc.detect_structure(document.id)
    before = snapshot(session, document.id)

    def explode(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("app.services.structure_service.StructureDetector.detect", explode)
    with pytest.raises(StructureServiceError, match="failed unexpectedly"):
        svc.detect_structure(document.id)

    assert snapshot(session, document.id) == before
    analysis = session.scalar(select(StructureAnalysis))
    assert analysis.structure_status == "failed" and "kaboom" in analysis.structure_error


def test_an_unavailable_tokenizer_is_a_clear_error(
    session: Session, settings: Settings, document: Document, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)

    with pytest.raises(StructureServiceError, match="tiktoken"):
        service(session, settings).detect_structure(document.id, SegmentationConfig(tokenizer="tiktoken"))
    assert count(session, StructuralNode) == 0


# -- book metadata -----------------------------------------------------------------------------


def test_detected_metadata_never_overwrites_the_stored_book(session: Session, settings: Settings, document: Document) -> None:
    service(session, settings).detect_structure(document.id)

    book = session.get(Book, document.book_id)
    session.refresh(book)
    assert (book.title, book.author, book.subtitle) == ("Sample Book", "A. Writer", None)


def test_empty_book_fields_can_be_filled_but_user_values_are_kept(session: Session, settings: Settings) -> None:
    doc = persist_document(session, numbered_textbook().build(), title="Chosen Title", author=None)
    enabled = settings.model_copy(
        update={"structure_fill_empty_book_metadata": True, "structure_metadata_min_confidence": 0.5}
    )

    analysis = service(session, enabled).detect_structure(doc.id)

    book = session.get(Book, doc.book_id)
    session.refresh(book)
    assert book.title == "Chosen Title"  # never replaced
    assert book.subtitle == "An Introduction to Representation"
    assert book.author == "Jane A. Smith"
    assert analysis.structure_summary["applied_book_metadata"]["author"] == "Jane A. Smith"


def test_uncertain_detections_are_not_applied(session: Session, settings: Settings) -> None:
    doc = persist_document(session, numbered_textbook().build(), title="Chosen Title", author=None)
    enabled = settings.model_copy(update={"structure_fill_empty_book_metadata": True})  # needs 0.8 confidence

    service(session, enabled).detect_structure(doc.id)

    book = session.get(Book, doc.book_id)
    session.refresh(book)
    assert book.author is None and book.subtitle is None  # the detections were only 0.6


def test_filling_the_author_never_creates_a_duplicate_book(session: Session, settings: Settings) -> None:
    session.add(Book(title="Chosen Title", author="Jane A. Smith"))
    session.commit()
    doc = persist_document(session, numbered_textbook().build(), title="Chosen Title", author=None)
    enabled = settings.model_copy(
        update={"structure_fill_empty_book_metadata": True, "structure_metadata_min_confidence": 0.5}
    )

    service(session, enabled).detect_structure(doc.id)

    book = session.get(Book, doc.book_id)
    session.refresh(book)
    assert book.author is None


# -- LLM through the service --------------------------------------------------------------------


def test_service_uses_an_injected_classifier(session: Session, settings: Settings) -> None:
    doc = persist_document(session, plain_text_book().build())
    classifier = FakeClassifier(
        {"Measures of Centre": {"classification": "section", "level": 2, "confidence": 0.95, "reasoning": "a heading"}}
    )

    service(session, settings, classifier=classifier).detect_structure(doc.id)

    node = session.scalar(select(StructuralNode).where(StructuralNode.title == "Measures of Centre"))
    assert node.node_type == "section" and node.node_metadata["method"] == "llm"


def test_misconfigured_llm_is_reported_but_does_not_block_detection(session: Session, settings: Settings) -> None:
    doc = persist_document(session, plain_text_book().build())
    broken = settings.model_copy(update={"structure_llm_enabled": True, "structure_llm_model": ""})

    analysis = service(session, broken).detect_structure(doc.id)

    assert analysis.structure_status == "completed"
    assert "llm_unavailable" in {i.code for i in analysis.structure_issues}


# -- reading -------------------------------------------------------------------------------------


def test_overview_lists_nodes_counts_and_metadata(session: Session, settings: Settings, document: Document) -> None:
    svc = service(session, settings)
    svc.detect_structure(document.id)

    overview = svc.get_overview(document.id)

    assert overview.book.title == "Sample Book"
    assert [n.sequence for n in overview.nodes] == sorted(n.sequence for n in overview.nodes)
    assert sum(overview.paragraph_counts.values()) == count(session, Paragraph)
    assert sum(overview.passage_counts.values()) == count(session, Passage)
    assert overview.analysis.structure_status == "completed"
    assert overview.is_outdated is False
    models = next(n for n in overview.nodes if n.title == "Models")
    assert models.metadata["method"] == "deterministic" and models.metadata["signals"]


def test_overview_before_detection_has_no_analysis(session: Session, settings: Settings, document: Document) -> None:
    overview = service(session, settings).get_overview(document.id)

    assert overview.analysis is None and overview.nodes == []


def test_overview_notices_when_the_document_was_processed_again(
    session: Session, settings: Settings, document: Document
) -> None:
    svc = service(session, settings)
    svc.detect_structure(document.id)
    session.add(
        ProcessingRun(
            document_id=document.id, status=ProcessingRunStatus.COMPLETED.value, result_summary={"page_count": 9}
        )
    )
    session.commit()

    assert svc.get_overview(document.id).is_outdated is True


def test_paragraphs_and_passages_can_be_listed_per_section_with_pages(
    session: Session, settings: Settings, document: Document
) -> None:
    svc = service(session, settings)
    svc.detect_structure(document.id)
    overview = svc.get_overview(document.id)
    chapter = next(n for n in overview.nodes if n.title == "Introduction")
    section = next(n for n in overview.nodes if n.title == "Background")

    section_passages = svc.list_passages(document.id, [section.id])
    chapter_tree = svc.list_passages(document.id, [chapter.id, section.id])
    paragraphs = svc.list_paragraphs(document.id, [section.id], limit=1)

    assert section_passages and all(p.structural_node_id == section.id for p in section_passages)
    assert len(chapter_tree) > len(section_passages)
    assert all(p.page_start == 5 and p.page_end == 5 and p.page_status == "exact" for p in section_passages)
    assert section_passages[0].paragraph_ids and section_passages[0].metadata["paragraph_ids"]
    assert len(paragraphs) == 1 and paragraphs[0].structural_node_id == section.id
    assert svc.count_paragraphs(document.id, [section.id]) == 2
    assert svc.count_passages(document.id) == count(session, Passage)
    assert svc.list_passages(document.id, [chapter.id, section.id], offset=1, limit=1)[0].id == chapter_tree[1].id


def test_equation_records_are_linked_to_their_paragraphs_and_passages(session: Session, settings: Settings) -> None:
    b = DocBuilder().title("Math").new_page().heading("1. Optimisation").prose(80, 1).equation("f(x) = x^2")
    doc = persist_document(session, b.build())
    block = session.scalar(
        select(ExtractedBlock).where(ExtractedBlock.document_id == doc.id, ExtractedBlock.block_type == "equation")
    )
    equation = Equation(
        document_id=doc.id,
        extracted_block_id=block.id,
        sequence_number=0,
        identity_hash="h",
        latex="f(x)=x^2",
        extraction_method="docling_enrichment",
        validation_status="valid",
    )
    session.add(equation)
    session.commit()

    service(session, settings).detect_structure(doc.id)

    paragraph = session.scalar(select(Paragraph).where(Paragraph.kind == "equation"))
    passage = session.scalar(select(Passage))
    assert paragraph.source_metadata["equation_ids"] == [str(equation.id)]
    assert passage.passage_metadata["equation_ids"] == [str(equation.id)]
    assert passage.passage_metadata["has_equation"] is True


def test_documents_without_pages_store_unknown_pages(session: Session, settings: Settings) -> None:
    b = DocBuilder(pages=False, file_format="epub", source_title="Pageless")
    b.heading("Chapter 1: One").prose(100, 1).heading("Chapter 2: Two").prose(100, 2)
    doc = persist_document(session, b.build())

    analysis = service(session, settings).detect_structure(doc.id)

    assert analysis.structure_status == "completed"
    assert all(n.page_start is None for n in session.scalars(select(StructuralNode)))
    assert all(p.page_status == "unknown" and p.page_start is None for p in session.scalars(select(Passage)))
    assert not [i for i in analysis.structure_issues if i.code == "missing_page_reference"]


def test_deleting_a_document_removes_its_structure(session: Session, settings: Settings, document: Document) -> None:
    service(session, settings).detect_structure(document.id)

    session.delete(session.get(Document, document.id))
    session.commit()

    for model in (StructuralNode, Paragraph, Passage, PassageParagraph, StructureAnalysis):
        assert count(session, model) == 0


def test_a_large_book_is_saved_in_batches(session: Session, settings: Settings) -> None:
    b = DocBuilder().title("Big")
    for chapter in range(60):
        b.new_page().heading(f"{chapter + 1}. Chapter {chapter + 1}")
        for section in range(5):
            b.heading(f"{chapter + 1}.{section + 1} Part {section + 1}", 2).prose(120, chapter * 10 + section)
    doc = persist_document(session, b.build())

    analysis = service(session, settings).detect_structure(doc.id)

    assert analysis.structure_summary["node_count"] > 300
    assert count(session, StructuralNode) == analysis.structure_summary["node_count"]
