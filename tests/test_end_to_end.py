"""The complete workflow on a real PDF: upload, process through the real Docling pipeline,
explore the structure, open passages, verify their page references against the original file,
then "restart" and look again.

This is the Phase 1 acceptance test. It needs Docling's models; it is skipped (and reported) when
they cannot be loaded. Equation extraction is switched off to keep it quick; it is not part of
this workflow.
"""

import io
import re
import time
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.schemas import BookMetadata
from app.services.document_service import DocumentService
from app.services.errors import DuplicateDocumentError
from app.services.job_runner import ProcessingJobRunner
from app.services.structure_service import StructureService
from database.connection import build_engine
from database.models import Book, Document, Paragraph, Passage, ProcessingRun, StructuralNode
from tests.factories import BOOK_CHAPTERS, BOOK_SECTIONS, BOOK_TITLE, make_book_pdf
from tests.ui_helpers import button, has_button, patch_ui, run_app, screen


@pytest.fixture
def settings(test_database_url: str, tmp_path: Path) -> Settings:
    return Settings(
        database_url=test_database_url,
        storage_root=tmp_path / "storage",
        max_upload_size_mb=5,
        equation_extraction_enabled=False,
        _env_file=None,
    )


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def session_factory(engine: Engine):
    from contextlib import contextmanager

    @contextmanager
    def scope():
        local = Session(bind=engine, autoflush=False, expire_on_commit=False)
        try:
            yield local
        finally:
            local.close()

    return scope


def test_upload_process_explore_trace_and_restart(
    docling_ready, session: Session, document_service: DocumentService, storage, settings, engine: Engine,
    monkeypatch, test_database_url,
) -> None:
    pdf = make_book_pdf()

    # -- upload: the book and document are registered and the original file is stored ------------
    result = document_service.upload(
        file=io.BytesIO(pdf), filename="signal-learning.pdf",
        metadata=BookMetadata(title=BOOK_TITLE, author="Maria Oliveira", publisher="Open Press", publication_year=2024),
    )
    document = result.document
    assert result.book_created and result.book.publisher == "Open Press" and result.book.publication_year == 2024
    assert document.book_id == result.book.id and document.processing_status == "uploaded"
    assert storage.exists(document.file_path) and len(document.sha256_hash) == 64
    with pytest.raises(DuplicateDocumentError):
        document_service.upload(file=io.BytesIO(pdf), filename="again.pdf", metadata=BookMetadata(title=BOOK_TITLE, author="Maria Oliveira"))
    assert session.scalar(select(func.count()).select_from(Document)) == 1

    # -- processing: status and stage change while it runs, in the background ---------------------
    runner = ProcessingJobRunner(settings, session_factory=session_factory(engine), storage=storage)
    runner.submit(document.id)
    seen, deadline = [], time.monotonic() + 180
    while time.monotonic() < deadline:
        with engine.connect() as conn:
            row = conn.execute(
                select(Document.processing_status, ProcessingRun.status, ProcessingRun.stage)
                .join(ProcessingRun, ProcessingRun.document_id == Document.id)
                .where(Document.id == document.id)
            ).first()
        seen.append(tuple(row))
        if row[0] in ("completed", "failed"):
            break
        time.sleep(0.05)
    runner.shutdown(wait=True)
    assert seen[-1] == ("completed", "completed", "completed"), seen[-1]
    assert {status for status, _, _ in seen[:-1]} <= {"pending", "processing"}  # never "completed" early
    session.rollback()

    # -- everything is persisted -----------------------------------------------------------------
    run = session.scalar(select(ProcessingRun).where(ProcessingRun.document_id == document.id))
    assert run.result_summary["page_count"] == 9 and run.stage_results["validation"]["errors"] == 0
    nodes = list(session.scalars(select(StructuralNode).where(StructuralNode.document_id == document.id).order_by(StructuralNode.sequence)))
    chapters = [(n.number, n.title) for n in nodes if n.node_type == "chapter"]
    sections = [n.number for n in nodes if n.node_type == "section"]
    assert chapters == BOOK_CHAPTERS and sections == BOOK_SECTIONS
    kinds = {n.node_type for n in nodes}
    assert {"front_matter", "book_title", "copyright", "table_of_contents", "preface", "back_matter", "appendix", "bibliography"} <= kinds
    assert nodes[0].node_type == "front_matter"  # original order is kept
    passages = list(session.scalars(select(Passage).where(Passage.document_id == document.id).order_by(Passage.document_sequence)))
    assert passages and session.scalar(select(func.count()).select_from(Paragraph)) > 0

    # -- passages trace back to the right pages of the original file ------------------------------
    with pymupdf.open(stream=pdf, filetype="pdf") as original:
        page_text = {number: squash(original[number - 1].get_text()) for number in range(1, original.page_count + 1)}
    checked = 0
    for passage in passages:
        body = squash(passage.text[int((passage.passage_metadata or {}).get("overlap_chars") or 0):])
        assert passage.page_status == "exact" and 1 <= passage.page_start <= passage.page_end <= 9
        if len(body) < 40:
            continue
        first_pages = page_text[passage.page_start]
        last_pages = page_text[passage.page_end]
        assert body[:12] in first_pages, (passage.page_start, passage.text[:60])
        assert body[-12:] in last_pages, (passage.page_end, passage.text[-60:])
        checked += 1
    assert checked >= 5
    spanning = [p for p in passages if p.page_end > p.page_start]
    assert spanning, "the paragraph that runs over a page break must keep both pages"

    # -- the screens: library -> structure -> passages -> a passage ------------------------------
    patch_ui(monkeypatch, engine, settings, storage)
    at = run_app()
    assert any("Completed" in str(row) for table in at.dataframe for row in table.value.to_dict("records"))
    button(at, f"library_open_{document.id}").click().run()
    assert screen(at) == "Structure" and not at.exception

    button(at, f"expand-all-{document.id}").click().run()
    background = next(n for n in nodes if n.number == "1.1")
    button(at, f"select-{background.id}").click().run()
    assert any(m.value.startswith("### 1.1  Signals") for m in at.markdown)
    assert any("pp. 5–6" in c.value for c in at.caption) or any("p. 5" in m.value for m in at.markdown)

    # A section's own passages, with their complete text and page reference.
    button(at, f"view-passages-{background.id}").click().run()
    assert screen(at) == "Passages" and not at.exception
    first = next(p for p in passages if p.structural_node_id == background.id)
    button(at, f"open-passage-{first.id}").click().run()
    assert any(first.text == t.value for t in at.text)  # the complete stored text, unchanged
    shown = " ".join(m.value for m in at.markdown)
    assert "Signals" in shown and "Foundations" in shown
    expected_pages = f"p. {first.page_start}" if first.page_start == first.page_end else f"pp. {first.page_start}–{first.page_end}"
    assert expected_pages in shown
    button(at, "passage-structure").click().run()  # and back to where it sits in the structure
    assert screen(at) == "Structure" and any(m.value.startswith("### 1.1  Signals") for m in at.markdown)

    # Navigating between the passages of a whole chapter.
    chapter = next(n for n in nodes if n.number == "1" and n.node_type == "chapter")
    button(at, f"select-{chapter.id}").click().run()
    button(at, f"view-passages-{chapter.id}").click().run()
    in_chapter = [p for p in passages if p.structural_node_id in {n.id for n in nodes if n.id == chapter.id or n.parent_id == chapter.id}]
    assert len(in_chapter) >= 2
    button(at, f"open-passage-{in_chapter[0].id}").click().run()
    assert any(in_chapter[0].text == t.value for t in at.text)
    button(at, "passage-next").click().run()
    assert any(in_chapter[1].text == t.value for t in at.text)
    button(at, "passage-prev").click().run()
    assert any(in_chapter[0].text == t.value for t in at.text)
    button(at, "passage-back").click().run()
    assert has_button(at, f"open-passage-{in_chapter[0].id}")

    # -- restart: a new engine and session see the same data, and the screens still work ---------
    counts_before = (
        len(nodes), len(passages), session.scalar(select(func.count()).select_from(Paragraph)),
    )
    engine.dispose()
    restarted = build_engine(test_database_url)
    with Session(restarted) as fresh:
        assert fresh.scalar(select(func.count()).select_from(Book)) == 1
        assert fresh.get(Document, document.id).processing_status == "completed"
        assert (
            fresh.scalar(select(func.count()).select_from(StructuralNode)),
            fresh.scalar(select(func.count()).select_from(Passage)),
            fresh.scalar(select(func.count()).select_from(Paragraph)),
        ) == counts_before
        overview = StructureService(fresh, settings).get_overview(document.id)
        assert overview.analysis.structure_status == "completed" and not overview.is_outdated
    patch_ui(monkeypatch, restarted, settings, storage)
    again = run_app()
    assert any(has_button(again, k) for k in (f"library_open_{document.id}",))
    restarted.dispose()
