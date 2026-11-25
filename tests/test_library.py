import io

from sqlalchemy.orm import Session

from app.schemas import BookMetadata
from app.services.book_service import BookService
from app.services.document_service import DocumentService
from tests.factories import make_epub, make_pdf
from tests.ui_helpers import patch_ui


def upload(service: DocumentService, data: bytes, filename: str, metadata: BookMetadata):
    return service.upload(file=io.BytesIO(data), filename=filename, metadata=metadata)


def test_empty_library(session: Session) -> None:
    assert BookService(session).list_books() == []


def test_library_lists_books_with_their_documents(
    document_service: DocumentService, session: Session
) -> None:
    dune = BookMetadata(title="Dune", author="Frank Herbert")
    emma = BookMetadata(title="Emma", author="Jane Austen")
    upload(document_service, make_pdf("l1"), "dune.pdf", dune)
    upload(document_service, make_epub("l2"), "dune.epub", dune)
    upload(document_service, make_pdf("l3"), "emma.pdf", emma)

    books = BookService(session).list_books()

    by_title = {book.title: book for book in books}
    assert set(by_title) == {"Dune", "Emma"}
    assert {d.file_format for d in by_title["Dune"].documents} == {"pdf", "epub"}
    assert [d.file_format for d in by_title["Emma"].documents] == ["pdf"]
    document = by_title["Emma"].documents[0]
    assert document.file_size > 0
    assert document.processing_status == "uploaded"
    assert document.created_at is not None


def test_library_shows_most_recent_books_first(
    document_service: DocumentService, session: Session
) -> None:
    upload(document_service, make_pdf("o1"), "a.pdf", BookMetadata(title="First"))
    upload(document_service, make_pdf("o2"), "b.pdf", BookMetadata(title="Second"))

    assert [book.title for book in BookService(session).list_books()] == ["Second", "First"]


def test_get_book_returns_metadata_and_documents(
    document_service: DocumentService, session: Session
) -> None:
    metadata = BookMetadata(title="Dune", subtitle="Book One", author="Frank Herbert", language="en")
    result = upload(document_service, make_pdf("g1"), "dune.pdf", metadata)

    book = BookService(session).get_book(result.book.id)

    assert book is not None
    assert (book.title, book.subtitle, book.language) == ("Dune", "Book One", "en")
    assert [d.id for d in book.documents] == [result.document.id]


def test_get_unknown_book_returns_none(session: Session) -> None:
    import uuid

    assert BookService(session).get_book(uuid.uuid4()) is None


# -- the Delete button ---------------------------------------------------------------------------


def library_app() -> None:
    from app.ui.library import render_library_page

    render_library_page()


def test_delete_button_asks_for_confirmation_then_removes_the_document(
    document_service: DocumentService, session: Session, storage, settings, engine, monkeypatch
) -> None:
    from streamlit.testing.v1 import AppTest

    document = upload(document_service, make_pdf("ui1"), "dune.pdf", BookMetadata(title="Dune", author="Frank Herbert")).document
    patch_ui(monkeypatch, engine, settings, storage)

    at = AppTest.from_function(library_app, default_timeout=30)
    at.run()
    delete = next(b for b in at.button if b.key == f"library_delete_{document.id}")
    assert delete.label == "Delete"

    delete.click().run()
    assert any("cannot be undone" in w.value for w in at.warning)
    assert storage.exists(document.file_path)

    next(b for b in at.button if b.key == f"library_delete_no_{document.id}").click().run()
    assert any(b.key == f"library_delete_{document.id}" for b in at.button)

    next(b for b in at.button if b.key == f"library_delete_{document.id}").click().run()
    next(b for b in at.button if b.key == f"library_delete_yes_{document.id}").click().run()

    assert not at.exception
    assert not storage.exists(document.file_path)
    assert BookService(session).list_books() == []
    assert any("empty" in i.value for i in at.info)
