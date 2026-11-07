import logging
import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.repositories.book_repository import BookRepository
from app.schemas import BookMetadata, BookRead
from database.models import Book

logger = logging.getLogger(__name__)


class BookService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.books = BookRepository(session)

    def get_or_create(self, metadata: BookMetadata) -> tuple[Book, bool]:
        """Return the book matching title and author, creating it if needed.

        An existing book is returned unchanged. The caller owns the transaction.
        """
        existing = self.books.find_by_identity(metadata.title, metadata.author)
        if existing is not None:
            return existing, False

        book = Book(**metadata.model_dump())
        try:
            # A savepoint lets us recover from a concurrent insert of the same book.
            with self.session.begin_nested():
                self.books.add(book)
        except IntegrityError:
            existing = self.books.find_by_identity(metadata.title, metadata.author)
            if existing is None:
                raise
            logger.info("Book '%s' was created concurrently; reusing it", metadata.title)
            return existing, False

        logger.info("Created book %s '%s'", book.id, book.title)
        return book, True

    def get_book(self, book_id: uuid.UUID) -> BookRead | None:
        book = self.books.get(book_id)
        return BookRead.model_validate(book) if book else None

    def list_books(self) -> list[BookRead]:
        return [BookRead.model_validate(book) for book in self.books.list_all()]
