import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from database.models import Book


class BookRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, book: Book) -> Book:
        self.session.add(book)
        self.session.flush()
        return book

    def get(self, book_id: uuid.UUID) -> Book | None:
        stmt = select(Book).where(Book.id == book_id).options(selectinload(Book.documents))
        return self.session.execute(stmt).scalar_one_or_none()

    def find_by_identity(self, title: str, author: str | None) -> Book | None:
        """Find a book by case-insensitive title and author (matches uq_books_title_author)."""
        stmt = select(Book).where(
            func.lower(Book.title) == func.lower(title),
            func.lower(func.coalesce(Book.author, "")) == func.lower(author or ""),
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def list_all(self) -> list[Book]:
        stmt = (
            select(Book)
            .options(selectinload(Book.documents))
            .order_by(Book.created_at.desc(), Book.title)
        )
        return list(self.session.execute(stmt).scalars().all())
