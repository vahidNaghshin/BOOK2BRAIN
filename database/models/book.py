from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from database.models.document import Document


class Book(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A logical book, independent of file format or edition."""

    __tablename__ = "books"

    title: Mapped[str] = mapped_column(String(500))
    subtitle: Mapped[str | None] = mapped_column(String(500))
    author: Mapped[str | None] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(64))
    # Not unique: one logical book can have several ISBNs (print, ebook, editions).
    isbn: Mapped[str | None] = mapped_column(String(20), index=True)

    documents: Mapped[list[Document]] = relationship(
        back_populates="book",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Document.created_at",
    )

    __table_args__ = (
        # A logical book is identified by case-insensitive title and author.
        Index(
            "uq_books_title_author",
            text("lower(title)"),
            text("lower(coalesce(author, ''))"),
            unique=True,
        ),
    )
