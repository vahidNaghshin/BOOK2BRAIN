import pytest
from pydantic import ValidationError

from app.schemas import BookMetadata


def test_only_title_is_required() -> None:
    metadata = BookMetadata(title="Dune")

    assert metadata.title == "Dune"
    assert metadata.author is None and metadata.isbn is None


@pytest.mark.parametrize("title", ["", "   ", "\n\t"])
def test_title_is_required_and_not_blank(title: str) -> None:
    with pytest.raises(ValidationError):
        BookMetadata(title=title)


def test_missing_title_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BookMetadata()  # type: ignore[call-arg]


def test_text_is_trimmed_and_whitespace_collapsed() -> None:
    metadata = BookMetadata(title="  The   Left \n Hand  ", author="  Ursula  K.  Le Guin ")

    assert metadata.title == "The Left Hand"
    assert metadata.author == "Ursula K. Le Guin"


def test_blank_optional_fields_become_none() -> None:
    metadata = BookMetadata(
        title="Dune", author=" ", subtitle="", isbn="  ", language="", description="\n"
    )

    assert metadata.model_dump(exclude={"title"}) == {
        "author": None,
        "subtitle": None,
        "isbn": None,
        "language": None,
        "description": None,
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("978-0-306-40615-7", "9780306406157"),
        ("9780306406157", "9780306406157"),
        ("0-306-40615-2", "0306406152"),
        ("0-8044-2957-x", "080442957X"),
    ],
)
def test_valid_isbns_are_normalised(raw: str, expected: str) -> None:
    assert BookMetadata(title="T", isbn=raw).isbn == expected


@pytest.mark.parametrize("isbn", ["123", "9780306406158", "0306406153", "abcdefghij", "978030640615X"])
def test_invalid_isbns_are_rejected(isbn: str) -> None:
    with pytest.raises(ValidationError):
        BookMetadata(title="T", isbn=isbn)


def test_overlong_title_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BookMetadata(title="x" * 501)
