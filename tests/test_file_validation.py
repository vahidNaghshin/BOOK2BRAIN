import io

import pytest

from app.services.errors import (
    EmptyFileError,
    FileTooLargeError,
    InvalidFileError,
    UnsupportedFileTypeError,
)
from app.services.file_validation import (
    FileFormat,
    check_declared_mime_type,
    detect_format,
    format_from_filename,
    measure_stream,
    sanitize_filename,
)
from tests.factories import make_docx, make_epub, make_pdf, make_plain_zip


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("book.pdf", FileFormat.PDF),
        ("book.EPUB", FileFormat.EPUB),
        ("My Book.v2.docx", FileFormat.DOCX),
    ],
)
def test_supported_extensions(name: str, expected: FileFormat) -> None:
    assert format_from_filename(name) is expected


@pytest.mark.parametrize("name", ["book.txt", "book.exe", "book.doc", "book", "pdf", "book.pdf.exe"])
def test_unsupported_extensions(name: str) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        format_from_filename(name)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("book.pdf", "book.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\me\\book.pdf", "book.pdf"),
        ("/abs/path/book.epub", "book.epub"),
        ("bo\x00ok\n.pdf", "book.pdf"),
    ],
)
def test_sanitize_filename_strips_paths_and_control_characters(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "..", "/", "\x00"])
def test_sanitize_filename_rejects_unusable_names(raw: str) -> None:
    with pytest.raises(InvalidFileError):
        sanitize_filename(raw)


def test_sanitize_filename_truncates_long_names_keeping_extension() -> None:
    name = sanitize_filename("a" * 400 + ".pdf")
    assert len(name) == 255 and name.endswith(".pdf")


def test_declared_mime_type_rules() -> None:
    check_declared_mime_type(FileFormat.PDF, None)
    check_declared_mime_type(FileFormat.PDF, "application/pdf")
    check_declared_mime_type(FileFormat.PDF, "application/pdf; charset=binary")
    check_declared_mime_type(FileFormat.EPUB, "application/octet-stream")
    check_declared_mime_type(FileFormat.EPUB, "")
    check_declared_mime_type(FileFormat.DOCX, "application/x-zip-compressed")
    with pytest.raises(InvalidFileError):
        check_declared_mime_type(FileFormat.PDF, "image/png")
    with pytest.raises(InvalidFileError):
        check_declared_mime_type(FileFormat.EPUB, "application/pdf")


def test_measure_stream_returns_hash_and_size_and_rewinds() -> None:
    data = b"hello world"
    stream = io.BytesIO(data)

    digest, size = measure_stream(stream, max_bytes=1024)

    assert size == len(data)
    assert digest == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert stream.tell() == 0


def test_measure_stream_rejects_empty_file() -> None:
    with pytest.raises(EmptyFileError):
        measure_stream(io.BytesIO(b""), max_bytes=1024)


def test_measure_stream_rejects_oversized_file() -> None:
    with pytest.raises(FileTooLargeError):
        measure_stream(io.BytesIO(b"x" * 11), max_bytes=10)
    measure_stream(io.BytesIO(b"x" * 10), max_bytes=10)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (make_pdf(), FileFormat.PDF),
        (b"\n\n" + make_pdf(), FileFormat.PDF),
        (make_epub(), FileFormat.EPUB),
        (make_docx(), FileFormat.DOCX),
    ],
)
def test_detect_format_from_content(data: bytes, expected: FileFormat) -> None:
    stream = io.BytesIO(data)
    assert detect_format(stream) is expected
    assert stream.tell() == 0


@pytest.mark.parametrize(
    "data",
    [
        b"just some text, not a document",
        b"PK\x03\x04 truncated zip",
        make_plain_zip(),
        b"MZ\x90\x00 executable",
    ],
    ids=["text", "corrupt-zip", "plain-zip", "executable"],
)
def test_detect_format_rejects_other_content(data: bytes) -> None:
    with pytest.raises(InvalidFileError):
        detect_format(io.BytesIO(data))
