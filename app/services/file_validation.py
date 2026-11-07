import enum
import hashlib
import unicodedata
import zipfile
from pathlib import PurePath
from typing import BinaryIO

from app.services.errors import (
    EmptyFileError,
    FileTooLargeError,
    InvalidFileError,
    UnsupportedFileTypeError,
)

CHUNK_SIZE = 1024 * 1024
MAX_FILENAME_LENGTH = 255
_ZIP_MAGIC = b"PK\x03\x04"
_PDF_MAGIC = b"%PDF-"
_PDF_HEADER_WINDOW = 1024
_ZIP_MIME_TYPES = {"application/zip", "application/x-zip-compressed"}
# Browsers frequently send no type, or a generic one, for .epub files.
_GENERIC_MIME_TYPES = {"", "application/octet-stream", "binary/octet-stream"}


class FileFormat(enum.StrEnum):
    PDF = "pdf"
    EPUB = "epub"
    DOCX = "docx"

    @property
    def extension(self) -> str:
        return f".{self.value}"

    @property
    def mime_type(self) -> str:
        return _CANONICAL_MIME_TYPES[self]


_CANONICAL_MIME_TYPES = {
    FileFormat.PDF: "application/pdf",
    FileFormat.EPUB: "application/epub+zip",
    FileFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_ACCEPTED_MIME_TYPES = {
    FileFormat.PDF: {"application/pdf", "application/x-pdf"},
    FileFormat.EPUB: {"application/epub+zip"} | _ZIP_MIME_TYPES,
    FileFormat.DOCX: {_CANONICAL_MIME_TYPES[FileFormat.DOCX]} | _ZIP_MIME_TYPES,
}

SUPPORTED_EXTENSIONS = tuple(fmt.extension for fmt in FileFormat)


def sanitize_filename(raw_name: str) -> str:
    """Return a display-safe filename: no directories, no control characters."""
    name = unicodedata.normalize("NFC", raw_name or "").replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    if not name or name.strip(".") == "":
        raise InvalidFileError("The uploaded file has no usable filename.")
    if len(name) > MAX_FILENAME_LENGTH:
        suffix = PurePath(name).suffix
        name = name[: MAX_FILENAME_LENGTH - len(suffix)] + suffix
    return name


def format_from_filename(filename: str) -> FileFormat:
    suffix = PurePath(filename).suffix.lower()
    for file_format in FileFormat:
        if suffix == file_format.extension:
            return file_format
    allowed = ", ".join(SUPPORTED_EXTENSIONS)
    shown = suffix or "no extension"
    raise UnsupportedFileTypeError(f"Unsupported file type ({shown}). Allowed types: {allowed}.")


def check_declared_mime_type(file_format: FileFormat, declared: str | None) -> None:
    """Reject a client-declared MIME type that contradicts the extension."""
    if declared is None:
        return
    base = declared.split(";")[0].strip().lower()
    if base in _GENERIC_MIME_TYPES or base in _ACCEPTED_MIME_TYPES[file_format]:
        return
    raise InvalidFileError(
        f"The file's declared type '{base}' does not match a {file_format.extension} file."
    )


def measure_stream(file: BinaryIO, max_bytes: int) -> tuple[str, int]:
    """Stream the file once, returning (sha256 hex digest, size). The stream is rewound."""
    digest = hashlib.sha256()
    size = 0
    file.seek(0)
    while chunk := file.read(CHUNK_SIZE):
        size += len(chunk)
        if size > max_bytes:
            raise FileTooLargeError(
                f"The file exceeds the maximum upload size of {max_bytes / 1024 / 1024:g} MB."
            )
        digest.update(chunk)
    if size == 0:
        raise EmptyFileError("The uploaded file is empty.")
    file.seek(0)
    return digest.hexdigest(), size


def detect_format(file: BinaryIO) -> FileFormat:
    """Identify the real format from file content, without parsing the document."""
    file.seek(0)
    head = file.read(_PDF_HEADER_WINDOW)

    if head.startswith(_ZIP_MAGIC):
        try:
            with zipfile.ZipFile(file) as archive:
                names = set(archive.namelist())
                if "mimetype" in names:
                    with archive.open("mimetype") as member:
                        if member.read(64).strip() == b"application/epub+zip":
                            return FileFormat.EPUB
                if {"[Content_Types].xml", "word/document.xml"} <= names:
                    return FileFormat.DOCX
        except zipfile.BadZipFile as exc:
            raise InvalidFileError("The file is a corrupt ZIP-based document.") from exc
        finally:
            file.seek(0)
        raise InvalidFileError("The file is a ZIP archive but not a valid EPUB or DOCX document.")

    file.seek(0)
    if _PDF_MAGIC in head:
        return FileFormat.PDF
    raise InvalidFileError("The file content is not a valid PDF, EPUB or DOCX document.")
