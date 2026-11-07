import io
import zipfile

PDF_SIZE_PADDING = b"0123456789abcdef" * 8


def make_pdf(seed: str = "default") -> bytes:
    """A minimal PDF-looking file. `seed` makes the content (and hash) unique."""
    body = f"%PDF-1.4\n% seed:{seed}\n".encode() + PDF_SIZE_PADDING
    return body + b"\n%%EOF\n"


def make_epub(seed: str = "default") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", "<container/>")
        archive.writestr("content.opf", f"<package><!-- {seed} --></package>")
    return buffer.getvalue()


def make_docx(seed: str = "default") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", f"<document><!-- {seed} --></document>")
    return buffer.getvalue()


def make_plain_zip(seed: str = "default") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", seed)
    return buffer.getvalue()
