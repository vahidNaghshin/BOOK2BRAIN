import io
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

import docx
import pymupdf
from ebooklib import epub
from PIL import Image, ImageDraw, ImageFont

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


# -- Realistic documents for parsing tests (generated in code; nothing committed as binary) --


def _page_image(lines: Sequence[str], *, size: tuple[int, int] = (1700, 2200)) -> bytes:
    """A page-sized PNG with the given lines of text drawn on it (a stand-in for a scan)."""
    font = ImageFont.load_default(size=44)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    y = 200
    for line in lines:
        draw.text((150, y), line, fill="black", font=font)
        y += 100
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def make_text_pdf(pages: Sequence[Sequence[str]], *, headings: bool = True) -> bytes:
    """A digital PDF. Each page is a list of lines; the first line of page 1 is a large heading."""
    document = pymupdf.open()
    for page_index, lines in enumerate(pages):
        page = document.new_page()
        y = 72
        for line_index, line in enumerate(lines):
            is_heading = headings and page_index == 0 and line_index == 0
            page.insert_text((72, y), line, fontsize=22 if is_heading else 11)
            y += 40 if is_heading else 18
    return document.tobytes()


def make_mixed_pdf(page_specs: Sequence[tuple[str, Sequence[str]]]) -> bytes:
    """A PDF whose pages are each 'text', 'scan', 'both' or 'blank'; `lines` is the content.

    'both' is a page with a (short) text layer *and* a page image, e.g. a scan with only a
    page number as text.
    """
    document = pymupdf.open()
    for kind, lines in page_specs:
        page = document.new_page()
        if kind == "text":
            y = 72
            for line in lines:
                page.insert_text((72, y), line, fontsize=11)
                y += 18
        elif kind == "titled":  # like 'text', but the first line is a large heading
            y = 72
            for index, line in enumerate(lines):
                page.insert_text((72, y), line, fontsize=22 if index == 0 else 11)
                y += 40 if index == 0 else 18
        elif kind == "scan":
            page.insert_image(page.rect, stream=_page_image(lines))
        elif kind == "both":
            page.insert_image(page.rect, stream=_page_image(["image content on the page"]))
            y = 72
            for line in lines:
                page.insert_text((72, y), line, fontsize=11)
                y += 18
        elif kind != "blank":
            raise ValueError(kind)
    return document.tobytes()


def make_scanned_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    return make_mixed_pdf([("scan", lines) for lines in pages])


def make_encrypted_pdf() -> bytes:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Secret content that nobody can read.")
    return document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner"
    )


def make_corrupt_pdf() -> bytes:
    return b"%PDF-1.4\nthis is definitely not a real pdf body\n"


def make_zero_page_pdf() -> bytes:
    return (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
        b"trailer<</Root 1 0 R/Size 3>>\n%%EOF\n"
    )


def make_blank_pdf() -> bytes:
    document = pymupdf.open()
    document.new_page()
    return document.tobytes()


def make_epub_book(
    chapters: Sequence[tuple[str, str, str]],
    *,
    spine_order: Sequence[int] | None = None,
    title: str = "Test Book",
    toc: bool = True,
) -> bytes:
    """An EPUB from (file_name, toc_title, html_body) chapters. `spine_order` reorders reading."""
    book = epub.EpubBook()
    book.set_identifier("test-book-id")
    book.set_title(title)
    book.set_language("en")
    items = []
    for file_name, chapter_title, body in chapters:
        item = epub.EpubHtml(title=chapter_title, file_name=file_name, lang="en")
        item.content = f"<html><body>{body}</body></html>"
        book.add_item(item)
        items.append(item)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    if toc:
        book.toc = [epub.Link(i.file_name, i.title, f"c{n}") for n, i in enumerate(items)]
    order = spine_order if spine_order is not None else range(len(items))
    book.spine = ["nav", *[items[n] for n in order]]

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "book.epub"
        epub.write_epub(str(path), book)
        return path.read_bytes()


def make_docx_document(
    *,
    heading: str | None = "Report Title",
    paragraphs: Sequence[str] = ("First paragraph of the document.",),
    bullets: Sequence[str] = (),
    numbered: Sequence[str] = (),
    table: Sequence[Sequence[str]] | None = None,
    closing: str | None = None,
) -> bytes:
    """A DOCX with a heading, paragraphs, lists and an optional table, in that order."""
    document = docx.Document()
    if heading:
        document.add_heading(heading, 1)
    for text in paragraphs:
        document.add_paragraph(text)
    for text in bullets:
        document.add_paragraph(text, style="List Bullet")
    for text in numbered:
        document.add_paragraph(text, style="List Number")
    if table:
        grid = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                grid.cell(r, c).text = value
    if closing:
        document.add_paragraph(closing)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_docx_with_broken_xml() -> bytes:
    """A ZIP with the files upload validation looks for, but unparseable content."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types")
        archive.writestr("word/document.xml", "<w:document><unclosed>")
    return buffer.getvalue()
