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


# -- Word equations (OMML) -------------------------------------------------------------

_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _r(text: str) -> str:
    return f"<m:r><m:t>{text}</m:t></m:r>"


def _sup(base: str, exponent: str) -> str:
    return f"<m:sSup><m:e>{_r(base)}</m:e><m:sup>{_r(exponent)}</m:sup></m:sSup>"


def _frac(numerator: str, denominator: str) -> str:
    return f"<m:f><m:num>{numerator}</m:num><m:den>{denominator}</m:den></m:f>"


OMML_SNIPPETS: dict[str, str] = {
    "force": _r("F=ma"),
    "energy": _r("E=") + _r("m") + _sup("c", "2"),
    "kinetic": _r("KE=") + _frac(_r("1"), _r("2")) + _r("m") + _sup("v", "2"),
    "integral": (
        "<m:nary><m:naryPr><m:chr m:val=\"∫\"/></m:naryPr><m:sub>" + _r("0") + "</m:sub><m:sup>"
        + _r("1") + "</m:sup><m:e>" + _r("x") + _r("dx") + "</m:e></m:nary>"
    ),
    "sum": (
        "<m:nary><m:naryPr><m:chr m:val=\"∑\"/></m:naryPr><m:sub>" + _r("i=1") + "</m:sub><m:sup>"
        + _r("n") + "</m:sup><m:e>" + _r("i") + "</m:e></m:nary>"
    ),
    "matrix": (
        _r("A=") + "<m:d><m:dPr/><m:e><m:m>"
        + "<m:mr><m:e>" + _r("a") + "</m:e><m:e>" + _r("b") + "</m:e></m:mr>"
        + "<m:mr><m:e>" + _r("c") + "</m:e><m:e>" + _r("d") + "</m:e></m:mr>"
        + "</m:m></m:e></m:d>"
    ),
    "sqrt": "<m:rad><m:radPr><m:degHide m:val=\"1\"/></m:radPr><m:deg/><m:e>" + _r("a") + _sup("b", "2") + "</m:e></m:rad>",
    "greek": _r("α") + "<m:sSub><m:e>" + _r("x") + "</m:e><m:sub>" + _r("i") + "</m:sub></m:sSub>",
    # Valid per the schema, but Docling's converter raises KeyError on it.
    "malformed": "<m:d><m:dPr/>" + _r("a") + "</m:d>",
}


def make_docx_with_equations(items: Sequence[tuple]) -> bytes:
    """A DOCX built from items, using real OMML:

    ("heading", text) · ("paragraph", text) · ("inline", before, key, after) ·
    ("display", key) · ("display", key, "(1.1)")   [number as separate text]
    `key` indexes OMML_SNIPPETS.
    """
    from lxml import etree

    def omml(inner: str) -> etree._Element:
        return etree.fromstring(f'<m:oMath xmlns:m="{_M}" xmlns:w="{_W}">{inner}</m:oMath>')

    document = docx.Document()
    for kind, *rest in items:
        if kind == "heading":
            document.add_heading(rest[0], 1)
        elif kind == "paragraph":
            document.add_paragraph(rest[0])
        elif kind == "inline":
            before, key, after = rest
            paragraph = document.add_paragraph(before)
            paragraph._p.append(omml(OMML_SNIPPETS[key]))
            paragraph.add_run(after)
        elif kind == "display":
            paragraph = document.add_paragraph()
            etree.SubElement(paragraph._p, f"{{{_M}}}oMathPara").append(omml(OMML_SNIPPETS[rest[0]]))
            if len(rest) > 1:
                paragraph.add_run(rest[1])
        else:
            raise ValueError(kind)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# -- Mathematical PDFs (ground-truth LaTeX rendered with matplotlib's mathtext) ----------

# (category, LaTeX). Written for matplotlib's mathtext, which has no matrix environments.
MATH_CORPUS: list[tuple[str, str]] = [
    ("algebra", r"x^{2} + 2x + 1 = 0"),
    ("algebra", r"x = \frac{-b \pm \sqrt{b^{2} - 4ac}}{2a}"),
    ("algebra", r"\frac{a + b}{c - d} = \sqrt{x^{2} + y^{2}}"),
    ("calculus", r"\int_{0}^{\infty} e^{-x^{2}} dx = \frac{\sqrt{\pi}}{2}"),
    ("calculus", r"\lim_{x \to 0} \frac{\sin x}{x} = 1"),
    ("differential", r"\frac{dy}{dx} + y = 0"),
    ("differential", r"\frac{\partial f}{\partial x} = 2x"),
    ("series", r"\sum_{n=1}^{\infty} \frac{1}{n^{2}} = \frac{\pi^{2}}{6}"),
    ("series", r"\prod_{k=1}^{n} k = n!"),
    ("greek", r"\alpha + \beta = \gamma \cdot \delta"),
    ("physics", r"F = ma"),
    ("physics", r"E = mc^{2}"),
    ("statistics", r"\sigma^{2} = \frac{1}{N} \sum_{i=1}^{N} (x_{i} - \mu)^{2}"),
    ("probability", r"P(A|B) = \frac{P(B|A) P(A)}{P(B)}"),
    ("finance", r"A = P \left(1 + \frac{r}{n}\right)^{nt}"),
    ("logic", r"A \cup B = \{x : x \in A \vee x \in B\}"),
    ("vector", r"\vec{F} = m \vec{a}"),
    ("chemistry", r"2 H_{2} + O_{2} \rightarrow 2 H_{2}O"),
    ("optimization", r"\min_{x} f(x) = x^{2} + 1"),
    ("engineering", r"V = I R + L \frac{dI}{dt}"),
]


def make_math_pdf(
    equations: Sequence[str],
    *,
    per_page: int = 5,
    scanned: bool = False,
    with_text: bool = True,
) -> bytes:
    """A book-like PDF: a heading, explanatory text and display equations (one per paragraph).

    Equations are real mathtext glyphs (digital) or, with `scanned=True`, only an image of each
    page. Returns the PDF bytes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    document = pymupdf.open()
    for start in range(0, len(equations), per_page):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.1, 0.93, "Chapter 3: Worked relations", fontsize=20, family="serif")
        y = 0.86
        for latex in equations[start : start + per_page]:
            if with_text:
                fig.text(0.1, y, "The following relation is used throughout this section:", fontsize=11, family="serif")
                y -= 0.06
            fig.text(0.5, y, f"${latex}$", fontsize=18, ha="center")
            y -= 0.1
            if with_text:
                fig.text(0.1, y, "where the symbols have their usual meaning.", fontsize=11, family="serif")
                y -= 0.07
        buffer = io.BytesIO()
        fig.savefig(buffer, format="pdf")
        plt.close(fig)
        page_pdf = pymupdf.open(stream=buffer.getvalue(), filetype="pdf")
        if scanned:
            pixmap = page_pdf[0].get_pixmap(dpi=200)
            page = document.new_page(width=page_pdf[0].rect.width, height=page_pdf[0].rect.height)
            page.insert_image(page.rect, pixmap=pixmap)
        else:
            document.insert_pdf(page_pdf)
    return document.tobytes()
