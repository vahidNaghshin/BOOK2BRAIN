"""Word equation (OMML) helpers shared by the DOCX text parser and the equation extractor."""

import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from lxml import etree

from app.parsing.models import ParsingIssue

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"m": MATH_NS, "w": WORD_NS}
_MAX_DOCUMENT_XML = 200 * 1024 * 1024
_XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def w(name: str) -> str:
    return f"{{{WORD_NS}}}{name}"


def m(name: str) -> str:
    return f"{{{MATH_NS}}}{name}"


def read_document_xml(path: Path) -> etree._Element:
    """Parse word/document.xml safely (no entity expansion, no network, bounded size)."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("word/document.xml")
        if info.file_size > _MAX_DOCUMENT_XML:
            raise ValueError("word/document.xml is unreasonably large")
        data = archive.read("word/document.xml")
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    return etree.fromstring(data, parser)


def omml_to_latex(element: etree._Element) -> str:
    """Convert one <m:oMath> with Docling's converter. Raises on structures it cannot handle."""
    from docling.backend.docx.latex.omml import oMath2Latex

    return str(oMath2Latex(element)).strip()


def top_level_math(paragraph: etree._Element) -> list[etree._Element]:
    return paragraph.xpath(".//m:oMath[not(ancestor::m:oMath)]", namespaces=NS)


def math_text(element: etree._Element) -> str:
    return "".join(element.xpath(".//m:t/text()", namespaces=NS))


@contextmanager
def readable_docx(path: Path) -> Iterator[tuple[Path, list[ParsingIssue]]]:
    """Yield a DOCX that Docling can read even when some equations are malformed.

    Docling's DOCX backend fails the *whole document* on one unconvertible equation. Here such
    equations are replaced, in a temporary copy only, by their plain text, so the document's
    ordinary text is still extracted. The original file is never modified.
    """
    try:
        root = read_document_xml(path)
    except (zipfile.BadZipFile, KeyError, etree.XMLSyntaxError, ValueError):
        yield path, []  # not diagnosed here: the text extraction will report it
        return

    broken: list[tuple[etree._Element, int]] = []
    for paragraph_index, paragraph in enumerate(root.iter(w("p"))):
        for element in top_level_math(paragraph):
            try:
                omml_to_latex(element)
            except Exception:  # Docling's converter raises KeyError and others on odd structures
                broken.append((element, paragraph_index))
    if not broken:
        yield path, []
        return

    issues: list[ParsingIssue] = []
    for element, paragraph_index in broken:
        text = math_text(element)
        target = element.getparent() if element.getparent().tag == m("oMathPara") else element
        run = etree.Element(w("r"))
        etree.SubElement(run, w("t"), {_XML_SPACE: "preserve"}).text = text
        target.getparent().replace(target, run)
        issues.append(
            ParsingIssue(
                code="equation_text_only",
                message=(
                    f"A Word equation in paragraph {paragraph_index + 1} could not be converted; "
                    "its text was kept as plain text so the rest of the document still extracts."
                ),
                details={"paragraph_index": paragraph_index, "text": text[:200]},
            )
        )

    with tempfile.TemporaryDirectory(prefix="docx-clean-") as directory:
        cleaned = Path(directory) / "document.docx"
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(cleaned, "w", zipfile.ZIP_DEFLATED) as target_zip:
            for entry in source.infolist():
                data = (
                    etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
                    if entry.filename == "word/document.xml"
                    else source.read(entry.filename)
                )
                target_zip.writestr(entry, data)
        yield cleaned, issues
