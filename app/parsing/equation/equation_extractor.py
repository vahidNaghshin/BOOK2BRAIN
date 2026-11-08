"""Equation extraction from structured sources: EPUB (MathML, LaTeX, images) and DOCX (OMML).

These routes use the document's own markup, so they are authoritative and need no model.
PDF (layout + recognition) is handled in `pdf_equations.py`.
"""

import logging
import re
import uuid
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag
from lxml import etree

from app.parsing.epub_parser import (
    _NAVIGATION_EPUB_TYPES,
    _NOISE_TAGS,
    _epub_types,
    _flatten_toc,
    _tag_text,
)
from app.parsing.equation.equation_detector import find_delimited_math, image_candidate
from app.parsing.equation.equation_models import (
    EquationBlock,
    EquationIssue,
    EquationType,
    ExtractionMethod,
    ValidationStatus,
)
from app.parsing.equation.mathml_converter import convert_mathml
from app.parsing.omml import (
    NS as _NS,
    m,
    math_text,
    omml_to_latex,
    read_document_xml,
    top_level_math,
    w,
)

logger = logging.getLogger(__name__)

_MARK_OPEN, _MARK_CLOSE = "\ue001", "\ue002"
_MARKER = re.compile(_MARK_OPEN + r"(\d+)" + _MARK_CLOSE)
_NUMBER_ONLY = re.compile(r"^\(\s*(\d+(?:\s*[.\-]\s*\d+)*\s*[a-z]?)\s*\)$")
_MAX_ORIGINAL = 20000
_UNIT_TAGS = {"p", "li", "dd", "dt", "td", "th", "blockquote", "figcaption", "caption", "div",
              "h1", "h2", "h3", "h4", "h5", "h6", "summary"}  # fmt: skip


@dataclass
class ExtractedEquations:
    """Equations from a structured source, plus problems and images that still need OCR."""

    equations: list[EquationBlock] = field(default_factory=list)
    issues: list[EquationIssue] = field(default_factory=list)
    # Equation id -> image bytes, for equations that exist only as pictures (EPUB).
    images: dict[uuid.UUID, bytes] = field(default_factory=dict)


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_ORIGINAL else text[:_MAX_ORIGINAL]


# ======================================================================================
# EPUB
# ======================================================================================


class EpubEquationExtractor:
    """Reads MathML, TeX annotations, delimited LaTeX and equation images from spine items."""

    def extract(self, path: Path, document_id: uuid.UUID) -> ExtractedEquations:
        import ebooklib
        from ebooklib import epub

        result = ExtractedEquations()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                book = epub.read_epub(str(path), options={"ignore_ncx": True})
        except Exception as exc:  # EbookLib raises many unrelated types for bad files
            raise ValueError(f"The EPUB could not be read for equations: {type(exc).__name__}") from exc

        toc_titles: dict[str, str] = {}
        for href, title in _flatten_toc(book.toc):
            toc_titles.setdefault(href, title)

        for spine_index, (idref, linear) in enumerate(book.spine):
            item = book.get_item_with_id(idref)
            if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            location: dict[str, Any] = {
                "spine_index": spine_index,
                "item_id": idref,
                "href": item.get_name(),
                "chapter_title": toc_titles.get(item.get_name()),
            }
            if str(linear).lower() == "no":
                location["linear"] = False
            try:
                self._extract_item(book, item, document_id, location, result)
            except Exception as exc:  # one broken chapter must not lose the rest
                logger.warning("Equation extraction failed for %s: %s", idref, exc)
                result.issues.append(
                    EquationIssue(
                        code="chapter_equations_failed",
                        message=f"Equations in '{item.get_name()}' could not be read: {type(exc).__name__}.",
                        details={"item_id": idref},
                    )
                )
        return result

    def _extract_item(
        self, book: Any, item: Any, document_id: uuid.UUID, location: dict[str, Any], out: ExtractedEquations
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            soup = BeautifulSoup(item.get_content(), "lxml")
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()
        for tag in soup.find_all(attrs={"epub:type": True}):
            if _epub_types(tag) & _NAVIGATION_EPUB_TYPES:
                tag.decompose()
        root = soup.body or soup

        # The text of each unit as the text extraction sees it, before math is replaced.
        unit_hints = {id(unit): _tag_text(unit) for unit in root.find_all(_UNIT_TAGS)}
        unit_hints[id(root)] = _tag_text(root) if not root.find(_UNIT_TAGS) else ""

        slots: list[EquationBlock] = []

        def owner(node: Tag | NavigableString) -> Tag:
            for parent in node.parents:
                if isinstance(parent, Tag) and (parent.name in _UNIT_TAGS or parent is root):
                    return parent
            return root

        # Math elements and equation images become markers, so reading order is preserved.
        for math in list(root.find_all("math")):
            equation = self._from_mathml(math, document_id, location)
            equation.link_hint = unit_hints.get(id(owner(math)))
            slots.append(equation)
            math.replace_with(NavigableString(f"{_MARK_OPEN}{len(slots) - 1}{_MARK_CLOSE}"))
        for img in list(root.find_all("img")):
            candidate = image_candidate(img)
            if candidate is None:
                continue
            equation = self._from_image(book, img, candidate, document_id, location, out)
            equation.link_hint = unit_hints.get(id(owner(img)))
            slots.append(equation)
            img.replace_with(NavigableString(f"{_MARK_OPEN}{len(slots) - 1}{_MARK_CLOSE}"))

        units = [*root.find_all(_UNIT_TAGS)]
        if root.name == "body" or not units:
            units.append(root)
        for unit in units:
            text = self._own_text(unit, owner)
            self._emit_unit(unit, text, slots, document_id, location, unit_hints, out)

    @staticmethod
    def _own_text(unit: Tag, owner: Any) -> str:
        """Text directly belonging to this unit (not to nested units), markers included."""
        parts = [
            str(node)
            for node in unit.descendants
            if isinstance(node, NavigableString) and type(node) is NavigableString and owner(node) is unit
        ]
        return re.sub(r"\s+", " ", "".join(parts)).strip()

    def _emit_unit(
        self,
        unit: Tag,
        text: str,
        slots: list[EquationBlock],
        document_id: uuid.UUID,
        location: dict[str, Any],
        unit_hints: dict[int, str],
        out: ExtractedEquations,
    ) -> None:
        events: list[tuple[int, str, Any]] = []
        removed = 0
        for marker in _MARKER.finditer(text):
            # Position of the marker in the text with all markers removed.
            events.append((marker.start() - removed, "slot", int(marker.group(1))))
            removed += marker.end() - marker.start()
        plain = _MARKER.sub("", text)
        for found in find_delimited_math(plain):
            events.append((found.start, "tex", found))
        if not events:
            return
        events.sort(key=lambda event: event[0])

        remainder = _MARKER.sub("", text)
        for _, kind, payload in events:
            if kind == "tex":
                remainder = remainder.replace(plain[payload.start : payload.end], "", 1)
        remainder = remainder.strip()
        number = _NUMBER_ONLY.match(remainder)
        stands_alone = remainder == "" or bool(number)
        hint = unit_hints.get(id(unit))

        readable = _MARKER.sub(lambda m: "$" + slots[int(m.group(1))].latex + "$", text)
        for _, kind, payload in events:
            equation = slots[payload] if kind == "slot" else self._from_delimiters(payload, document_id, location)
            if kind == "tex":
                equation.link_hint = hint
            display = equation.metadata.pop("_display", False) or (stands_alone and len(events) == 1)
            equation.equation_type = EquationType.DISPLAY if display else EquationType.INLINE
            if display and number:
                equation.equation_number = re.sub(r"\s+", "", number.group(1))
                equation.equation_type = EquationType.NUMBERED
            if not display:
                equation.surrounding_text = readable
            out.equations.append(equation)

    def _from_mathml(self, math: Tag, document_id: uuid.UUID, location: dict[str, Any]) -> EquationBlock:
        conversion = convert_mathml(math)
        method = ExtractionMethod.TEX_ANNOTATION if conversion.tex_annotation else ExtractionMethod.MATHML
        metadata: dict[str, Any] = {"original_mathml": _clip(str(math)), "_display": conversion.display}
        if conversion.tex_annotation:
            metadata["source"] = "tex annotation inside MathML"
        equation = EquationBlock(
            document_id=document_id,
            latex=conversion.latex,
            original_latex=conversion.latex,
            plain_text=conversion.alttext or math.get_text(" ", strip=True) or None,
            extraction_method=method,
            source_location={**location, "kind": "mathml"},
            metadata=metadata,
        )
        if conversion.warnings:
            equation.validation_warnings.extend(conversion.warnings)
            equation.metadata["converter_warnings"] = True
        return equation

    def _from_delimiters(self, found: Any, document_id: uuid.UUID, location: dict[str, Any]) -> EquationBlock:
        return EquationBlock(
            document_id=document_id,
            latex=found.latex,
            original_latex=found.latex,
            extraction_method=ExtractionMethod.LATEX_DELIMITERS,
            source_location={**location, "kind": "latex", "delimiter": found.delimiter},
            metadata={"delimiter": found.delimiter, "_display": found.display},
        )

    def _from_image(
        self,
        book: Any,
        img: Tag,
        candidate: Any,
        document_id: uuid.UUID,
        location: dict[str, Any],
        out: ExtractedEquations,
    ) -> EquationBlock:
        src = str(img.get("src", ""))
        metadata: dict[str, Any] = {"image_src": src, "detected_because": candidate.reason}
        if candidate.alt_latex:
            return EquationBlock(
                document_id=document_id,
                latex=candidate.alt_latex,
                original_latex=candidate.alt_latex,
                extraction_method=ExtractionMethod.IMAGE_ALT_TEXT,
                source_location={**location, "kind": "image", "image_src": src},
                metadata=metadata,
            )
        equation = EquationBlock(
            document_id=document_id,
            latex="",
            extraction_method=ExtractionMethod.MATH_OCR,
            source_location={**location, "kind": "image", "image_src": src},
            metadata={**metadata, "needs_math_ocr": True},
        )
        image_item = self._find_image(book, location["href"], src)
        if image_item is not None:
            out.images[equation.id] = image_item.get_content()
        else:
            equation.validation_warnings.append(f"The image '{src}' was not found in the EPUB.")
            equation.validation_status = ValidationStatus.INVALID
        return equation

    @staticmethod
    def _find_image(book: Any, document_href: str, src: str) -> Any:
        if not src or src.startswith(("http:", "https:", "data:")):
            return None
        base = document_href.rsplit("/", 1)[0] if "/" in document_href else ""
        candidates = [src.lstrip("/"), f"{base}/{src}".lstrip("/")]
        for candidate in candidates:
            normalized = str(Path(candidate)) if ".." not in candidate else re.sub(r"[^/]+/\.\./", "", candidate)
            item = book.get_item_with_href(normalized)
            if item is not None:
                return item
        return None


# ======================================================================================
# DOCX
# ======================================================================================

_HEADING_STYLE = re.compile(r"^(?:Heading\d*|Title|Subtitle)$", re.I)

_w, _m = w, m  # short aliases used throughout this section
_read_document_xml, _top_level_math, _math_text = read_document_xml, top_level_math, math_text


class DocxEquationExtractor:
    """Reads Word equations (OMML). The XML says which are display equations (m:oMathPara)."""

    def extract(self, path: Path, document_id: uuid.UUID) -> ExtractedEquations:
        result = ExtractedEquations()
        try:
            root = _read_document_xml(path)
        except (zipfile.BadZipFile, KeyError, etree.XMLSyntaxError, ValueError) as exc:
            raise ValueError(f"The DOCX could not be read for equations: {type(exc).__name__}") from exc

        section: str | None = None
        for paragraph_index, paragraph in enumerate(root.iter(_w("p"))):
            style = paragraph.find("w:pPr/w:pStyle", _NS)
            style_name = style.get(_w("val")) if style is not None else None
            maths = _top_level_math(paragraph)
            if not maths:
                if style_name and _HEADING_STYLE.match(style_name):
                    section = "".join(paragraph.xpath(".//w:t/text()", namespaces=_NS)).strip() or section
                continue

            converted = [self._convert(el, document_id, paragraph_index, index, section, result)
                         for index, el in enumerate(maths)]
            plain_text_parts = paragraph.xpath(".//w:t[not(ancestor::m:oMath)]/text()", namespaces=_NS)
            text = "".join(plain_text_parts).strip()
            number = _NUMBER_ONLY.match(text)
            readable = self._readable(paragraph, converted, maths)
            for equation, element in zip(converted, maths):
                display = element.getparent().tag == _m("oMathPara")
                equation.equation_type = EquationType.DISPLAY if display else EquationType.INLINE
                if display and number:
                    equation.equation_number = re.sub(r"\s+", "", number.group(1))
                    equation.equation_type = EquationType.NUMBERED
                if not display:
                    equation.surrounding_text = readable
                result.equations.append(equation)
        return result

    @staticmethod
    def _readable(paragraph: etree._Element, converted: list[EquationBlock], maths: list[etree._Element]) -> str:
        """Paragraph text with equations written inline as $latex$, in order."""
        by_element = {id(element): equation for element, equation in zip(maths, converted)}
        parts: list[str] = []

        def visit(node: etree._Element) -> None:
            for child in node:
                if child.tag == _m("oMath") and id(child) in by_element:
                    parts.append("$" + by_element[id(child)].latex + "$")
                elif child.tag == _w("t"):
                    parts.append(child.text or "")
                elif child.tag in (_m("oMath"),):
                    continue
                else:
                    visit(child)

        visit(paragraph)
        return re.sub(r"\s+", " ", "".join(parts)).strip()

    def _convert(
        self,
        element: etree._Element,
        document_id: uuid.UUID,
        paragraph_index: int,
        math_index: int,
        section: str | None,
        out: ExtractedEquations,
    ) -> EquationBlock:
        original = _clip(etree.tostring(element, encoding="unicode"))
        location = {
            "kind": "omml",
            "part": "word/document.xml",
            "paragraph_index": paragraph_index,
            "math_index": math_index,
        }
        plain = _math_text(element) or None
        try:
            latex = omml_to_latex(element)
        except Exception as exc:  # Docling's converter raises KeyError etc. on odd structures
            logger.warning("OMML equation %d.%d is not convertible: %s", paragraph_index, math_index, exc)
            out.issues.append(
                EquationIssue(
                    code="unsupported_omml",
                    message=(
                        f"A Word equation (paragraph {paragraph_index + 1}) uses a structure that "
                        f"could not be converted ({type(exc).__name__}). Its original OMML is kept."
                    ),
                    details={"paragraph_index": paragraph_index, "math_index": math_index},
                )
            )
            return EquationBlock(
                document_id=document_id,
                latex="",
                plain_text=plain,
                extraction_method=ExtractionMethod.OMML,
                validation_status=ValidationStatus.INVALID,
                validation_warnings=[f"The OMML structure could not be converted to LaTeX ({type(exc).__name__})."],
                section_title=section,
                source_location=location,
                metadata={"original_omml": original, "conversion_failed": True},
            )
        return EquationBlock(
            document_id=document_id,
            latex=latex,
            original_latex=latex,
            plain_text=plain,
            extraction_method=ExtractionMethod.OMML,
            section_title=section,
            source_location=location,
            metadata={"original_omml": original},
            link_hint=latex,
        )
