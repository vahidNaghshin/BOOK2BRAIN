import logging
import re
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from app.parsing.base import DocumentParser, Extraction, ProgressCallback
from app.parsing.errors import CorruptDocumentError
from app.parsing.models import BlockType, ContentBlock, ParseStage, ParsingIssue

logger = logging.getLogger(__name__)

_LINE_BREAK = "\ue000"  # private-use marker for <br>, so it survives whitespace collapsing
_HEADINGS = {f"h{level}": level for level in range(1, 7)}
_NOISE_TAGS = ("script", "style", "noscript", "svg", "nav", "head", "template")
_NAVIGATION_EPUB_TYPES = {"toc", "landmarks", "page-list", "lot", "loi"}
_BLOCK_LEVEL = set(_HEADINGS) | {
    "p", "pre", "table", "ul", "ol", "dl", "li", "dd", "dt", "blockquote", "div", "section",
    "article", "aside", "figure", "figcaption", "caption", "header", "footer", "main", "body",
    "html", "hr", "form", "fieldset", "details", "summary",
}  # fmt: skip
_MAX_SECTION_TITLE = 1000


def _text_parts(node: Tag, skip: frozenset[str] = frozenset()) -> Iterator[str]:
    for child in node.children:
        if isinstance(child, Tag):
            if child.name in skip:
                continue
            if child.name == "br":
                yield _LINE_BREAK
            else:
                yield from _text_parts(child, skip)
        elif type(child) is NavigableString:  # excludes comments, CDATA, doctype
            yield str(child)


def _clean_text(parts: Iterator[str]) -> str:
    """Collapse source whitespace like a browser would, keeping explicit <br> line breaks."""
    text = re.sub(r"\s+", " ", "".join(parts))
    text = text.replace(f" {_LINE_BREAK}", _LINE_BREAK).replace(f"{_LINE_BREAK} ", _LINE_BREAK)
    return text.replace(_LINE_BREAK, "\n").strip()


def _tag_text(tag: Tag, skip: frozenset[str] = frozenset()) -> str:
    return _clean_text(_text_parts(tag, skip))


def _epub_types(tag: Tag) -> set[str]:
    value = tag.get("epub:type") or ""
    return set(str(value).split())


def _has_block_children(tag: Tag) -> bool:
    return tag.find(list(_BLOCK_LEVEL)) is not None


class _ItemExtractor:
    """Turns one spine item (an XHTML document) into blocks."""

    def __init__(self, location: dict[str, Any], chapter_title: str | None) -> None:
        self.location = location
        self.chapter_title = chapter_title
        self.section_title = chapter_title
        self.blocks: list[ContentBlock] = []

    def run(self, soup: BeautifulSoup) -> list[ContentBlock]:
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()
        for tag in soup.find_all(attrs={"epub:type": True}):
            if _epub_types(tag) & _NAVIGATION_EPUB_TYPES:
                tag.decompose()
        root = soup.body or soup
        self._walk(root)

        # Give every block the chapter title, now that the first heading is known.
        if self.chapter_title is None:
            first_heading = next((b.text for b in self.blocks if b.block_type is BlockType.HEADING), None)
            for block in self.blocks:
                block.source_location["chapter_title"] = first_heading  # type: ignore[index]
        return self.blocks

    def _emit(self, text: str, block_type: BlockType, tag: Tag, **metadata: Any) -> None:
        if not text.strip():
            return
        epub_types = _epub_types(tag)
        if epub_types:
            metadata["epub_type"] = sorted(epub_types)
        if block_type is BlockType.HEADING:
            self.section_title = text[:_MAX_SECTION_TITLE]
        location = dict(self.location)
        location["chapter_title"] = self.chapter_title
        location["element_index"] = len(self.blocks)
        self.blocks.append(
            ContentBlock(
                text=text,
                block_type=block_type,
                page_number=None,  # EPUB has no reliable pages
                section_title=self.section_title,
                source_location=location,
                metadata=metadata or None,
            )
        )

    def _walk(self, node: Tag) -> None:
        inline_buffer: list[str] = []

        def flush() -> None:
            text = _clean_text(iter(inline_buffer))
            inline_buffer.clear()
            if text:
                self._emit(text, BlockType.PARAGRAPH, node)

        for child in node.children:
            if isinstance(child, Tag) and child.name in _BLOCK_LEVEL:
                flush()
                self._block(child)
            elif isinstance(child, Tag):
                inline_buffer.extend(_text_parts(child) if child.name != "br" else [_LINE_BREAK])
            elif type(child) is NavigableString:
                inline_buffer.append(str(child))
        flush()

    def _block(self, tag: Tag) -> None:
        name = tag.name
        if name in _HEADINGS:
            self._emit(_tag_text(tag), BlockType.HEADING, tag, heading_level=_HEADINGS[name])
        elif name == "p" or name in ("dt", "dd", "summary"):
            self._emit(_tag_text(tag), BlockType.PARAGRAPH, tag)
        elif name == "table":
            self._table(tag)
        elif name == "pre":
            self._emit(_pre_text(tag), BlockType.OTHER, tag, preformatted=True)
        elif name in ("figcaption", "caption"):
            self._emit(_tag_text(tag), BlockType.CAPTION, tag)
        elif name == "li":
            self._list_item(tag)
        elif name == "hr":
            return
        elif _has_block_children(tag):
            self._walk(tag)
        else:
            self._emit(_tag_text(tag), BlockType.PARAGRAPH, tag)

    def _list_item(self, tag: Tag) -> None:
        parent = tag.parent.name if isinstance(tag.parent, Tag) else ""
        self._emit(
            _tag_text(tag, skip=frozenset({"ul", "ol"})),
            BlockType.LIST,
            tag,
            enumerated=parent == "ol",
        )
        for nested in tag.find_all(["ul", "ol"], recursive=False):
            self._walk(nested)

    def _table(self, tag: Tag) -> None:
        rows = []
        for row in tag.find_all("tr"):
            cells = [_tag_text(cell) for cell in row.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            columns = max(row.count(" | ") + 1 for row in rows)
            self._emit("\n".join(rows), BlockType.TABLE, tag, rows=len(rows), columns=columns)


def _pre_text(tag: Tag) -> str:
    return "".join(
        "\n" if isinstance(part, Tag) and part.name == "br" else str(part)
        for part in tag.strings
    ).strip("\n")


def _flatten_toc(entries: Any) -> Iterator[tuple[str, str]]:
    """Yield (href without fragment, title) for every entry in an ebooklib TOC."""
    for entry in entries:
        if isinstance(entry, (tuple, list)):
            section, children = entry[0], entry[1] if len(entry) > 1 else []
            href = getattr(section, "href", None)
            title = getattr(section, "title", None)
            if href and title:
                yield href.split("#")[0], str(title)
            yield from _flatten_toc(children)
        else:
            href = getattr(entry, "href", None)
            title = getattr(entry, "title", None)
            if href and title:
                yield href.split("#")[0], str(title)


class EpubParser(DocumentParser):
    """EPUB via EbookLib: spine items in reading order, one source reference per chapter."""

    file_format = "epub"
    extraction_method = "ebooklib"

    def _extract(self, path: Path, on_progress: ProgressCallback) -> Extraction:
        import ebooklib
        from ebooklib import epub

        on_progress(ParseStage.PARSING, 0, 1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                book = epub.read_epub(str(path), options={"ignore_ncx": True})
        except Exception as exc:  # EbookLib raises many unrelated types for bad files
            logger.warning("EPUB %s could not be read: %s: %s", path.name, type(exc).__name__, exc)
            raise CorruptDocumentError(
                "The EPUB file is corrupted, malformed or not supported.",
                details={"exception": type(exc).__name__, "reason": str(exc)[:500]},
            ) from exc

        toc_titles: dict[str, str] = {}
        for href, title in _flatten_toc(book.toc):
            toc_titles.setdefault(href, title)

        spine = list(book.spine)
        issues: list[ParsingIssue] = []
        blocks: list[ContentBlock] = []
        documents_with_text = 0

        for spine_index, (idref, linear) in enumerate(spine):
            on_progress(ParseStage.PARSING, spine_index, len(spine))
            item = book.get_item_with_id(idref)
            if item is None:
                issues.append(
                    ParsingIssue(
                        code="spine_item_missing",
                        message=f"Spine entry '{idref}' points to a missing file and was skipped.",
                        details={"item_id": idref},
                    )
                )
                continue
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue  # navigation documents and non-text items

            location: dict[str, Any] = {
                "spine_index": spine_index,
                "item_id": idref,
                "href": item.get_name(),
            }
            if str(linear).lower() == "no":
                location["linear"] = False

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    soup = BeautifulSoup(item.get_content(), "lxml")
                item_blocks = _ItemExtractor(location, toc_titles.get(item.get_name())).run(soup)
            except Exception as exc:  # one broken chapter must not lose the whole book
                logger.warning("EPUB chapter %s failed: %s: %s", idref, type(exc).__name__, exc)
                issues.append(
                    ParsingIssue(
                        code="chapter_parse_failed",
                        message=f"Chapter '{item.get_name()}' could not be parsed and was skipped.",
                        details={"item_id": idref, "exception": type(exc).__name__},
                    )
                )
                continue

            if item_blocks:
                documents_with_text += 1
                blocks.extend(item_blocks)

        on_progress(ParseStage.PARSING, len(spine), len(spine))

        titles = book.get_metadata("DC", "title")
        languages = book.get_metadata("DC", "language")
        return Extraction(
            blocks=blocks,
            page_count=None,
            warnings=issues,
            partial=any(issue.code == "chapter_parse_failed" for issue in issues),
            metadata={
                "title": titles[0][0] if titles else None,
                "language": languages[0][0] if languages else None,
                "spine_items": len(spine),
                "spine_items_with_text": documents_with_text,
            },
        )
