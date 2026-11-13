"""Synthetic documents for the structure tests: blocks with pages, boxes and heading levels."""

from __future__ import annotations

import uuid

from app.structure.models import DocumentText, SourceBlock

BODY_LINE = 13.0
HEADING_HEIGHTS = {0: 40.0, 1: 22.0, 2: 17.0, 3: 15.0}

_WORDS = (
    "learning representation model signal system energy network structure gradient distribution "
    "feature layer function parameter sample estimate theory method result process variable"
).split()


def sentence(index: int, words: int = 12) -> str:
    chosen = [_WORDS[(index * 7 + i * 3) % len(_WORDS)] for i in range(words)]
    return f"The {' '.join(chosen)} is described in item {index}."


def prose(tokens: int, seed: int = 0) -> str:
    """Several sentences of roughly `tokens` tokens (4 characters per token, the default)."""
    text, index = "", seed
    while len(text) < tokens * 4:
        text += (" " if text else "") + sentence(index)
        index += 1
    return text


class DocBuilder:
    """Builds a `DocumentText`. Blocks go on the current page; `new_page()` advances it."""

    def __init__(self, *, pages: bool = True, file_format: str = "pdf", source_title: str | None = None) -> None:
        self.blocks: list[SourceBlock] = []
        self.pages = pages
        self.page = 1
        self.file_format = file_format
        self.source_title = source_title
        self._y = 700.0

    def new_page(self, count: int = 1) -> DocBuilder:
        self.page += count
        self._y = 700.0
        return self

    def _add(self, text: str, block_type: str, height: float, level: int | None = None, **metadata) -> SourceBlock:
        self._y -= height + 6
        bbox = (72.0, self._y + height, 540.0, self._y) if self.pages else None
        block = SourceBlock(
            sequence=len(self.blocks),
            text=text,
            block_type=block_type,
            page=self.page if self.pages else None,
            bbox=bbox,
            coord_origin="BOTTOMLEFT" if bbox else None,
            heading_level=level,
            id=uuid.uuid4(),
            metadata=dict(metadata),
        )
        self.blocks.append(block)
        return block

    def heading(self, text: str, level: int = 1, height: float | None = None) -> DocBuilder:
        self._add(text, "heading", height or HEADING_HEIGHTS.get(level, 15.0), level)
        return self

    def title(self, text: str) -> DocBuilder:
        return self.heading(text, 0)

    def para(self, text: str, lines: int | None = None) -> DocBuilder:
        lines = lines or max(1, len(text) // 90 + 1)
        self._add(text, "paragraph", BODY_LINE * lines)
        return self

    def prose(self, tokens: int = 120, seed: int = 0) -> DocBuilder:
        return self.para(prose(tokens, seed))

    def plain(self, text: str, height: float = BODY_LINE) -> DocBuilder:
        """A short line typed as a paragraph by the parser (no heading label)."""
        self._add(text, "paragraph", height)
        return self

    def table(self, text: str) -> DocBuilder:
        self._add(text, "table", 60.0, rows=text.count("\n") + 1, columns=2)
        return self

    def equation(self, text: str) -> DocBuilder:
        self._add(text, "equation", 30.0)
        return self

    def caption(self, text: str) -> DocBuilder:
        self._add(text, "caption", BODY_LINE)
        return self

    def item(self, text: str) -> DocBuilder:
        self._add(text, "list", BODY_LINE)
        return self

    def build(self, page_count: int | None = None) -> DocumentText:
        offset = 0
        for block in self.blocks:
            block.start_offset = offset
            offset += len(block.text)
            block.end_offset = offset
            offset += 2
        return DocumentText(
            blocks=self.blocks,
            page_count=(page_count or self.page) if self.pages else None,
            file_format=self.file_format,
            source_title=self.source_title,
        )


def numbered_textbook() -> DocBuilder:
    """Title page, copyright, contents, preface, three numbered chapters with sections, appendix."""
    b = DocBuilder()
    b.title("Foundations of Learning").plain("An Introduction to Representation", 20.0).plain("Jane A. Smith", 16.0)
    b.new_page().para("Copyright © 2024 Example Press. All rights reserved. ISBN 978-3-16-148410-0.")
    b.new_page().heading("Contents")
    for line in ("Preface ........ 5", "1. Introduction ........ 7", "2. Models ........ 20", "3. Learning ........ 40"):
        b.plain(line)
    b.new_page().heading("Preface").prose(150, 1).prose(120, 2)
    b.new_page().heading("1. Introduction").prose(200, 3).heading("1.1 Background", 2).prose(180, 4).prose(150, 5)
    b.heading("1.2 Motivation", 2).prose(160, 6)
    b.new_page().heading("2. Models").prose(120, 7).heading("2.1 Linear Models", 2).prose(220, 8)
    b.heading("2.1.1 Least Squares", 3).prose(150, 9)
    b.new_page().heading("3. Learning").prose(200, 10).heading("3.1 Gradient Methods", 2).prose(210, 11)
    b.new_page().heading("Appendix A — Mathematical Foundations").prose(180, 12)
    b.new_page().heading("References").prose(100, 13)
    return b


def parts_book() -> DocBuilder:
    b = DocBuilder()
    b.title("Decisions Under Uncertainty").plain("Mark Jones and Ana Lopez", 16.0)
    b.new_page().heading("PART I — Foundations", 1, 28.0).prose(60, 1)
    b.heading("Chapter 1: Probability", 2).prose(180, 2).heading("1.1 Events", 3).prose(150, 3)
    b.new_page().heading("Chapter 2: Utility", 2).prose(200, 4)
    b.new_page().heading("PART II — Decision Making", 1, 28.0)
    b.heading("Chapter 3: Choices", 2).prose(210, 5).heading("3.1 Trade-offs", 3).prose(140, 6)
    b.new_page().heading("Part Three: Applications", 1, 28.0)
    b.heading("Chapter 4: Case Studies", 2).prose(190, 7)
    return b


def unnumbered_book() -> DocBuilder:
    b = DocBuilder()
    b.title("How Minds Learn")
    b.new_page().heading("Introduction").prose(180, 1)
    b.new_page().heading("The Foundations of Learning").prose(160, 2).heading("Early Experience", 2).prose(140, 3)
    b.new_page().heading("Understanding Representation").prose(200, 4).heading("Symbols and Signals", 2).prose(120, 5)
    b.new_page().heading("A New Perspective").prose(150, 6)
    b.new_page().heading("Conclusion").prose(130, 7)
    return b


def plain_text_book() -> DocBuilder:
    """No parser heading labels and no table of contents: headings are only bigger paragraphs."""
    b = DocBuilder()
    b.plain("Practical Statistics", 30.0).plain("Dana Whitfield", 16.0)
    b.new_page().plain("CHAPTER 1", 26.0).plain("Describing Data", 26.0).prose(200, 1)
    b.plain("Measures of Centre", 18.0).prose(160, 2)
    b.new_page().plain("CHAPTER 2", 26.0).plain("Probability Basics", 26.0).prose(220, 3)
    b.plain("Conditional Probability", 18.0).prose(150, 4)
    return b


def persist_document(session, doc: DocumentText, *, title: str = "Sample Book", author: str | None = "A. Writer"):
    """Store `doc` as a processed document (completed run + extracted blocks). Returns the Document."""
    from database.models import (
        Book,
        Document,
        ExtractedBlock,
        ProcessingRun,
        ProcessingRunStatus,
        ProcessingStatus,
    )
    from database.models.base import utcnow

    book = Book(title=title, author=author)
    document = Document(
        book=book,
        original_filename="sample.pdf",
        stored_filename=f"{uuid.uuid4()}.pdf",
        file_path=f"original/{uuid.uuid4()}/sample.pdf",
        file_format=doc.file_format,
        mime_type="application/pdf",
        file_size=1234,
        sha256_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        processing_status=ProcessingStatus.COMPLETED.value,
    )
    session.add_all([book, document])
    session.flush()
    summary = {"page_count": doc.page_count, "metadata": {"title": doc.source_title} if doc.source_title else {}}
    session.add(
        ProcessingRun(
            document_id=document.id,
            status=ProcessingRunStatus.COMPLETED.value,
            completed_at=utcnow(),
            text_completed_at=utcnow(),
            result_summary=summary,
        )
    )
    for block in doc.blocks:
        location = None
        if block.page is not None and block.bbox:
            location = {"page": block.page, "bbox": list(block.bbox), "coord_origin": block.coord_origin}
        metadata = dict(block.metadata)
        if block.heading_level is not None:
            metadata["heading_level"] = block.heading_level
        session.add(
            ExtractedBlock(
                id=block.id,
                document_id=document.id,
                sequence_number=block.sequence,
                block_type=block.block_type,
                text=block.text,
                page_number=block.page,
                source_location=location,
                block_metadata=metadata or None,
            )
        )
    session.commit()
    return document


def technical_book() -> DocBuilder:
    """Four levels of decimal numbering."""
    b = DocBuilder()
    b.title("Signals and Systems in Practice")
    b.new_page().heading("1. Fundamentals").prose(120, 1)
    b.heading("1.1 Signals", 2).prose(100, 2)
    b.heading("1.1.1 Continuous Signals", 3).prose(100, 3)
    b.heading("1.1.1.1 Sampling", 4).prose(90, 4)
    b.heading("1.1.2 Discrete Signals", 3).prose(100, 5)
    b.heading("1.2 Systems", 2).prose(100, 6)
    b.new_page().heading("2. Transforms").prose(110, 7)
    b.heading("2.1 Fourier Transform", 2).prose(100, 8)
    return b


def front_and_back_matter_book() -> DocBuilder:
    b = DocBuilder()
    b.new_page()  # a blank cover page comes first
    b.title("Quiet Machines")
    b.plain("Notes on Computation", 20.0)
    b.new_page().para("© 2023 Open Books. All rights reserved. Printed in Canada.")
    b.new_page().heading("Foreword").prose(150, 1)
    b.new_page().heading("Preface").prose(150, 2)
    b.new_page().heading("Introduction").prose(180, 3)
    b.new_page().heading("Chapter 1", 1).heading("Beginnings", 1).prose(200, 4)
    b.new_page().heading("Chapter 2", 1).heading("Middles", 1).prose(200, 5)
    b.new_page().heading("Appendix A — Proofs", 1).prose(150, 6)
    b.new_page().heading("Appendix B — Data", 1).prose(120, 7)
    b.new_page().heading("Notes", 1).prose(100, 8)
    b.new_page().heading("Glossary", 1).prose(100, 9)
    b.new_page().heading("Bibliography", 1).prose(100, 10)
    return b


def document_from_parsing(result) -> DocumentText:
    """A `DocumentText` from a Day 3 `ParsingResult` (normalised blocks), as the service builds it."""
    blocks, offset = [], 0
    for index, block in enumerate(result.blocks):
        location = block.source_location or {}
        raw = location.get("bbox")
        metadata = block.metadata or {}
        spanned = [p for p in location.get("pages", []) if isinstance(p, int)]
        blocks.append(
            SourceBlock(
                sequence=index,
                text=block.text,
                block_type=block.block_type.value,
                page=block.page_number,
                last_page=max(spanned) if spanned and block.page_number is not None else None,
                bbox=tuple(raw) if isinstance(raw, list) and len(raw) == 4 else None,
                coord_origin=location.get("coord_origin"),
                heading_level=metadata.get("heading_level"),
                id=uuid.uuid4(),
                start_offset=offset,
                end_offset=offset + len(block.text),
                metadata=metadata,
            )
        )
        offset += len(block.text) + 2
    return DocumentText(blocks=blocks, page_count=result.page_count, file_format=result.file_format)


def content_blocks(document: DocumentText):
    """The `ContentBlock`s a Day 3 parser would have produced for `document` (for stub parsers)."""
    from app.parsing.models import BlockType, ContentBlock

    result = []
    for block in document.blocks:
        location = None
        if block.page is not None and block.bbox:
            location = {"page": block.page, "bbox": list(block.bbox), "coord_origin": block.coord_origin}
        metadata = dict(block.metadata)
        if block.heading_level is not None:
            metadata["heading_level"] = block.heading_level
        result.append(
            ContentBlock(
                text=block.text,
                block_type=BlockType(block.block_type),
                page_number=block.page,
                source_location=location,
                sequence_number=block.sequence,
                metadata=metadata or None,
            )
        )
    return result
