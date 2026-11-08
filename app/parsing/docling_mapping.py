"""Map a DoclingDocument to ContentBlocks, in reading order, with source locations."""

from typing import Any

from app.parsing.models import BlockType, ContentBlock

_MAX_SECTION_TITLE = 1000


def _table_text(table: Any) -> str:
    rows = []
    for row in table.data.grid:
        cells = [cell.text.strip() for cell in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _location(item: Any, *, index: int, with_pages: bool) -> tuple[int | None, dict[str, Any]]:
    location: dict[str, Any] = {"docling_ref": item.self_ref, "element_index": index}
    provenance = getattr(item, "prov", None) or []
    if not with_pages or not provenance:
        return None, location

    first = provenance[0]
    page = int(first.page_no)
    location["page"] = page
    bbox = first.bbox
    location["bbox"] = [round(float(v), 2) for v in (bbox.l, bbox.t, bbox.r, bbox.b)]
    location["coord_origin"] = getattr(bbox.coord_origin, "value", str(bbox.coord_origin))
    pages = sorted({int(p.page_no) for p in provenance})
    if len(pages) > 1:
        location["pages"] = pages
    return page, location


def blocks_from_docling(document: Any, *, with_pages: bool) -> list[ContentBlock]:
    """Convert body items of a DoclingDocument into blocks.

    Page headers and footers are not in the body layer, so they are naturally excluded.
    Pictures carry no text and are skipped (their captions come through as caption blocks).
    """
    from docling_core.types.doc import ListItem, SectionHeaderItem, TableItem, TitleItem

    blocks: list[ContentBlock] = []
    section_title: str | None = None

    for index, (item, _depth) in enumerate(document.iterate_items()):
        metadata: dict[str, Any] = {}
        block_type = BlockType.PARAGRAPH

        if isinstance(item, TableItem):
            text = _table_text(item)
            block_type = BlockType.TABLE
            grid = item.data.grid
            metadata = {"rows": len(grid), "columns": len(grid[0]) if grid else 0}
        else:
            text = getattr(item, "text", None)
            label = getattr(getattr(item, "label", None), "value", "")
            if isinstance(item, TitleItem):
                block_type = BlockType.HEADING
                metadata = {"heading_level": 0}
            elif isinstance(item, SectionHeaderItem):
                block_type = BlockType.HEADING
                metadata = {"heading_level": int(item.level)}
            elif isinstance(item, ListItem):
                block_type = BlockType.LIST
                metadata = {"enumerated": bool(item.enumerated)}
                if item.marker:
                    metadata["marker"] = item.marker
            elif label == "caption":
                block_type = BlockType.CAPTION
            elif label in ("text", "paragraph"):
                block_type = BlockType.PARAGRAPH
            else:
                block_type = BlockType.OTHER
                metadata = {"docling_label": label}
                if label == "code":
                    metadata["preformatted"] = True

        if not text or not str(text).strip():
            continue

        if block_type is BlockType.HEADING:
            section_title = str(text).strip()[:_MAX_SECTION_TITLE]

        page, location = _location(item, index=index, with_pages=with_pages)
        blocks.append(
            ContentBlock(
                text=str(text),
                block_type=block_type,
                page_number=page,
                section_title=section_title,
                source_location=location,
                metadata=metadata or None,
            )
        )
    return blocks
