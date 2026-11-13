from datetime import datetime, timezone


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def format_pages(start: int | None, end: int | None, status: str = "exact") -> str:
    """Page reference for display. Unknown pages are said to be unknown, never guessed."""
    if start is None:
        return "page unknown"
    label = f"p. {start}" if end in (None, start) else f"pp. {start}–{end}"
    return f"{label} (partly known)" if status == "partial" else label


_NODE_TYPE_LABELS = {
    "book_title": "Book title",
    "front_matter": "Front matter",
    "back_matter": "Back matter",
    "acknowledgement": "Acknowledgements",
    "other_heading": "Other heading",
    "table_of_contents": "Table of contents",
    "about_author": "About the author",
}


def node_type_label(node_type: str) -> str:
    return _NODE_TYPE_LABELS.get(node_type, node_type.replace("_", " ").capitalize())


STAGE_LABELS = {
    "document_registration": "Registering the document",
    "parsing": "Parsing the document",
    "ocr": "Running OCR on scanned pages",
    "normalizing": "Normalizing text",
    "text_extraction": "Saving the extracted text",
    "equation_extraction": "Extracting equations",
    "structural_detection": "Detecting structure",
    "paragraph_identification": "Identifying paragraphs",
    "passage_segmentation": "Generating passages",
    "persistence": "Saving the structure",
    "validation": "Validating the data",
    "completed": "Completed",
}


def stage_label(stage: str | None) -> str:
    return STAGE_LABELS.get(stage or "", (stage or "").replace("_", " ").capitalize())


def status_label(status: str, stage: str | None = None, *, incomplete: bool = False) -> str:
    """One line a user can read: what the document is doing, or where it failed."""
    if status == "uploaded":
        return "Uploaded, not processed"
    if status == "pending":
        return "Queued"
    if status == "processing":
        return f"Processing: {stage_label(stage).lower()}" if stage else "Processing"
    if status == "failed":
        return f"Failed during {stage_label(stage).lower()}" if stage else "Failed"
    if status == "completed":
        return "Completed with warnings (some pages unreadable)" if incomplete else "Completed"
    return status
