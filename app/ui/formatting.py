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
