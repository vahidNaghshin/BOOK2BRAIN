"""Deterministic text normalization shared by all parsers.

The goal is clean text without changing the author's words: it fixes encoding and
whitespace artifacts only. It never merges paragraphs, never drops punctuation,
numbering, citations or table content, and is idempotent.
"""

import re
import unicodedata

from app.parsing.models import ContentBlock, ParsingIssue

_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}
# Soft hyphen, zero-width space, word joiner, BOM: invisible and never meaningful in prose.
# (Zero-width joiner/non-joiner are kept: they matter in several scripts.)
_INVISIBLE = "\u00ad\u200b\u2060\ufeff"
_UNICODE_SPACES = "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
_LINE_BREAKS = {"\x0b": "\n", "\x0c": "\n", "\x85": "\n", "\u2028": "\n", "\u2029": "\n\n"}

_TRANSLATION = {ord(char): None for char in _INVISIBLE}
_TRANSLATION.update({ord(char): " " for char in _UNICODE_SPACES})
_TRANSLATION.update({ord(char): replacement for char, replacement in _LINE_BREAKS.items()})
_TRANSLATION.update({ord(char): replacement for char, replacement in _LIGATURES.items()})

_HORIZONTAL_SPACE = re.compile(r"[ \t]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def _is_stray_control(char: str) -> bool:
    return unicodedata.category(char) == "Cc" and char not in "\n\t"


def normalize_text(text: str, *, keep_indentation: bool = False) -> str:
    """Clean one block of text.

    Paragraph boundaries inside a block are kept as a single blank line; single line
    breaks are kept as they are. `keep_indentation` is for preformatted text (code,
    verse) where leading and repeated spaces carry meaning.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_TRANSLATION)
    text = "".join(char for char in text if not _is_stray_control(char))

    if keep_indentation:
        lines = [line.replace("\t", "    ").rstrip(" ") for line in text.split("\n")]
    else:
        lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _EXCESS_NEWLINES.sub("\n\n", text).strip("\n")


def join_wrapped_lines(lines: list[str], *, dehyphenate: bool = True) -> str:
    """Join the visual lines of one paragraph (e.g. from OCR) into a single line of text.

    With `dehyphenate`, a word broken across lines ("exam-" / "ple") is rejoined, but only
    when a letter+hyphen ends the line and the next line starts in lowercase, so real
    compounds and dashes are left alone.
    """
    result = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not result:
            result = line
        elif (
            dehyphenate
            and result.endswith("-")
            and len(result) >= 2
            and result[-2].isalpha()
            and line[0].islower()
        ):
            result = result[:-1] + line
        else:
            result = f"{result} {line}"
    return result


def normalize_blocks(blocks: list[ContentBlock]) -> tuple[list[ContentBlock], list[ParsingIssue]]:
    """Normalize every block, keep the original text for inspection, drop empty blocks.

    Returns the cleaned blocks (renumbered from 0) and any warnings.
    """
    cleaned: list[ContentBlock] = []
    dropped = 0
    for block in blocks:
        keep_indentation = bool((block.metadata or {}).get("preformatted"))
        normalized = normalize_text(block.text, keep_indentation=keep_indentation)
        if not normalized:
            dropped += 1
            continue
        original = block.raw_text if block.raw_text is not None else block.text
        # PostgreSQL text and JSONB cannot store NUL characters, even in the original text.
        original = original.replace("\x00", "")
        section_title = block.section_title
        if section_title is not None:
            section_title = normalize_text(section_title).replace("\n", " ") or None
        cleaned.append(
            block.model_copy(
                update={
                    "text": normalized,
                    "raw_text": original if original != normalized else None,
                    "section_title": section_title,
                    "sequence_number": len(cleaned),
                }
            )
        )

    warnings: list[ParsingIssue] = []
    if dropped:
        warnings.append(
            ParsingIssue(
                code="empty_blocks_removed",
                message=f"{dropped} block(s) contained no text after normalization and were removed.",
                details={"count": dropped},
            )
        )
    return cleaned, warnings
