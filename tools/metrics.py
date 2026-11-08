"""LaTeX comparison metrics shared by the evaluation tools (no heavy dependencies)."""

import difflib
import re

_TOKEN = re.compile(r"\\[A-Za-z]+|[A-Za-z0-9]|[^\sA-Za-z0-9]")


def canonical(latex: str) -> str:
    """LaTeX with spacing and \\left/\\right removed, so only the written content is compared."""
    text = re.sub(r"\\(?:left|right)\b", "", latex)
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    return re.sub(r"\s+", "", text)


def similarity(predicted: str, truth: str) -> float:
    """Token-level similarity in 0-1 (difflib ratio over LaTeX tokens)."""
    a, b = _TOKEN.findall(canonical(predicted)), _TOKEN.findall(canonical(truth))
    return difflib.SequenceMatcher(None, a, b).ratio()
