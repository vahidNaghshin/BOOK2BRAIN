"""Pluggable token counting for passage sizes.

The default is an approximation (characters per token) that needs no extra dependency.
`tiktoken` can be selected when it is installed, for counts that match a specific model.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

_WORD_OR_PUNCT = re.compile(r"\w+|[^\w\s]")


class TokenizerUnavailableError(Exception):
    """The configured tokenizer cannot be used. The message is safe to show to users."""


class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class ApproximateTokenizer:
    def __init__(self, chars_per_token: float = 4.0) -> None:
        self.chars_per_token = chars_per_token
        self.name = f"approx:{chars_per_token:g}"

    def count(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token) if text else 0


class WordTokenizer:
    """Words and punctuation marks, one token each."""

    name = "words"

    def count(self, text: str) -> int:
        return len(_WORD_OR_PUNCT.findall(text))


class WhitespaceTokenizer:
    name = "whitespace"

    def count(self, text: str) -> int:
        return len(text.split())


class TiktokenTokenizer:
    def __init__(self, encoding: str) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise TokenizerUnavailableError(
                "PASSAGE_TOKENIZER=tiktoken needs the optional 'tiktoken' package "
                "(pip install tiktoken), or choose another tokenizer."
            ) from exc
        try:
            self._encoding = tiktoken.get_encoding(encoding)
        except Exception as exc:  # tiktoken raises ValueError or network errors for bad/uncached names
            raise TokenizerUnavailableError(
                f"The tiktoken encoding '{encoding}' could not be loaded ({exc})."
            ) from exc
        self.name = f"tiktoken:{encoding}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=())) if text else 0


def build_tokenizer(
    name: str, *, chars_per_token: float = 4.0, tiktoken_encoding: str = "cl100k_base"
) -> Tokenizer:
    if name == "approx":
        return ApproximateTokenizer(chars_per_token)
    if name == "words":
        return WordTokenizer()
    if name == "whitespace":
        return WhitespaceTokenizer()
    if name == "tiktoken":
        return TiktokenTokenizer(tiktoken_encoding)
    raise TokenizerUnavailableError(f"Unknown tokenizer '{name}'.")
