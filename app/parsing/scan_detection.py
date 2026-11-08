"""Page-level detection of pages that need OCR. Pure functions, easy to test and tune."""

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class PageTextQuality:
    alnum_chars: int
    visible_chars: int
    garbled_chars: int

    @property
    def garbled_ratio(self) -> float:
        return self.garbled_chars / self.visible_chars if self.visible_chars else 0.0


@dataclass(frozen=True)
class OcrDecision:
    needs_ocr: bool
    reason: str = ""


def measure_text(text: str) -> PageTextQuality:
    """Text quality indicators: letters/digits present, and how much looks like garbage.

    Garbage means the Unicode replacement character, private-use code points and stray
    control characters, which is what broken font encodings typically produce.
    """
    alnum = visible = garbled = 0
    for char in text:
        if char.isspace():
            continue
        visible += 1
        category = unicodedata.category(char)
        if char.isalnum():
            alnum += 1
        elif char == "\ufffd" or category in ("Co", "Cc", "Cn"):
            garbled += 1
    return PageTextQuality(alnum_chars=alnum, visible_chars=visible, garbled_chars=garbled)


def decide_ocr(
    quality: PageTextQuality,
    *,
    has_images: bool,
    min_text_chars: int,
    max_garbled_ratio: float,
) -> OcrDecision:
    """Decide whether one PDF page should be OCR'd.

    - Garbled extracted text: OCR, because the text layer is unusable.
    - Too little text AND the page has images: a scan (or mostly a scan).
    - Too little text and no images: a blank page, so OCR would only waste time.
    """
    if quality.visible_chars and quality.garbled_ratio > max_garbled_ratio:
        return OcrDecision(True, "garbled_text")
    if quality.alnum_chars < min_text_chars and has_images:
        return OcrDecision(True, "no_text_layer" if quality.alnum_chars == 0 else "insufficient_text")
    return OcrDecision(False)
