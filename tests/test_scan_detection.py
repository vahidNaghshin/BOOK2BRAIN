import pytest

from app.parsing.scan_detection import decide_ocr, measure_text

DEFAULTS = {"min_text_chars": 25, "max_garbled_ratio": 0.3}


def test_measure_text_counts_alphanumerics_and_ignores_whitespace() -> None:
    quality = measure_text("Hello, world 123 \n\t")

    assert quality.alnum_chars == 13
    assert quality.visible_chars == 14  # the 13 alphanumerics plus the comma; whitespace excluded
    assert quality.garbled_chars == 0


def test_measure_text_flags_garbled_characters() -> None:
    quality = measure_text("ab\ufffd\ufffd\ue000\x07")  # replacement, private-use, control

    assert quality.garbled_chars == 4
    assert quality.garbled_ratio == pytest.approx(4 / 6)


def test_empty_text_has_zero_ratio() -> None:
    assert measure_text("").garbled_ratio == 0.0


def test_page_with_enough_text_does_not_need_ocr() -> None:
    quality = measure_text("This page has plenty of perfectly readable text on it.")

    assert not decide_ocr(quality, has_images=True, **DEFAULTS).needs_ocr


def test_page_without_text_but_with_an_image_is_a_scan() -> None:
    decision = decide_ocr(measure_text(""), has_images=True, **DEFAULTS)

    assert decision.needs_ocr and decision.reason == "no_text_layer"


def test_page_with_little_text_and_an_image_needs_ocr() -> None:
    decision = decide_ocr(measure_text("Page 12"), has_images=True, **DEFAULTS)

    assert decision.needs_ocr and decision.reason == "insufficient_text"


def test_blank_page_without_images_is_not_sent_to_ocr() -> None:
    assert not decide_ocr(measure_text(""), has_images=False, **DEFAULTS).needs_ocr
    assert not decide_ocr(measure_text("Chapter 3"), has_images=False, **DEFAULTS).needs_ocr


def test_garbled_text_needs_ocr_even_without_images() -> None:
    garbled = "\ufffd" * 30 + "abc"

    decision = decide_ocr(measure_text(garbled), has_images=False, **DEFAULTS)

    assert decision.needs_ocr and decision.reason == "garbled_text"


def test_thresholds_are_configurable() -> None:
    quality = measure_text("Short but real text here")  # 20 alphanumerics

    assert decide_ocr(quality, has_images=True, min_text_chars=25, max_garbled_ratio=0.3).needs_ocr
    assert not decide_ocr(quality, has_images=True, min_text_chars=10, max_garbled_ratio=0.3).needs_ocr
    mild = measure_text("abc\ufffd")
    assert decide_ocr(mild, has_images=False, min_text_chars=0, max_garbled_ratio=0.1).needs_ocr
    assert not decide_ocr(mild, has_images=False, min_text_chars=0, max_garbled_ratio=0.5).needs_ocr
