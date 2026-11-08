import pytest

from app.parsing.models import BlockType, ContentBlock
from app.parsing.text_normalizer import join_wrapped_lines, normalize_blocks, normalize_text


def test_unicode_is_normalized_to_nfc() -> None:
    decomposed = "Cafe\u0301"  # 'e' + combining acute
    assert normalize_text(decomposed) == "Caf\u00e9"
    assert len(normalize_text(decomposed)) == 4


def test_newlines_are_made_consistent() -> None:
    assert normalize_text("a\r\nb\rc\u2028d") == "a\nb\nc\nd"
    assert normalize_text("one\u2029two") == "one\n\ntwo"
    assert normalize_text("a\x0bb\x0cc") == "a\nb\nc"


def test_whitespace_is_cleaned_but_words_are_untouched() -> None:
    assert normalize_text("  hello \t  world  ") == "hello world"
    assert normalize_text("non\u00a0breaking\u2009thin\u3000wide") == "non breaking thin wide"


def test_excessive_blank_lines_collapse_to_one_paragraph_break() -> None:
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"
    assert normalize_text("\n\n a \n\n") == "a"
    assert normalize_text("a \n   \n b") == "a\n\nb"


def test_paragraph_boundaries_are_preserved_not_merged() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird."
    assert normalize_text(text) == text


def test_single_line_breaks_are_kept() -> None:
    assert normalize_text("roses are red\nviolets are blue") == "roses are red\nviolets are blue"


def test_extraction_artifacts_are_removed() -> None:
    assert normalize_text("ef\u00adfi\u00adcient") == "efficient"  # soft hyphens
    assert normalize_text("zero\u200bwidth\ufeff") == "zerowidth"
    assert normalize_text("\ufb01nal \ufb02ow o\ufb03ce") == "final flow office"  # ligatures
    assert normalize_text("bell\x07 and null\x00 gone") == "bell and null gone"


@pytest.mark.parametrize(
    "text",
    [
        "Caf\u00e9 \u201cquoted\u201d \u2014 it\u2019s fine\u2026",  # smart punctuation
        "Price: \u20ac10, \u00a35, \u00a53 \u2022 \u00a9 2024 \u2122",
        "\u0645\u0631\u062d\u0628\u0627 \u4f60\u597d \u041f\u0440\u0438\u0432\u0435\u0442",  # other scripts
        "H\u2082O and x\u00b2 \u2260 y\u00b3",  # sub/superscripts must not become plain digits
        "caf\u00e9 \u00fc na\u00efve",
    ],
)
def test_special_characters_are_safe(text: str) -> None:
    result = normalize_text(text)
    assert result == normalize_text(result)
    for char in text:
        if not char.isspace():
            assert char in result, f"{char!r} was lost"


def test_zero_width_joiner_is_kept() -> None:
    assert normalize_text("a\u200db") == "a\u200db"


@pytest.mark.parametrize(
    "text",
    [
        "1. First item\n2. Second item (see [3], p. 14; ibid.)",
        "Section 3.2.1 \u2014 Results\u00b9",
        "Smith et al. (2020), pp. 12\u201318; doi:10.1000/xyz-123",
        "- dash item\n\u2022 bullet item\n* star item",
        'He said: "Wait!" (really?); e.g. a, b & c.',
    ],
)
def test_numbering_punctuation_and_citations_are_preserved(text: str) -> None:
    assert normalize_text(text) == text


def test_table_rows_are_kept_on_separate_lines() -> None:
    table = "Name | Value\nAlpha | 42\nBeta | 7"
    assert normalize_text(table) == table


def test_normalization_is_idempotent_and_deterministic() -> None:
    messy = "  Caf\u00e9\u00a0 \ufb01nal\u00ad\r\n\r\n\r\n\u2022  item\t\t1  \n"
    once = normalize_text(messy)
    assert normalize_text(once) == once
    assert normalize_text(messy) == once


def test_keep_indentation_for_preformatted_text() -> None:
    code = "def f():\n    return 1\n\tindented tab"
    result = normalize_text(code, keep_indentation=True)
    assert result == "def f():\n    return 1\n    indented tab"


def test_join_wrapped_lines_unwraps_a_paragraph() -> None:
    assert join_wrapped_lines(["The quick brown", "fox jumps", "over."]) == "The quick brown fox jumps over."


def test_join_wrapped_lines_repairs_line_break_hyphenation_conservatively() -> None:
    assert join_wrapped_lines(["an exam-", "ple here"]) == "an example here"
    # Capitalised continuation, spaced dash and double hyphen are left alone.
    assert join_wrapped_lines(["self-", "Sustaining"]) == "self- Sustaining"
    assert join_wrapped_lines(["well -", "known"]) == "well - known"
    assert join_wrapped_lines(["wait--", "what"]) == "wait-- what"
    assert join_wrapped_lines(["an exam-", "ple"], dehyphenate=False) == "an exam- ple"


def test_normalize_blocks_keeps_original_text_only_when_changed() -> None:
    blocks = [
        ContentBlock(text="Clean text.", sequence_number=0),
        ContentBlock(text="  Messy\u00a0 text\ufb01  ", sequence_number=1),
    ]

    cleaned, warnings = normalize_blocks(blocks)

    assert [b.text for b in cleaned] == ["Clean text.", "Messy textfi"]
    assert cleaned[0].raw_text is None
    assert cleaned[1].raw_text == "  Messy\u00a0 text\ufb01  "
    assert warnings == []


def test_normalize_blocks_drops_empty_blocks_and_renumbers() -> None:
    blocks = [
        ContentBlock(text="keep one", sequence_number=0),
        ContentBlock(text=" \u200b \n ", sequence_number=1),
        ContentBlock(text="keep two", sequence_number=2),
    ]

    cleaned, warnings = normalize_blocks(blocks)

    assert [(b.sequence_number, b.text) for b in cleaned] == [(0, "keep one"), (1, "keep two")]
    assert warnings[0].code == "empty_blocks_removed"
    assert warnings[0].details == {"count": 1}


def test_normalize_blocks_preserves_source_information_and_types() -> None:
    block = ContentBlock(
        text="Heading  text",
        block_type=BlockType.HEADING,
        page_number=7,
        section_title="  A   Section ",
        source_location={"page": 7},
        metadata={"heading_level": 2},
    )

    (cleaned,), _ = normalize_blocks([block])

    assert cleaned.block_type is BlockType.HEADING
    assert cleaned.page_number == 7
    assert cleaned.section_title == "A Section"
    assert cleaned.source_location == {"page": 7}
    assert cleaned.metadata == {"heading_level": 2}


def test_normalize_blocks_strips_nul_characters_from_original_text() -> None:
    (cleaned,), _ = normalize_blocks([ContentBlock(text="a\x00b")])

    assert cleaned.text == "ab"
    assert cleaned.raw_text is None  # nothing but the unstorable NUL differed
    (with_difference,), _ = normalize_blocks([ContentBlock(text=" a\x00b ")])
    assert with_difference.raw_text == " ab "  # original is kept for inspection, NUL-free


def test_normalize_blocks_keeps_preformatted_indentation() -> None:
    block = ContentBlock(
        text="if x:\n    y()", block_type=BlockType.OTHER, metadata={"preformatted": True}
    )

    (cleaned,), _ = normalize_blocks([block])

    assert cleaned.text == "if x:\n    y()"
    assert cleaned.raw_text is None
