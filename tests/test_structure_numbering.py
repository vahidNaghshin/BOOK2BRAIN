import pytest

from app.structure.models import NodeType
from app.structure.numbering import (
    int_to_roman,
    looks_like_caption,
    match_keyword,
    parse_heading,
    roman_to_int,
    word_to_int,
)


@pytest.mark.parametrize(
    ("text", "node_type", "number", "style", "title"),
    [
        ("CHAPTER 1", NodeType.CHAPTER, "1", "arabic", ""),
        ("Chapter One", NodeType.CHAPTER, "1", "word", ""),
        ("Chapter 1: Introduction", NodeType.CHAPTER, "1", "arabic", "Introduction"),
        ("Chapter twenty-one - Waves", NodeType.CHAPTER, "21", "word", "Waves"),
        ("PART II — Decision Making", NodeType.PART, "II", "roman", "Decision Making"),
        ("Part Three: Applications", NodeType.PART, "3", "word", "Applications"),
        ("Appendix A — Mathematical Foundations", NodeType.APPENDIX, "A", "letter", "Mathematical Foundations"),
        ("Section 3.2 Error Bounds", NodeType.SECTION, "3.2", "decimal", "Error Bounds"),
    ],
)
def test_labelled_headings(text, node_type, number, style, title) -> None:
    pattern = parse_heading(text)
    assert pattern is not None
    assert (pattern.node_type, pattern.number, pattern.number_style, pattern.title) == (
        node_type, number, style, title,
    )


@pytest.mark.parametrize(
    ("text", "form", "number", "depth", "rank"),
    [
        ("1. Introduction", "numeric", "1", 1, 20),
        ("2 Methods", "numeric", "2", 1, 20),
        ("1.1 Background", "decimal", "1.1", 2, 30),
        ("1.1.1 Historical Context", "decimal", "1.1.1", 3, 40),
        ("1.1.1.1 Details", "decimal", "1.1.1.1", 4, 50),
        ("IV. Results", "roman_dot", "IV", 1, 20),
    ],
)
def test_numbered_headings(text, form, number, depth, rank) -> None:
    pattern = parse_heading(text)
    assert pattern is not None
    assert (pattern.form, pattern.number, pattern.depth, pattern.explicit_rank) == (form, number, depth, rank)


@pytest.mark.parametrize(
    "text",
    [
        "Part of the problem is simple",  # "Part" followed by a word, not a number
        "Chapter Mix",  # not a roman numeral
        "Part C",  # a letter is only a number for appendices
        "3.14 is the value of pi",  # decimal followed by lowercase text
        "1980 was a good year",  # four digits are a year, not a chapter number
        "Introduction",
        "",
    ],
)
def test_text_without_numbering_is_not_a_pattern(text) -> None:
    assert parse_heading(text) is None


def test_chapter_sentence_still_parses_but_scores_as_prose() -> None:
    # The pattern alone cannot tell; the scoring stage uses the long, sentence-like title.
    pattern = parse_heading("Chapter 3 shows that gradients vanish.")
    assert pattern is not None and pattern.title.endswith(".")


@pytest.mark.parametrize("value", [1, 4, 9, 14, 40, 90, 400, 1994, 3999])
def test_roman_numerals_round_trip(value) -> None:
    assert roman_to_int(int_to_roman(value)) == value
    assert roman_to_int(int_to_roman(value).lower()) == value


@pytest.mark.parametrize("text", ["IIII", "VX", "Mix", "", "ABC"])
def test_invalid_roman_numerals(text) -> None:
    assert roman_to_int(text) is None


@pytest.mark.parametrize(("text", "value"), [("one", 1), ("Nineteen", 19), ("forty-two", 42), ("ninety nine", 99)])
def test_number_words(text, value) -> None:
    assert word_to_int(text) == value


@pytest.mark.parametrize(
    ("text", "node_type"),
    [
        ("Preface", NodeType.PREFACE),
        ("PREFACE TO THE SECOND EDITION", NodeType.PREFACE),
        ("Foreword", NodeType.FOREWORD),
        ("Acknowledgements", NodeType.ACKNOWLEDGEMENT),
        ("Acknowledgments", NodeType.ACKNOWLEDGEMENT),
        ("Introduction", NodeType.INTRODUCTION),
        ("Table of Contents", NodeType.TABLE_OF_CONTENTS),
        ("References", NodeType.BIBLIOGRAPHY),
        ("Glossary", NodeType.GLOSSARY),
        ("Index", NodeType.INDEX),
        ("Notes", NodeType.NOTES),
        ("Conclusion", None),
    ],
)
def test_keyword_headings(text, node_type) -> None:
    rule = match_keyword(text)
    assert rule is not None
    assert rule.node_type == node_type


@pytest.mark.parametrize("text", ["Notes on method", "The Introduction of Steam", "Referencing styles", ""])
def test_keywords_must_match_the_whole_heading(text) -> None:
    assert match_keyword(text) is None


def test_captions_are_recognised() -> None:
    assert looks_like_caption("Figure 2.1: Error versus epochs")
    assert looks_like_caption("Table 3 Results")
    assert not looks_like_caption("Figures of speech")
