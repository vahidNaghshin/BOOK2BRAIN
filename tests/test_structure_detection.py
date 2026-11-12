import pytest

from app.structure.detector import StructureDetector
from app.structure.models import Severity, StructureResult
from tests.structure_helpers import (
    DocBuilder,
    front_and_back_matter_book,
    numbered_textbook,
    parts_book,
    plain_text_book,
    technical_book,
    unnumbered_book,
)


def detect(builder: DocBuilder, **kwargs) -> StructureResult:
    return StructureDetector(**kwargs).detect(builder.build())


def outline(result: StructureResult) -> list[tuple[int, str, str | None, str]]:
    return [(n.level, n.node_type, n.number, n.title) for n in result.nodes]


def codes(result: StructureResult, severity: Severity | None = None) -> set[str]:
    return {i.code for i in result.issues if severity is None or i.severity is severity}


def node(result: StructureResult, title: str):
    return next(n for n in result.nodes if n.title == title)


def test_conventional_textbook_with_numbered_chapters() -> None:
    result = detect(numbered_textbook())

    assert outline(result) == [
        (0, "front_matter", None, "Front Matter"),
        (1, "book_title", None, "Foundations of Learning"),
        (1, "copyright", None, "Copyright"),
        (1, "table_of_contents", None, "Contents"),
        (1, "preface", None, "Preface"),
        (0, "chapter", "1", "Introduction"),
        (1, "section", "1.1", "Background"),
        (1, "section", "1.2", "Motivation"),
        (0, "chapter", "2", "Models"),
        (1, "section", "2.1", "Linear Models"),
        (2, "subsection", "2.1.1", "Least Squares"),
        (0, "chapter", "3", "Learning"),
        (1, "section", "3.1", "Gradient Methods"),
        (0, "back_matter", None, "Back Matter"),
        (1, "appendix", "A", "Mathematical Foundations"),
        (1, "bibliography", None, "References"),
    ]
    assert not codes(result, Severity.ERROR)
    assert not codes(result, Severity.WARNING)


def test_original_heading_text_is_preserved() -> None:
    result = detect(numbered_textbook())

    chapter = node(result, "Introduction")
    assert chapter.original_title == "1. Introduction"
    assert node(result, "Mathematical Foundations").original_title == "Appendix A — Mathematical Foundations"


def test_numbered_book_introduction_is_chapter_one_not_front_matter() -> None:
    result = detect(numbered_textbook())

    assert node(result, "Introduction").node_type == "chapter"
    assert node(result, "Introduction").level == 0


def test_technical_book_with_multiple_subsection_levels() -> None:
    result = detect(technical_book())

    assert [(n.level, n.node_type, n.number) for n in result.nodes if n.number] == [
        (0, "chapter", "1"),
        (1, "section", "1.1"),
        (2, "subsection", "1.1.1"),
        (3, "subsubsection", "1.1.1.1"),
        (2, "subsection", "1.1.2"),
        (1, "section", "1.2"),
        (0, "chapter", "2"),
        (1, "section", "2.1"),
    ]
    assert node(result, "Discrete Signals").parent_id == node(result, "Signals").id
    assert node(result, "Sampling").parent_id == node(result, "Continuous Signals").id


def test_unnumbered_chapters_and_sections() -> None:
    result = detect(unnumbered_book())

    assert outline(result) == [
        (0, "front_matter", None, "Front Matter"),
        (1, "book_title", None, "How Minds Learn"),
        (1, "introduction", None, "Introduction"),
        (0, "chapter", None, "The Foundations of Learning"),
        (1, "section", None, "Early Experience"),
        (0, "chapter", None, "Understanding Representation"),
        (1, "section", None, "Symbols and Signals"),
        (0, "chapter", None, "A New Perspective"),
        (0, "chapter", None, "Conclusion"),
    ]


def test_book_organised_into_parts() -> None:
    result = detect(parts_book())

    assert [(n.level, n.node_type, n.number, n.title) for n in result.nodes if n.level or n.node_type == "part"][1:] == [
        (0, "part", "I", "Foundations"),
        (1, "chapter", "1", "Probability"),
        (2, "section", "1.1", "Events"),
        (1, "chapter", "2", "Utility"),
        (0, "part", "II", "Decision Making"),
        (1, "chapter", "3", "Choices"),
        (2, "section", "3.1", "Trade-offs"),
        (0, "part", "3", "Applications"),
        (1, "chapter", "4", "Case Studies"),
    ]
    assert node(result, "Probability").parent_id == node(result, "Foundations").id


def test_book_with_preface_introduction_and_appendices() -> None:
    result = detect(front_and_back_matter_book())

    top = [(n.level, n.node_type, n.title) for n in result.nodes]
    assert top == [
        (0, "front_matter", "Front Matter"),
        (1, "book_title", "Quiet Machines"),
        (1, "copyright", "Copyright"),
        (1, "foreword", "Foreword"),
        (1, "preface", "Preface"),
        (1, "introduction", "Introduction"),
        (0, "chapter", "Beginnings"),
        (0, "chapter", "Middles"),
        (0, "back_matter", "Back Matter"),
        (1, "appendix", "Proofs"),
        (1, "appendix", "Data"),
        (1, "notes", "Notes"),
        (1, "glossary", "Glossary"),
        (1, "bibliography", "Bibliography"),
    ]
    front, back = node(result, "Front Matter"), node(result, "Back Matter")
    assert node(result, "Preface").parent_id == front.id
    assert node(result, "Proofs").parent_id == back.id


def test_chapter_label_and_title_on_separate_lines_form_one_heading() -> None:
    result = detect(front_and_back_matter_book())

    chapter = node(result, "Beginnings")
    assert chapter.number == "1"
    assert chapter.original_title == "Chapter 1 Beginnings"
    assert len(chapter.heading_blocks) == 2
    assert "Chapter 1" not in {n.title for n in result.nodes}


def test_book_without_table_of_contents_and_without_heading_labels() -> None:
    result = detect(plain_text_book())

    chapters = [n for n in result.nodes if n.node_type == "chapter"]
    assert [(c.number, c.title) for c in chapters] == [("1", "Describing Data"), ("2", "Probability Basics")]
    assert result.summary["toc_entries"] == 0
    sections = [n for n in result.nodes if n.title in ("Measures of Centre", "Conditional Probability")]
    assert len(sections) == 2 and all(s.level == 1 for s in sections)


def test_unlabelled_headings_found_by_typography_are_flagged_as_ambiguous_not_dropped() -> None:
    result = detect(plain_text_book())

    section = node(result, "Measures of Centre")
    assert section.node_type == "other_heading"
    assert section.metadata["status"] == "ambiguous"
    assert "larger_than_body" in section.metadata["signals"] or "much_larger_than_body" in section.metadata["signals"]
    assert "ambiguous_heading" in codes(result, Severity.WARNING)


def test_inconsistent_heading_formatting() -> None:
    b = DocBuilder()
    b.title("A Mixed Bag")
    b.new_page().heading("CHAPTER ONE").heading("THE FIRST STEPS").prose(150, 1)
    b.new_page().heading("Chapter 2: Second Wind").prose(150, 2)
    b.new_page().heading("3 Third Time Lucky").prose(150, 3)
    b.heading("3.1 Details", 2).prose(100, 4)
    b.new_page().heading("Part Four", 1).prose(30, 5)
    result = detect(b)

    main = [(n.node_type, n.number, n.title) for n in result.nodes if n.node_type in ("chapter", "section", "part")]
    assert main == [
        ("chapter", "1", "THE FIRST STEPS"),
        ("chapter", "2", "Second Wind"),
        ("chapter", "3", "Third Time Lucky"),
        ("section", "3.1", "Details"),
        ("part", "4", "Part Four"),
    ]


def test_book_with_equations_keeps_equations_inside_their_section() -> None:
    b = DocBuilder()
    b.title("Math for Machines")
    b.new_page().heading("1. Optimisation").prose(100, 1)
    b.equation("f(x) = x^2 + 2x + 1")
    b.para("where x is the decision variable and f is the objective function to minimise.")
    b.heading("1.1 Gradients", 2).prose(80, 2).equation(r"\nabla f = 2x + 2")
    result = detect(b)

    kinds = {p.kind for p in result.paragraphs}
    assert "equation" in kinds
    optimisation = node(result, "Optimisation")
    gradients = node(result, "Gradients")
    by_node = {nid: [p for p in result.paragraphs if p.node_id == nid] for nid in (optimisation.id, gradients.id)}
    assert [p.kind for p in by_node[optimisation.id]] == ["paragraph", "equation", "paragraph"]
    assert [p.kind for p in by_node[gradients.id]] == ["paragraph", "equation"]
    assert not codes(result, Severity.ERROR)


def test_book_with_tables_and_figures() -> None:
    b = DocBuilder()
    b.title("Data Handbook")
    b.new_page().heading("1. Tables").prose(100, 1)
    b.caption("Table 1.1: Sample means").table("group | mean\nA | 1.0\nB | 2.0")
    b.prose(60, 2).caption("Figure 1.1: Distribution of values")
    result = detect(b)

    paragraphs = [p for p in result.paragraphs if p.node_id == node(result, "Tables").id]
    assert [p.kind for p in paragraphs] == ["paragraph", "caption", "table", "paragraph", "caption"]
    assert paragraphs[1].metadata["table_ref"] == "Table 1.1"
    assert paragraphs[4].metadata["figure_ref"] == "Figure 1.1"
    assert paragraphs[2].metadata["rows"] == 3


def test_chapters_spanning_multiple_pages_get_a_page_range() -> None:
    b = DocBuilder()
    b.title("Long Chapters")
    b.new_page().heading("1. First").prose(100, 1)
    b.new_page().prose(100, 2)
    b.new_page().prose(100, 3).heading("1.1 Deep", 2).prose(100, 4)
    b.new_page().prose(100, 5)
    b.new_page().heading("2. Second").prose(100, 6)
    result = detect(b)

    first, deep, second = node(result, "First"), node(result, "Deep"), node(result, "Second")
    assert (first.page_start, first.page_end) == (2, 5)
    assert (deep.page_start, deep.page_end) == (4, 5)
    assert (second.page_start, second.page_end) == (6, 6)


def test_title_is_not_assumed_to_be_on_the_first_page() -> None:
    result = detect(front_and_back_matter_book())

    guess = result.metadata_guess
    assert guess["title"] == "Quiet Machines"
    assert guess["subtitle"] == "Notes on Computation"
    assert guess["copyright_page"] == 3
    assert node(result, "Quiet Machines").page_start == 2


def test_title_subtitle_and_author_are_detected() -> None:
    result = detect(numbered_textbook())

    assert result.metadata_guess["title"] == "Foundations of Learning"
    assert result.metadata_guess["subtitle"] == "An Introduction to Representation"
    assert result.metadata_guess["author"] == "Jane A. Smith"


def test_author_after_by_is_detected_with_high_confidence() -> None:
    b = DocBuilder()
    b.title("Understanding Waves").plain("by Priya Raman", 16.0)
    b.new_page().heading("1. Waves").prose(100, 1)
    result = detect(b)

    assert result.metadata_guess["author"] == "Priya Raman"
    assert result.metadata_guess["author_confidence"] == 0.9


def test_copyright_information_is_kept_as_front_matter_content() -> None:
    result = detect(numbered_textbook())

    copyright_node = node(result, "Copyright")
    texts = [p.text for p in result.paragraphs if p.node_id == copyright_node.id]
    assert any("All rights reserved" in t for t in texts)


def test_file_metadata_title_is_used_when_the_layout_has_no_title() -> None:
    b = DocBuilder(pages=False, file_format="epub", source_title="The Hidden Pattern")
    b.heading("Chapter 1: Beginnings").prose(100, 1)
    result = detect(b)

    assert result.metadata_guess["title"] == "The Hidden Pattern"
    assert result.metadata_guess["title_source"] == "file_metadata"


def test_table_of_contents_lines_are_not_headings_but_are_kept() -> None:
    result = detect(numbered_textbook())

    chapters = [n for n in result.nodes if n.node_type == "chapter"]
    assert len(chapters) == 3
    contents = node(result, "Contents")
    lines = [p.text for p in result.paragraphs if p.node_id == contents.id]
    assert "1. Introduction ........ 7" in lines
    assert result.summary["toc_entries"] == 4


def test_table_of_contents_mismatch_is_reported() -> None:
    b = DocBuilder()
    b.title("Odd Book").new_page().heading("Contents")
    for line in ("Alpha ........ 3", "Beta ........ 5", "Gamma ........ 9", "Delta ........ 11"):
        b.plain(line)
    b.new_page().heading("1. Something Else").prose(100, 1)
    result = detect(b)

    assert "toc_mismatch" in codes(result, Severity.WARNING)


def test_running_page_headers_are_not_headings() -> None:
    b = DocBuilder()
    b.title("Header Test")
    for page in range(2, 8):
        b.new_page().plain("FOUNDATIONS OF LEARNING", 12.0).prose(150, page)
    b.heading("1. Real Chapter").prose(100, 20)
    result = detect(b)

    assert [n.title for n in result.nodes if n.node_type == "chapter"] == ["Real Chapter"]
    assert "running_headers_ignored" in codes(result)


def test_page_numbers_beside_a_running_header_do_not_hide_it() -> None:
    b = DocBuilder()
    b.title("Header Test")
    for page in range(2, 8):
        b.new_page().plain(f"{page}  Foundations of Learning", 12.0).prose(150, page)
    b.heading("1. Real Chapter").prose(100, 20)
    result = detect(b)

    assert [n.title for n in result.nodes if n.node_type == "chapter"] == ["Real Chapter"]


def test_chapter_headings_on_consecutive_pages_are_not_mistaken_for_running_headers() -> None:
    b = DocBuilder()
    b.title("Short Chapters")
    for number in range(1, 6):
        b.new_page().heading(f"Chapter {number}: Topic {number}").prose(80, number)
    result = detect(b)

    assert [n.number for n in result.nodes if n.node_type == "chapter"] == ["1", "2", "3", "4", "5"]


def test_a_numbered_sentence_is_not_a_heading() -> None:
    b = DocBuilder()
    b.title("Prose").new_page().heading("1. Introduction").prose(80, 1)
    b.para("2 Peter wrote this long sentence about how the numbers 1980 and 2020 relate to one another.")
    b.para("Chapter 3 shows that gradients vanish when the learning rate is too small, as expected.")
    result = detect(b)

    assert [n.title for n in result.nodes if n.node_type == "chapter"] == ["Introduction"]
    texts = [p.text for p in result.paragraphs]
    assert any(t.startswith("2 Peter") for t in texts)
    assert any(t.startswith("Chapter 3 shows") for t in texts)


def test_parser_heading_that_is_really_a_sentence_stays_as_text_with_a_note() -> None:
    sentence = "This is clearly a long sentence that the parser mislabelled as a section heading, and it ends here."
    b = DocBuilder()
    b.title("Sentences").new_page().heading("1. Intro").prose(80, 1)
    b.heading(sentence, 2).prose(80, 2)
    result = detect(b)

    assert sentence not in {n.title for n in result.nodes}
    assert any(p.text == sentence for p in result.paragraphs)
    assert "heading_rejected" in codes(result, Severity.INFO)


def test_document_without_any_heading_keeps_all_text_under_one_node() -> None:
    b = DocBuilder()
    b.new_page().prose(100, 1).prose(100, 2)
    result = detect(b)

    assert len(result.nodes) == 1
    assert result.nodes[0].metadata["synthetic"] is True
    assert len(result.paragraphs) == 2
    assert "no_headings_detected" in codes(result, Severity.WARNING)


def test_formats_without_pages_have_no_page_references() -> None:
    b = DocBuilder(pages=False, file_format="epub")
    b.heading("Chapter 1: Start").prose(100, 1).heading("Chapter 2: Next").prose(100, 2)
    result = detect(b)

    assert all(n.page_start is None and n.page_end is None for n in result.nodes)
    assert all(p.page_start is None and p.page_status == "unknown" for p in result.paragraphs)
    assert "no_page_information" in codes(result, Severity.INFO)
    assert "missing_page_reference" not in codes(result)


def test_every_block_is_kept_exactly_once() -> None:
    for builder in (numbered_textbook, parts_book, unnumbered_book, plain_text_book, front_and_back_matter_book):
        doc = builder().build()
        result = StructureDetector().detect(doc)
        covered = [s for n in result.nodes for s in n.heading_blocks] + [
            s for p in result.paragraphs for s in p.block_sequences
        ]
        assert sorted(covered) == [b.sequence for b in doc.blocks if b.text.strip()], builder.__name__


def test_node_offsets_follow_document_order() -> None:
    result = detect(numbered_textbook())

    chapters = [n for n in result.nodes if n.node_type == "chapter"]
    assert all(a.end_offset < b.start_offset for a, b in zip(chapters, chapters[1:]))
    for child in result.nodes:
        parent = next((n for n in result.nodes if n.id == child.parent_id), None)
        if parent:
            assert parent.start_offset <= child.start_offset and child.end_offset <= parent.end_offset


def test_ambiguous_numbered_line_without_neighbours_is_flagged_not_trusted() -> None:
    b = DocBuilder()
    b.title("Lists").new_page().heading("1. Intro").prose(80, 1)
    b.plain("7 Final Thoughts")
    b.prose(80, 2)
    result = detect(b)

    assert "Final Thoughts" not in {n.title for n in result.nodes if n.node_type == "chapter"}


def test_confident_and_ambiguous_scores_follow_the_thresholds() -> None:
    strict = detect(plain_text_book(), accept_threshold=0.95, reject_threshold=0.95)
    lenient = detect(plain_text_book(), accept_threshold=0.5, reject_threshold=0.3)

    assert "Measures of Centre" not in {n.title for n in strict.nodes}
    assert node(lenient, "Measures of Centre").metadata["status"] == "accepted"
    assert any(p.text == "Measures of Centre" for p in strict.paragraphs)  # kept as body text, not lost


@pytest.mark.parametrize("builder", [numbered_textbook, parts_book, technical_book, front_and_back_matter_book])
def test_detection_is_deterministic(builder) -> None:
    first, second = detect(builder()), detect(builder())

    assert outline(first) == outline(second)
    assert [p.text for p in first.paragraphs] == [p.text for p in second.paragraphs]
