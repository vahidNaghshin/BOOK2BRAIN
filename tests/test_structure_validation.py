import copy
import uuid

import pytest

from app.structure.detector import StructureDetector
from app.structure.models import DocumentText, ParagraphDraft, Severity, StructureResult
from app.structure.segmentation import SegmentationConfig, generate_passages
from app.structure.validation import count_by_severity, validate_passages, validate_structure
from tests.structure_helpers import DocBuilder, numbered_textbook, prose

CONFIG = SegmentationConfig(target_tokens=200, min_tokens=100, max_tokens=300)
TOKENIZER = CONFIG.build_tokenizer()


def fresh() -> tuple[DocumentText, StructureResult]:
    document = numbered_textbook().build()
    return document, StructureDetector().detect(document)


def structure_codes(document, nodes, paragraphs) -> set[str]:
    return {i.code for i in validate_structure(document, nodes, paragraphs)}


def passage_codes(document, result, passages, config=CONFIG) -> set[str]:
    return {i.code for i in validate_passages(document, result.nodes, result.paragraphs, passages, config, config.build_tokenizer())}


def node_named(result, title):
    return next(n for n in result.nodes if n.title == title)


def test_a_clean_book_validates_without_errors_or_warnings() -> None:
    document, result = fresh()
    passages = generate_passages(result.nodes, result.paragraphs, CONFIG)

    assert [i for i in validate_structure(document, result.nodes, result.paragraphs) if i.severity is not Severity.INFO] == []
    assert [i for i in validate_passages(document, result.nodes, result.paragraphs, passages, CONFIG, TOKENIZER) if i.severity is not Severity.INFO] == []


# -- hierarchy ------------------------------------------------------------------------------


def test_circular_hierarchy_is_an_error() -> None:
    document, result = fresh()
    nodes = copy.deepcopy(result.nodes)
    chapter = next(n for n in nodes if n.title == "Introduction")
    section = next(n for n in nodes if n.title == "Background")
    chapter.parent_id = section.id  # the chapter is now its own grandchild

    assert "circular_hierarchy" in structure_codes(document, nodes, result.paragraphs)


def test_missing_parent_and_inconsistent_levels_are_errors() -> None:
    document, result = fresh()
    nodes = copy.deepcopy(result.nodes)
    node_named_ = next(n for n in nodes if n.title == "Background")
    node_named_.level = 5
    orphan = next(n for n in nodes if n.title == "Motivation")
    orphan.parent_id = uuid.uuid4()

    codes = structure_codes(document, nodes, result.paragraphs)

    assert {"level_inconsistent", "parent_missing"} <= codes


def test_sequence_must_follow_document_order() -> None:
    document, result = fresh()
    nodes = copy.deepcopy(result.nodes)
    nodes[3].sequence, nodes[4].sequence = nodes[4].sequence, nodes[3].sequence

    assert "sequence_not_increasing" in structure_codes(document, nodes, result.paragraphs)


def test_duplicate_nodes_are_flagged() -> None:
    b = DocBuilder()
    b.title("Dupes").new_page().heading("1. Same").prose(60, 1).heading("1. Same").prose(60, 2)
    document = b.build()
    result = StructureDetector().detect(document)

    codes = {i.code for i in result.issues}

    assert "duplicate_node" in codes
    assert "duplicate_number" in codes


def test_numbering_out_of_order_and_gaps_are_reported() -> None:
    b = DocBuilder()
    b.title("Numbers").new_page().heading("1. A").prose(60, 1).heading("2. B").prose(60, 2)
    b.heading("5. E").prose(60, 3).heading("4. D").prose(60, 4)
    result = StructureDetector().detect(b.build())

    issues = {i.code: i for i in result.issues}
    assert issues["numbering_gap"].severity is Severity.INFO
    assert issues["numbering_out_of_order"].severity is Severity.WARNING


def test_child_must_lie_within_its_parent() -> None:
    document, result = fresh()
    nodes = copy.deepcopy(result.nodes)
    child = next(n for n in nodes if n.title == "Background")
    child.end_block = (child.end_block or 0) + 500

    assert "child_outside_parent" in structure_codes(document, nodes, result.paragraphs)


def test_skipped_levels_are_noted() -> None:
    b = DocBuilder()
    b.title("Skips").new_page().heading("1. Chapter").prose(60, 1).heading("1.1.1 Deep", 3).prose(60, 2)
    result = StructureDetector().detect(b.build())

    assert "level_skipped" in {i.code for i in result.issues}


def test_empty_nodes_are_flagged() -> None:
    b = DocBuilder()
    b.title("Empty").new_page().heading("1. Chapter").heading("2. Chapter Two").prose(60, 1)
    result = StructureDetector().detect(b.build())

    empty = [i for i in result.issues if i.code == "empty_node"]
    assert len(empty) == 1


# -- pages ----------------------------------------------------------------------------------


def test_pages_must_be_within_the_document_and_ordered() -> None:
    document, result = fresh()
    nodes = copy.deepcopy(result.nodes)
    chapter = next(n for n in nodes if n.title == "Models")
    chapter.page_end = 999
    other = next(n for n in nodes if n.title == "Learning")
    other.page_start, other.page_end = 7, 6

    codes = structure_codes(document, nodes, result.paragraphs)

    assert {"page_out_of_range", "page_range_inverted"} <= codes


def test_child_pages_must_be_consistent_with_the_parent() -> None:
    document, result = fresh()
    nodes = copy.deepcopy(result.nodes)
    section = next(n for n in nodes if n.title == "Linear Models")
    section.page_start, section.page_end = 1, 2

    assert "child_pages_outside_parent" in structure_codes(document, nodes, result.paragraphs)


def test_missing_page_references_are_flagged_for_paginated_documents() -> None:
    document, result = fresh()
    paragraphs = copy.deepcopy(result.paragraphs)
    for paragraph in paragraphs[:3]:
        paragraph.page_start = paragraph.page_end = None

    issues = [i for i in validate_structure(document, result.nodes, paragraphs) if i.code == "missing_page_reference"]

    assert issues and issues[0].details["count"] == 3


# -- content --------------------------------------------------------------------------------


def test_missing_content_is_an_error() -> None:
    document, result = fresh()
    paragraphs = result.paragraphs[:-1]  # silently dropping a paragraph

    issues = [i for i in validate_structure(document, result.nodes, paragraphs) if i.code == "content_missing"]

    assert issues and issues[0].severity is Severity.ERROR


def test_a_heading_repeated_as_body_text_is_an_error() -> None:
    document, result = fresh()
    paragraphs = copy.deepcopy(result.paragraphs)
    chapter = next(n for n in result.nodes if n.title == "Models")
    paragraphs[0].block_sequences = list(paragraphs[0].block_sequences) + [chapter.heading_blocks[0]]

    assert "content_duplicated" in structure_codes(document, result.nodes, paragraphs)


def test_paragraph_text_must_match_its_source_blocks() -> None:
    document, result = fresh()
    paragraphs = copy.deepcopy(result.paragraphs)
    paragraphs[2].text += " invented words"

    assert "paragraph_text_mismatch" in structure_codes(document, result.nodes, paragraphs)


def test_heading_text_as_first_paragraph_is_flagged() -> None:
    document, result = fresh()
    paragraphs = copy.deepcopy(result.paragraphs)
    node = node_named(result, "Models")
    first = next(p for p in paragraphs if p.node_id == node.id)
    first.text = "Models"

    assert "heading_duplicated_in_body" in structure_codes(document, result.nodes, paragraphs)


def test_unclassified_content_is_retained_with_a_warning() -> None:
    b = DocBuilder()
    b.title("Doubt").new_page().heading("1. Intro").prose(60, 1)
    b.plain("Measures of Centre", 18.0).prose(60, 2).plain("Another Odd Line", 18.0).prose(60, 3)
    document = b.build()
    result = StructureDetector().detect(document)

    texts = " ".join(p.text for p in result.paragraphs) + " ".join(n.title for n in result.nodes)
    assert "Measures of Centre" in texts and "Another Odd Line" in texts
    assert not [i for i in result.issues if i.code == "content_missing"]


def test_issue_counts_by_severity() -> None:
    _, result = fresh()
    counts = count_by_severity(result.issues)

    assert set(counts) == {"error", "warning", "info"}


# -- passages -------------------------------------------------------------------------------


def passages_fixture():
    document = DocBuilder().title("P").new_page().heading("1. Long").build()
    builder = DocBuilder()
    builder.title("P").new_page().heading("1. Long")
    for i in range(8):
        builder.prose(150, i)
    document = builder.build()
    result = StructureDetector().detect(document)
    passages = generate_passages(result.nodes, result.paragraphs, CONFIG)
    assert len(passages) >= 3
    return document, result, passages


def test_clean_passages_validate() -> None:
    document, result, passages = passages_fixture()

    assert not [c for c in passage_codes(document, result, passages) if c not in ("oversized_passage", "undersized_passage")]


def test_token_count_is_verified() -> None:
    document, result, passages = passages_fixture()
    passages = copy.deepcopy(passages)
    passages[0].token_count += 5

    assert "token_count_mismatch" in passage_codes(document, result, passages)


def test_empty_and_orphan_passages_are_errors() -> None:
    document, result, passages = passages_fixture()
    passages = copy.deepcopy(passages)
    passages[0].text = "   "
    passages[1].node_id = uuid.uuid4()

    codes = passage_codes(document, result, passages)

    assert {"empty_passage", "passage_node_missing"} <= codes


def test_omitted_text_is_detected() -> None:
    document, result, passages = passages_fixture()

    assert "passage_text_mismatch" in passage_codes(document, result, passages[:-1])


def test_duplicated_text_is_detected() -> None:
    document, result, passages = passages_fixture()
    passages = copy.deepcopy(passages)
    passages[1].text = passages[1].text + "\n\n" + passages[0].text

    assert "passage_text_mismatch" in passage_codes(document, result, passages)


def test_passage_order_and_sequence_are_checked() -> None:
    document, result, passages = passages_fixture()
    passages = copy.deepcopy(passages)
    passages[1].sequence = 7
    passages[2].document_sequence = passages[0].document_sequence

    codes = passage_codes(document, result, passages)

    assert {"passage_sequence_gap", "passage_order"} <= codes


def test_passage_pages_are_checked() -> None:
    document, result, passages = passages_fixture()
    passages = copy.deepcopy(passages)
    passages[0].page_end = 500
    passages[1].page_start, passages[1].page_end = 3, 2

    codes = passage_codes(document, result, passages)

    assert {"page_out_of_range", "page_range_inverted"} <= codes


def test_passage_pages_must_match_the_paragraphs_they_contain() -> None:
    document, result, passages = passages_fixture()
    passages = copy.deepcopy(passages)
    passages[0].page_start = 1
    passages[0].page_end = 1

    assert "passage_pages_inconsistent" in passage_codes(document, result, passages)


def test_passages_must_reference_real_paragraphs_of_the_same_section() -> None:
    document, result, passages = passages_fixture()
    passages = copy.deepcopy(passages)
    passages[0].parts[0].paragraph_id = uuid.uuid4()

    assert "passage_paragraph_invalid" in passage_codes(document, result, passages)


def test_overlap_must_come_from_the_previous_passage_and_follow_the_policy() -> None:
    document, result, passages = passages_fixture()
    passages = copy.deepcopy(passages)
    target = passages[2]
    prefix = "Words that never appeared before.\n\n"
    target.text = prefix + target.text
    target.metadata["overlap_chars"] = len(prefix) + int(target.metadata["overlap_chars"])

    codes = passage_codes(document, result, passages)

    assert "overlap_not_from_previous" in codes
    assert "passage_text_mismatch" not in codes  # repeated text is excluded from the new-content check


def test_oversized_overlap_is_rejected() -> None:
    document, result, passages = passages_fixture()
    passages = copy.deepcopy(passages)
    previous, current = passages[0], passages[1]
    big = previous.text[-900:]
    current.text = big + "\n\n" + current.text[int(current.metadata["overlap_chars"]) :]
    current.metadata["overlap_chars"] = len(big) + 2
    current.token_count = TOKENIZER.count(current.text)

    assert "overlap_exceeds_policy" in passage_codes(document, result, passages)


def test_the_first_passage_cannot_start_with_overlap() -> None:
    document, result, passages = passages_fixture()
    passages = copy.deepcopy(passages)
    passages[0].metadata["overlap_chars"] = 10

    assert "overlap_without_predecessor" in passage_codes(document, result, passages)


def test_sections_with_paragraphs_but_no_passages_are_flagged() -> None:
    document, result, passages = passages_fixture()

    assert "node_without_passages" in passage_codes(document, result, [])


def test_oversized_passages_are_reported_as_information() -> None:
    node_doc = DocBuilder().title("T").new_page().heading("1. Table")
    node_doc.table("\n".join(f"row {i} | " + prose(30, i) for i in range(12)))
    document = node_doc.build()
    result = StructureDetector().detect(document)
    passages = generate_passages(result.nodes, result.paragraphs, CONFIG)

    issues = validate_passages(document, result.nodes, result.paragraphs, passages, CONFIG, TOKENIZER)

    assert {i.code: i.severity for i in issues}["oversized_passage"] is Severity.INFO
    assert not [i for i in issues if i.severity is Severity.ERROR]


@pytest.mark.parametrize("field", ["text", "kind"])
def test_paragraph_records_are_plain_data(field) -> None:
    document, result = fresh()
    paragraph: ParagraphDraft = result.paragraphs[0]

    assert getattr(paragraph, field)
