import re
import uuid

import pytest

from app.structure.models import NodeDraft, ParagraphDraft, Severity
from app.structure.segmentation import (
    PARAGRAPH_SEPARATOR,
    SegmentationConfig,
    generate_passages,
    segment_node,
    split_sentences,
)
from app.structure.tokenizer import TokenizerUnavailableError, build_tokenizer
from app.structure.validation import validate_passages
from tests.structure_helpers import prose

CONFIG = SegmentationConfig()  # 750 target, 500 minimum, 1000 maximum, 12% overlap, ~4 chars/token
TOKENIZER = CONFIG.build_tokenizer()


def make_node(title: str = "Section", sequence: int = 0) -> NodeDraft:
    return NodeDraft(id=uuid.uuid4(), node_type="section", title=title, level=0, sequence=sequence)


def make_paragraph(
    node: NodeDraft,
    sequence: int,
    text: str,
    *,
    kind: str = "paragraph",
    page: int | None = 1,
    page_end: int | None = None,
    document_sequence: int | None = None,
    **metadata,
) -> ParagraphDraft:
    status = "exact" if page is not None else "unknown"
    return ParagraphDraft(
        id=uuid.uuid4(),
        node_id=node.id,
        sequence=sequence,
        document_sequence=sequence if document_sequence is None else document_sequence,
        kind=kind,
        text=text,
        block_sequences=[sequence],
        page_start=page,
        page_end=page_end if page_end is not None else page,
        page_status=status,
        start_offset=sequence * 10_000,
        end_offset=sequence * 10_000 + len(text),
        offsets_exact=True,
        metadata=dict(metadata),
    )


def paragraphs_of(node: NodeDraft, sizes: list[int], *, start_page: int = 1) -> list[ParagraphDraft]:
    return [make_paragraph(node, i, prose(size, seed=i * 31), page=start_page + i // 2) for i, size in enumerate(sizes)]


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def passages_for(node, paragraphs, config=CONFIG):
    return segment_node(node, paragraphs, config, config.build_tokenizer())


def validate(node, paragraphs, passages, config=CONFIG, *, page_count=20):
    from app.structure.models import DocumentText

    document = DocumentText(blocks=[], page_count=page_count, known_pages=True)
    return validate_passages(document, [node], paragraphs, passages, config, config.build_tokenizer())


def errors(issues):
    return [i for i in issues if i.severity is Severity.ERROR]


# -- configuration -------------------------------------------------------------------------


def test_defaults_match_the_specification() -> None:
    assert (CONFIG.target_tokens, CONFIG.min_tokens, CONFIG.max_tokens) == (750, 500, 1000)
    assert 0.10 <= CONFIG.overlap_ratio <= 0.15
    assert CONFIG.overlap_max_tokens == 112  # 15% of the target


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_tokens": 400, "min_tokens": 500},
        {"target_tokens": 1200, "max_tokens": 1000},
        {"overlap_ratio": 0.9},
        {"min_tokens": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        SegmentationConfig(**kwargs)


def test_settings_drive_the_configuration(parsing_settings) -> None:
    settings = parsing_settings.model_copy(update={"passage_target_tokens": 300, "passage_min_tokens": 200, "passage_max_tokens": 400})

    config = SegmentationConfig.from_settings(settings)

    assert (config.target_tokens, config.min_tokens, config.max_tokens) == (300, 200, 400)


def test_application_settings_reject_inconsistent_sizes(tmp_path) -> None:
    from app.config import Settings

    with pytest.raises(ValueError, match="PASSAGE_MIN_TOKENS"):
        Settings(database_url="postgresql://x/y", passage_min_tokens=900, passage_target_tokens=800, _env_file=None)


# -- tokenizers ----------------------------------------------------------------------------


def test_tokenizers_count_differently_and_are_selectable() -> None:
    text = "Hello, world! This is a test."

    assert build_tokenizer("whitespace").count(text) == 6
    assert build_tokenizer("words").count(text) == 9
    assert build_tokenizer("approx", chars_per_token=4).count(text) == 8
    assert build_tokenizer("approx").count("") == 0


def test_unknown_tokenizer_is_an_error() -> None:
    with pytest.raises(TokenizerUnavailableError):
        build_tokenizer("nonsense")


def test_segmentation_uses_the_configured_tokenizer() -> None:
    node = make_node()
    paragraphs = paragraphs_of(node, [300, 300])
    approx = passages_for(node, paragraphs, SegmentationConfig(tokenizer="approx"))
    words = passages_for(node, paragraphs, SegmentationConfig(tokenizer="words"))

    assert approx[0].token_count != words[0].token_count


# -- sentence splitting ---------------------------------------------------------------------


def test_sentence_splitting_respects_abbreviations() -> None:
    text = "See Fig. 3 for details, e.g. the loss curve. Dr. Lee agrees. A final sentence."

    spans = split_sentences(text)

    assert [text[s:e] for s, e in spans] == [
        "See Fig. 3 for details, e.g. the loss curve.",
        "Dr. Lee agrees.",
        "A final sentence.",
    ]


# -- sizes ---------------------------------------------------------------------------------


def test_section_shorter_than_the_target_is_one_passage() -> None:
    node = make_node()
    paragraphs = paragraphs_of(node, [100, 80, 120])  # about 300 tokens, below the 500 minimum

    passages = passages_for(node, paragraphs)

    assert len(passages) == 1
    assert passages[0].token_count < CONFIG.min_tokens
    assert squash(passages[0].text) == squash("".join(p.text for p in paragraphs))
    assert not errors(validate(node, paragraphs, passages))


def test_section_longer_than_the_maximum_becomes_several_ordered_passages() -> None:
    node = make_node()
    paragraphs = paragraphs_of(node, [300] * 10)  # about 3000 tokens

    passages = passages_for(node, paragraphs)

    assert len(passages) >= 3
    assert [p.sequence for p in passages] == list(range(len(passages)))
    assert all(p.node_id == node.id for p in passages)
    body_sizes = [p.metadata["body_tokens"] for p in passages]
    assert all(size <= CONFIG.max_tokens for size in body_sizes)
    assert all(size >= CONFIG.min_tokens for size in body_sizes[:-1])
    texts = [p.text[int(p.metadata["overlap_chars"]) :] for p in passages]
    assert squash("".join(texts)) == squash("".join(p.text for p in paragraphs))
    assert not errors(validate(node, paragraphs, passages))


def test_passages_prefer_paragraph_boundaries() -> None:
    node = make_node()
    paragraphs = paragraphs_of(node, [260] * 8)

    passages = passages_for(node, paragraphs)

    for passage in passages:
        body = passage.text[int(passage.metadata["overlap_chars"]) :]
        assert body.rstrip().endswith(".")
        parts = [p for p in passage.parts if not p.is_overlap]
        assert all(p.text_start is None for p in parts)  # whole paragraphs, none cut


def test_short_trailing_passage_is_merged_when_the_maximum_allows() -> None:
    node = make_node()
    paragraphs = paragraphs_of(node, [400, 400, 100])  # 800 + 100 fits within 1000

    passages = passages_for(node, paragraphs)

    assert len(passages) == 1


def test_sections_are_never_combined_to_reach_the_target() -> None:
    first, second = make_node("A", 0), make_node("B", 1)
    paragraphs = [
        make_paragraph(first, 0, prose(100, 1), document_sequence=0),
        make_paragraph(second, 0, prose(100, 2), document_sequence=1),
    ]

    passages = generate_passages([first, second], paragraphs, CONFIG)

    assert len(passages) == 2
    assert {p.node_id for p in passages} == {first.id, second.id}
    assert [p.document_sequence for p in passages] == [0, 1]
    assert all(p.sequence == 0 for p in passages)


def test_exceptionally_long_paragraph_is_split_at_sentence_boundaries() -> None:
    node = make_node()
    long_text = prose(4000, 5)  # a single ~4000-token paragraph
    paragraph = make_paragraph(node, 0, long_text)

    passages = passages_for(node, [paragraph])

    assert len(passages) >= 4
    for passage in passages:
        body = passage.text[int(passage.metadata["overlap_chars"]) :]
        assert body.rstrip().endswith(".")  # cut between sentences
        assert passage.metadata["body_tokens"] <= CONFIG.max_tokens
        assert str(paragraph.id) in passage.metadata["split_paragraph_ids"]
    texts = [p.text[int(p.metadata["overlap_chars"]) :] for p in passages]
    assert squash("".join(texts)) == squash(long_text)
    assert not errors(validate(node, [paragraph], passages))


def test_split_paragraph_records_which_part_of_it_each_passage_holds() -> None:
    node = make_node()
    paragraph = make_paragraph(node, 0, prose(3000, 5))

    passages = passages_for(node, [paragraph])

    body_parts = [part for passage in passages for part in passage.parts if not part.is_overlap]
    assert all(part.paragraph_id == paragraph.id for part in body_parts)
    ranges = [(part.text_start, part.text_end) for part in body_parts]
    assert ranges[0][0] == 0 and ranges[-1][1] == len(paragraph.text.rstrip())
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    # Offsets of split pieces are exact for single-block paragraphs.
    first = passages[0]
    assert first.start_offset == paragraph.start_offset


def test_a_single_sentence_longer_than_the_maximum_is_split_at_words() -> None:
    node = make_node()
    giant = " ".join(["token"] * 2500) + "."  # one sentence of ~3000 tokens
    paragraph = make_paragraph(node, 0, giant)

    passages = passages_for(node, [paragraph])

    assert len(passages) >= 3
    assert all(p.metadata["body_tokens"] <= CONFIG.max_tokens for p in passages)
    texts = [p.text[int(p.metadata["overlap_chars"]) :] for p in passages]
    assert squash("".join(texts)) == squash(giant)


# -- equations, tables, captions ------------------------------------------------------------


def test_equation_stays_with_the_text_before_and_after_it() -> None:
    node = make_node()
    paragraphs = [
        make_paragraph(node, 0, prose(380, 1)),
        make_paragraph(node, 1, prose(380, 2)),
        make_paragraph(node, 2, "E = m c^2", kind="equation"),
        make_paragraph(node, 3, "where E is energy and m is mass, as in " + prose(150, 3)),
        make_paragraph(node, 4, prose(380, 4)),
    ]

    passages = passages_for(node, paragraphs)

    holding = [p for p in passages if paragraphs[2].id in {part.paragraph_id for part in p.parts if not part.is_overlap}]
    assert len(holding) == 1
    ids = {part.paragraph_id for part in holding[0].parts if not part.is_overlap}
    assert paragraphs[1].id in ids and paragraphs[3].id in ids
    assert holding[0].metadata["has_equation"] is True


def test_tables_are_kept_whole_even_beyond_the_maximum() -> None:
    node = make_node()
    rows = "\n".join(f"row {i} | " + prose(30, i) for i in range(40))  # about 1500 tokens
    table = make_paragraph(node, 1, rows, kind="table")
    paragraphs = [make_paragraph(node, 0, prose(200, 1)), table, make_paragraph(node, 2, prose(200, 2))]

    passages = passages_for(node, paragraphs)

    holding = [p for p in passages if any(part.paragraph_id == table.id for part in p.parts)]
    assert len(holding) == 1
    assert rows in holding[0].text
    assert holding[0].token_count > CONFIG.max_tokens
    assert holding[0].metadata["oversized"] == "table"
    assert holding[0].metadata["has_table"] is True
    assert "oversized_passage" in {i.code for i in validate(node, paragraphs, passages)}


def test_a_huge_table_is_split_between_rows() -> None:
    node = make_node()
    rows = [f"row {i} | " + prose(60, i) for i in range(120)]  # about 7000 tokens, over 2x the maximum
    table = make_paragraph(node, 0, "\n".join(rows), kind="table")

    passages = passages_for(node, [table])

    assert len(passages) >= 7
    for passage in passages:
        body = passage.text[int(passage.metadata["overlap_chars"]) :]
        assert all(line.startswith("row ") for line in body.splitlines())  # only whole rows
    assert squash("".join(p.text for p in passages)) == squash("\n".join(rows))


def test_figure_caption_stays_with_the_preceding_content() -> None:
    node = make_node()
    paragraphs = [
        make_paragraph(node, 0, prose(700, 1)),
        make_paragraph(node, 1, "The plot below shows the result of the experiment.", page=2),
        make_paragraph(node, 2, "Figure 4.2: Loss over training steps", kind="caption", page=2, figure_ref="Figure 4.2"),
        make_paragraph(node, 3, prose(400, 2), page=2),
    ]

    passages = passages_for(node, paragraphs)

    holding = next(p for p in passages if any(part.paragraph_id == paragraphs[2].id and not part.is_overlap for part in p.parts))
    ids = [part.paragraph_id for part in holding.parts if not part.is_overlap]
    assert paragraphs[1].id in ids
    assert holding.metadata["figure_refs"] == ["Figure 4.2"]


def test_table_caption_stays_with_its_table() -> None:
    node = make_node()
    paragraphs = [
        make_paragraph(node, 0, prose(720, 1)),
        make_paragraph(node, 1, "Table 2: Means", kind="caption", table_ref="Table 2"),
        make_paragraph(node, 2, "a | b\n1 | 2", kind="table"),
    ]

    passages = passages_for(node, paragraphs)

    holding = next(p for p in passages if any(part.paragraph_id == paragraphs[2].id for part in p.parts))
    assert paragraphs[1].id in {part.paragraph_id for part in holding.parts if not part.is_overlap}
    assert holding.metadata["table_refs"] == ["Table 2"]


def test_text_ending_with_a_colon_stays_with_the_list_it_introduces() -> None:
    node = make_node()
    paragraphs = [
        make_paragraph(node, 0, prose(720, 1)),
        make_paragraph(node, 1, "The three steps are:"),
        make_paragraph(node, 2, "first step\nsecond step\nthird step", kind="list"),
    ]

    passages = passages_for(node, paragraphs)

    holding = next(p for p in passages if any(part.paragraph_id == paragraphs[2].id for part in p.parts))
    assert paragraphs[1].id in {part.paragraph_id for part in holding.parts if not part.is_overlap}


# -- overlap -------------------------------------------------------------------------------


def test_consecutive_passages_overlap_within_the_policy() -> None:
    node = make_node()
    paragraphs = paragraphs_of(node, [300] * 8)

    passages = passages_for(node, paragraphs)

    assert passages[0].metadata["overlap_chars"] == 0
    for previous, current in zip(passages, passages[1:]):
        overlap_chars = current.metadata["overlap_chars"]
        assert overlap_chars > 0
        overlap = current.text[: overlap_chars - len(PARAGRAPH_SEPARATOR)]
        assert previous.text.endswith(overlap)
        assert 0 < current.metadata["overlap_tokens"] <= CONFIG.overlap_max_tokens
        assert current.metadata["overlap_with_passage_id"] == str(previous.id)
        flagged = [part for part in current.parts if part.is_overlap]
        assert flagged and all(part.position < len([p for p in current.parts if p.is_overlap]) for part in flagged)


def test_overlap_text_is_never_counted_as_new_content() -> None:
    node = make_node()
    paragraphs = paragraphs_of(node, [300] * 8)

    passages = passages_for(node, paragraphs)

    new_text = "".join(p.text[int(p.metadata["overlap_chars"]) :] for p in passages)
    assert squash(new_text) == squash("".join(p.text for p in paragraphs))


def test_overlap_never_crosses_into_another_section() -> None:
    first, second = make_node("A", 0), make_node("B", 1)
    paragraphs = [make_paragraph(first, 0, prose(900, 1), document_sequence=0), make_paragraph(second, 0, prose(900, 2), document_sequence=1)]

    passages = generate_passages([first, second], paragraphs, CONFIG)

    second_node_first = next(p for p in passages if p.node_id == second.id and p.sequence == 0)
    assert second_node_first.metadata["overlap_chars"] == 0


def test_overlap_can_be_switched_off() -> None:
    node = make_node()
    paragraphs = paragraphs_of(node, [300] * 8)

    passages = passages_for(node, paragraphs, SegmentationConfig(overlap_ratio=0))

    assert all(p.metadata["overlap_chars"] == 0 for p in passages)
    assert all(not part.is_overlap for p in passages for part in p.parts)


def test_overlap_is_not_taken_from_equations_or_tables() -> None:
    node = make_node()
    paragraphs = [
        make_paragraph(node, 0, prose(700, 1)),
        make_paragraph(node, 1, "x = 1", kind="equation"),
        make_paragraph(node, 2, prose(700, 2)),
    ]
    config = SegmentationConfig(target_tokens=300, min_tokens=200, max_tokens=400)

    passages = passages_for(node, paragraphs, config)

    assert len(passages) >= 2
    for passage in passages:
        assert "x = 1" not in passage.text[: int(passage.metadata["overlap_chars"])]


def test_passage_sizes_follow_a_changed_configuration() -> None:
    node = make_node()
    paragraphs = paragraphs_of(node, [150] * 20)
    small = SegmentationConfig(target_tokens=300, min_tokens=200, max_tokens=400)

    default_passages = passages_for(node, paragraphs)
    small_passages = passages_for(node, paragraphs, small)

    assert len(small_passages) > len(default_passages)
    assert all(p.metadata["body_tokens"] <= 400 for p in small_passages)
    assert not errors(validate(node, paragraphs, small_passages, small))


# -- page mapping --------------------------------------------------------------------------


def test_passages_take_their_pages_from_the_paragraphs_not_from_text_length() -> None:
    node = make_node()
    paragraphs = [
        make_paragraph(node, 0, prose(300, 1), page=42),
        make_paragraph(node, 1, prose(300, 2), page=43),
        make_paragraph(node, 2, prose(300, 3), page=44),
    ]

    passages = passages_for(node, paragraphs)

    pages = {(p.page_start, p.page_end) for p in passages}
    all_pages = [page for p in passages for page in (p.page_start, p.page_end)]
    assert min(all_pages) == 42 and max(all_pages) == 44
    assert all(start in (42, 43, 44) and end in (42, 43, 44) for start, end in pages)
    assert all(p.page_status == "exact" for p in passages)
    assert set(passages[0].metadata["page_numbers"]) <= {42, 43, 44}


def test_a_paragraph_spanning_pages_gives_its_passage_the_full_range() -> None:
    node = make_node()
    spanning = make_paragraph(node, 0, prose(200, 1), page=7, page_end=8)

    passages = passages_for(node, [spanning])

    assert (passages[0].page_start, passages[0].page_end) == (7, 8)


def test_split_pieces_of_a_multi_page_paragraph_use_the_recorded_block_pages() -> None:
    node = make_node()
    first, second = prose(1800, 1), prose(1800, 2)
    text = first + " " + second
    paragraph = make_paragraph(
        node, 0, text, page=10, page_end=11,
        block_spans=[
            {"sequence": 0, "page": 10, "char_start": 0, "char_end": len(first)},
            {"sequence": 1, "page": 11, "char_start": len(first) + 1, "char_end": len(text)},
        ],
    )
    paragraph.offsets_exact = False

    passages = passages_for(node, [paragraph])

    assert passages[0].page_start == 10 and passages[0].page_end == 10
    assert passages[-1].page_start == 11 and passages[-1].page_end == 11
    assert all(p.start_offset is None for p in passages)  # offsets of a merged paragraph are not guessed


def test_unknown_pages_stay_unknown() -> None:
    node = make_node()
    paragraphs = [make_paragraph(node, i, prose(200, i), page=None) for i in range(3)]

    passages = passages_for(node, paragraphs)

    assert all(p.page_start is None and p.page_end is None and p.page_status == "unknown" for p in passages)


def test_partially_known_pages_are_marked_partial() -> None:
    node = make_node()
    paragraphs = [make_paragraph(node, 0, prose(200, 1), page=5), make_paragraph(node, 1, prose(200, 2), page=None)]

    passages = passages_for(node, paragraphs)

    assert passages[0].page_status == "partial"
    assert (passages[0].page_start, passages[0].page_end) == (5, 5)


def test_passage_keeps_bounding_boxes_and_equation_references() -> None:
    node = make_node()
    equation_id = str(uuid.uuid4())
    paragraphs = [
        make_paragraph(node, 0, prose(100, 1), bboxes=[{"sequence": 0, "page": 1, "bbox": [1, 2, 3, 4], "origin": "BOTTOMLEFT"}]),
        make_paragraph(node, 1, "a = b", kind="equation", equation_ids=[equation_id]),
    ]

    passage = passages_for(node, paragraphs)[0]

    assert passage.metadata["bboxes"] == [{"sequence": 0, "page": 1, "bbox": [1, 2, 3, 4], "origin": "BOTTOMLEFT"}]
    assert passage.metadata["equation_ids"] == [equation_id]
    assert [part.paragraph_id for part in passage.parts] == [p.id for p in paragraphs]


def test_passages_from_a_detected_book_keep_every_reference() -> None:
    from app.structure.detector import StructureDetector
    from tests.structure_helpers import numbered_textbook

    document = numbered_textbook().build()
    result = StructureDetector().detect(document)
    passages = generate_passages(result.nodes, result.paragraphs, CONFIG)

    node_ids = {n.id for n in result.nodes}
    assert passages and all(p.node_id in node_ids and p.text.strip() for p in passages)
    assert all(p.page_start is not None and p.page_end >= p.page_start for p in passages)
    assert [p.document_sequence for p in passages] == list(range(len(passages)))
    assert not errors(validate_passages(document, result.nodes, result.paragraphs, passages, CONFIG, TOKENIZER))
