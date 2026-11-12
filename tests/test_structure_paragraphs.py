from app.structure.detector import StructureDetector
from app.structure.paragraphs import continues_paragraph
from app.structure.models import SourceBlock
from tests.structure_helpers import DocBuilder


def build(builder: DocBuilder):
    document = builder.build()
    return document, StructureDetector().detect(document)


def block(text: str, page: int | None) -> SourceBlock:
    return SourceBlock(sequence=0, text=text, block_type="paragraph", page=page)


def test_paragraph_continuing_across_a_page_break_is_kept_whole() -> None:
    b = DocBuilder()
    b.title("Spanning").new_page().heading("1. Flow").prose(80, 1)
    b.para("The estimator converges quickly when the step size is chosen carefully and the")
    b.new_page().para("gradient noise is small, which is the case for most of the experiments we ran.")
    b.para("A new paragraph starts here and is separate from the previous one.")
    document, result = build(b)

    paragraphs = [p for p in result.paragraphs if p.kind == "paragraph"]
    spanning = next(p for p in paragraphs if p.text.startswith("The estimator"))
    assert spanning.text.endswith("most of the experiments we ran.")
    assert "the gradient noise" in spanning.text  # joined with a single space, nothing lost
    assert (spanning.page_start, spanning.page_end) == (2, 3)
    assert spanning.metadata["continued_across_pages"] is True
    assert [s["page"] for s in spanning.metadata["block_spans"]] == [2, 3]
    assert len(spanning.block_sequences) == 2
    assert spanning.offsets_exact is False
    assert (spanning.start_offset, spanning.end_offset) == (
        document.blocks[spanning.block_sequences[0]].start_offset,
        document.blocks[spanning.block_sequences[-1]].end_offset,
    )
    following = next(p for p in paragraphs if p.text.startswith("A new paragraph"))
    assert following.page_start == following.page_end == 3


def test_a_finished_paragraph_before_a_page_break_is_not_merged() -> None:
    assert not continues_paragraph(block("The end of a sentence.", 4), block("another starts here", 5))
    assert not continues_paragraph(block("An unfinished clause", 4), block("Starting with a capital", 5))
    assert not continues_paragraph(block("An unfinished clause", 4), block("continues lowercase", 4))
    assert not continues_paragraph(block("An unfinished clause", None), block("continues lowercase", None))
    assert continues_paragraph(block("An unfinished clause", 4), block("continues lowercase", 5))
    assert continues_paragraph(block("A hyphen-", 4), block("Ated word", 5))


def test_hyphenated_break_keeps_the_hyphen_and_loses_nothing() -> None:
    b = DocBuilder()
    b.title("Hyphens").new_page().heading("1. Words").prose(60, 1)
    b.para("The representa-")
    b.new_page().para("tion of the signal matters.")
    _, result = build(b)

    joined = next(p for p in result.paragraphs if p.text.startswith("The representa"))
    assert joined.text == "The representa-tion of the signal matters."


def test_consecutive_list_items_become_one_list_paragraph() -> None:
    b = DocBuilder()
    b.title("Lists").new_page().heading("1. Steps").prose(60, 1)
    b.item("collect data").item("clean data").item("train a model")
    b.prose(40, 2)
    _, result = build(b)

    lists = [p for p in result.paragraphs if p.kind == "list"]
    assert len(lists) == 1
    assert lists[0].text == "collect data\nclean data\ntrain a model"
    assert len(lists[0].block_sequences) == 3


def test_paragraphs_never_merge_across_sections() -> None:
    b = DocBuilder()
    b.title("Sections").new_page().heading("1. One").para("This paragraph ends without punctuation and")
    b.new_page().heading("2. Two").para("continues in a different section.")
    _, result = build(b)

    texts = [p.text for p in result.paragraphs]
    assert "This paragraph ends without punctuation and" in texts
    assert "continues in a different section." in texts


def test_headings_are_not_repeated_as_paragraph_text() -> None:
    b = DocBuilder()
    b.title("No Echo").new_page().heading("1. Introduction").prose(60, 1).heading("1.1 Scope", 2).prose(60, 2)
    _, result = build(b)

    assert all(p.text not in ("1. Introduction", "1.1 Scope") for p in result.paragraphs)
    assert "heading_duplicated_in_body" not in {i.code for i in result.issues}


def test_paragraph_identity_ordering_and_association() -> None:
    document, result = build(
        DocBuilder()
        .title("Order")
        .new_page()
        .heading("1. A")
        .prose(60, 1)
        .prose(60, 2)
        .heading("2. B")
        .prose(60, 3)
    )

    assert len({p.id for p in result.paragraphs}) == len(result.paragraphs)
    assert [p.document_sequence for p in result.paragraphs] == list(range(len(result.paragraphs)))
    node_ids = {n.id for n in result.nodes}
    assert all(p.node_id in node_ids for p in result.paragraphs)
    for node in result.nodes:
        own = [p for p in result.paragraphs if p.node_id == node.id]
        assert [p.sequence for p in own] == list(range(len(own)))


def test_single_block_paragraphs_have_exact_offsets_into_the_canonical_text() -> None:
    document, result = build(DocBuilder().title("Offsets").new_page().heading("1. A").prose(60, 1).prose(60, 2))
    canonical = "\n\n".join(b.text for b in document.blocks)

    for paragraph in result.paragraphs:
        if paragraph.offsets_exact:
            assert canonical[paragraph.start_offset : paragraph.end_offset] == paragraph.text


def test_paragraph_keeps_bounding_boxes_for_later_page_image_lookup() -> None:
    _, result = build(DocBuilder().title("Boxes").new_page().heading("1. A").prose(60, 1))

    paragraph = next(p for p in result.paragraphs if p.kind == "paragraph")
    boxes = paragraph.metadata["bboxes"]
    assert boxes and boxes[0]["page"] == 2 and len(boxes[0]["bbox"]) == 4
    assert paragraph.metadata["block_sequences"] == paragraph.block_sequences


def test_equation_blocks_keep_their_equation_record_ids() -> None:
    import uuid

    b = DocBuilder()
    b.title("Math").new_page().heading("1. A").prose(60, 1).equation("x = y")
    document = b.build()
    equation_id = uuid.uuid4()
    equation_block = next(blk for blk in document.blocks if blk.block_type == "equation")
    document.equation_ids_by_block[equation_block.id] = equation_id

    result = StructureDetector().detect(document)

    equation = next(p for p in result.paragraphs if p.kind == "equation")
    assert equation.metadata["equation_ids"] == [str(equation_id)]
