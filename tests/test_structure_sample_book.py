"""Structure detection on a real textbook, through the real Day 3 PDF pipeline.

Uses the first 70 pages of the sample book found in storage/original (cover, copyright, contents,
preface, acknowledgements, author biography, chapter 1 and the start of chapter 2). Skipped when the
sample is absent or the Docling models are unavailable, so it never blocks a fresh checkout.
"""

import uuid
from pathlib import Path

import pymupdf
import pytest

from app.config import PROJECT_ROOT, Settings
from app.parsing.models import ExtractionStatus
from app.parsing.parser_factory import ParserFactory
from app.parsing.text_normalizer import normalize_blocks
from app.structure.detector import StructureDetector
from app.structure.models import Severity
from app.structure.segmentation import SegmentationConfig, generate_passages
from app.structure.validation import validate_passages
from tests.structure_helpers import document_from_parsing

pytestmark = pytest.mark.sample_book
EXCERPT_PAGES = 70


def sample_pdf() -> Path | None:
    found = sorted((PROJECT_ROOT / "storage" / "original").glob("*/*.pdf"))
    return found[0] if found else None


@pytest.fixture(scope="module")
def detected(docling_ready, tmp_path_factory):
    source = sample_pdf()
    if source is None:
        pytest.skip("no sample book in storage/original")
    excerpt = tmp_path_factory.mktemp("sample") / "excerpt.pdf"
    with pymupdf.open(source) as full:
        if full.page_count < EXCERPT_PAGES:
            pytest.skip("the sample book is shorter than the excerpt used by this test")
        part = pymupdf.open()
        part.insert_pdf(full, from_page=0, to_page=EXCERPT_PAGES - 1)
        part.save(excerpt)

    settings = Settings(
        database_url="postgresql://u:p@localhost/unused",
        equation_extraction_enabled=False,
        _env_file=None,
    )
    parser = ParserFactory(settings).get_parser("pdf")
    result = parser.parse(excerpt, document_id=uuid.uuid4(), original_filename="excerpt.pdf")
    if result.status is ExtractionStatus.FAILED:
        pytest.skip("the sample book could not be parsed: " + "; ".join(e.message for e in result.errors))
    blocks, _ = normalize_blocks(result.blocks)
    document = document_from_parsing(result.model_copy(update={"blocks": blocks}))
    return document, StructureDetector().detect(document)


def top_level(result):
    return [(n.node_type, n.number, n.title) for n in result.nodes if n.level == 0 or n.node_type in ("preface", "table_of_contents")]


def test_front_matter_and_chapters_are_found(detected) -> None:
    _, result = detected
    outline = top_level(result)

    assert ("front_matter", None, "Front Matter") in outline
    assert ("table_of_contents", None, "Contents") in outline
    assert ("preface", None, "Preface") in outline
    chapters = [(n, t) for kind, n, t in outline if kind == "chapter"]
    assert chapters[0] == ("1", "An Introduction to Recommender Systems")
    assert chapters[1] == ("2", "Neighborhood-Based Collaborative Filtering")


def test_title_author_and_copyright_come_from_the_title_pages(detected) -> None:
    _, result = detected
    guess = result.metadata_guess

    assert guess["title"] == "Recommender Systems"
    assert guess["author"] == "Charu C. Aggarwal"
    assert guess["copyright_page"] == 5
    kinds = {n.node_type for n in result.nodes}
    assert {"book_title", "copyright", "acknowledgement", "about_author"} <= kinds


def test_multi_level_numbering_becomes_a_nested_hierarchy(detected) -> None:
    _, result = detected
    by_number = {n.number: n for n in result.nodes if n.number}

    assert by_number["1.2"].node_type == "section" and by_number["1.2"].level == 1
    assert by_number["1.3.1"].node_type == "subsection" and by_number["1.3.1"].level == 2
    assert by_number["1.3.1.1"].node_type == "subsubsection" and by_number["1.3.1.1"].level == 3
    assert by_number["1.3.1.1"].parent_id == by_number["1.3.1"].id
    assert by_number["1.3.1"].parent_id == by_number["1.3"].id
    assert by_number["1.3"].parent_id == by_number["1"].id


def test_table_of_contents_pages_are_content_not_headings(detected) -> None:
    _, result = detected

    contents = next(n for n in result.nodes if n.node_type == "table_of_contents")
    toc_paragraphs = [p for p in result.paragraphs if p.node_id == contents.id]
    assert len(toc_paragraphs) > 3
    assert sum(1 for n in result.nodes if n.node_type == "chapter") == 2  # TOC lines created no chapters
    assert result.summary["toc_entries"] > 50


def test_pages_come_from_the_pdf_and_chapters_span_their_pages(detected) -> None:
    _, result = detected
    chapter = next(n for n in result.nodes if n.number == "1" and n.node_type == "chapter")

    assert (chapter.page_start, chapter.page_end) == (23, 50)
    assert all(p.page_start is not None for p in result.paragraphs)


def test_nothing_is_lost_and_validation_finds_no_errors(detected) -> None:
    document, result = detected
    covered = sorted(s for n in result.nodes for s in n.heading_blocks) + sorted(
        s for p in result.paragraphs for s in p.block_sequences
    )

    assert sorted(covered) == [b.sequence for b in document.blocks if b.text.strip()]
    assert not [i for i in result.issues if i.severity is Severity.ERROR]
    # The contents list the whole book but only 70 pages are parsed here, so a mismatch is expected.
    assert {i.code for i in result.issues if i.severity is Severity.WARNING} <= {"toc_mismatch"}


def test_passages_of_the_real_book_validate_and_keep_pages(detected) -> None:
    document, result = detected
    config = SegmentationConfig()
    passages = generate_passages(result.nodes, result.paragraphs, config)

    issues = validate_passages(document, result.nodes, result.paragraphs, passages, config, config.build_tokenizer())

    assert not [i for i in issues if i.severity is Severity.ERROR]
    assert all(p.page_start is not None for p in passages)
    assert max(p.metadata["body_tokens"] for p in passages if not p.metadata.get("oversized")) <= config.max_tokens
