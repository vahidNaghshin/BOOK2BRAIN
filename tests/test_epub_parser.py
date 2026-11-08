import uuid
import zipfile
from pathlib import Path

import pytest

from app.parsing.epub_parser import EpubParser
from app.parsing.models import BlockType, ExtractionStatus
from tests.factories import make_epub_book


def parse(path: Path):
    return EpubParser().parse(path, document_id=uuid.uuid4(), original_filename=path.name)


@pytest.fixture
def write(tmp_path: Path):
    def _write(data: bytes, name: str = "book.epub") -> Path:
        path = tmp_path / name
        path.write_bytes(data)
        return path

    return _write


TWO_CHAPTERS = [
    ("ch1.xhtml", "Chapter One", "<h1>The Beginning</h1><p>It all started here.</p><p>Second paragraph.</p>"),
    ("ch2.xhtml", "Chapter Two", "<h1>The Middle</h1><p>Things happened next.</p>"),
]


def test_normal_epub_is_extracted(write) -> None:
    result = parse(write(make_epub_book(TWO_CHAPTERS, title="My Book")))

    assert result.status is ExtractionStatus.SUCCESS
    assert result.file_format == "epub" and result.extraction_method == "ebooklib"
    assert [b.text for b in result.blocks] == [
        "The Beginning", "It all started here.", "Second paragraph.", "The Middle", "Things happened next.",
    ]
    assert result.metadata["title"] == "My Book" and result.metadata["language"] == "en"


def test_multiple_chapters_keep_chapter_level_source_references(write) -> None:
    result = parse(write(make_epub_book(TWO_CHAPTERS)))

    first, middle = result.blocks[0], result.blocks[3]
    assert first.source_location["href"] == "ch1.xhtml"
    assert first.source_location["chapter_title"] == "Chapter One"
    assert middle.source_location["href"] == "ch2.xhtml"
    assert middle.source_location["chapter_title"] == "Chapter Two"
    assert first.source_location["item_id"] != middle.source_location["item_id"]
    assert {b.source_location["spine_index"] for b in result.blocks} == {1, 2}  # 0 is the nav doc


def test_reading_order_follows_the_spine_not_file_order(write) -> None:
    result = parse(write(make_epub_book(TWO_CHAPTERS, spine_order=[1, 0])))

    headings = [b.text for b in result.blocks if b.block_type is BlockType.HEADING]
    assert headings == ["The Middle", "The Beginning"]
    assert [b.sequence_number for b in result.blocks] == list(range(len(result.blocks)))


def test_headings_are_extracted_with_levels_and_section_titles(write) -> None:
    body = "<h1>Main</h1><p>Intro.</p><h2>Sub</h2><p>Detail.</p>"

    result = parse(write(make_epub_book([("c.xhtml", "C", body)])))

    h1, intro, h2, detail = result.blocks
    assert (h1.block_type, h1.metadata) == (BlockType.HEADING, {"heading_level": 1})
    assert (h2.block_type, h2.metadata) == (BlockType.HEADING, {"heading_level": 2})
    assert intro.section_title == "Main" and detail.section_title == "Sub"


def test_chapter_title_falls_back_to_first_heading_without_a_toc(write) -> None:
    result = parse(write(make_epub_book([("c.xhtml", "ignored", "<h1>Real Title</h1><p>Text here.</p>")], toc=False)))

    assert {b.source_location["chapter_title"] for b in result.blocks} == {"Real Title"}


def test_page_numbers_are_never_invented(write) -> None:
    result = parse(write(make_epub_book(TWO_CHAPTERS)))

    assert result.page_count is None
    assert all(b.page_number is None for b in result.blocks)


def test_scripts_styles_and_navigation_are_removed(write) -> None:
    body = (
        "<style>p {color: red}</style><script>alert('x')</script>"
        "<nav epub:type='toc'><ol><li>Table of contents entry</li></ol></nav>"
        "<p>Visible text.</p>"
    )

    result = parse(write(make_epub_book([("c.xhtml", "C", body)])))

    assert [b.text for b in result.blocks] == ["Visible text."]


def test_html_entities_are_decoded_and_inline_markup_kept_together(write) -> None:
    body = "<p>Tom &amp; Jerry &mdash; it&#8217;s <em>un</em>believably <b>f</b>un &lt;3 &copy;</p>"

    result = parse(write(make_epub_book([("c.xhtml", "C", body)])))

    assert result.blocks[0].text == "Tom & Jerry \u2014 it\u2019s unbelievably fun <3 \u00a9"


def test_line_breaks_inside_a_paragraph_are_preserved(write) -> None:
    body = "<p>Roses are red,<br/>Violets are blue,<br />Sugar is sweet</p>"

    result = parse(write(make_epub_book([("c.xhtml", "C", body)])))

    assert result.blocks[0].text == "Roses are red,\nViolets are blue,\nSugar is sweet"


def test_source_whitespace_is_collapsed_like_a_browser(write) -> None:
    body = "<p>Words   split\n     across\n lines</p>"

    result = parse(write(make_epub_book([("c.xhtml", "C", body)])))

    assert result.blocks[0].text == "Words split across lines"


def test_lists_tables_captions_and_preformatted_text(write) -> None:
    body = (
        "<ul><li>One<ul><li>Nested</li></ul></li><li>Two</li></ul>"
        "<ol><li>First</li></ol>"
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        "<figure><figcaption>A caption</figcaption></figure>"
        "<pre>  indented\n    code</pre>"
    )

    result = parse(write(make_epub_book([("c.xhtml", "C", body)])))

    by_type = {t: [b for b in result.blocks if b.block_type is t] for t in BlockType}
    assert [b.text for b in by_type[BlockType.LIST]] == ["One", "Nested", "Two", "First"]
    assert by_type[BlockType.LIST][3].metadata == {"enumerated": True}
    assert by_type[BlockType.TABLE][0].text == "A | B\n1 | 2"
    assert by_type[BlockType.CAPTION][0].text == "A caption"
    assert by_type[BlockType.OTHER][0].text == "  indented\n    code"
    assert by_type[BlockType.OTHER][0].metadata["preformatted"] is True


def test_nested_containers_are_walked_and_stray_text_becomes_a_paragraph(write) -> None:
    body = "<div><div><p>Deep paragraph.</p></div><div>Bare text in a div</div></div>"

    result = parse(write(make_epub_book([("c.xhtml", "C", body)])))

    assert [b.text for b in result.blocks] == ["Deep paragraph.", "Bare text in a div"]


def test_chapters_are_not_flattened_into_one_string(write) -> None:
    result = parse(write(make_epub_book(TWO_CHAPTERS)))

    assert len(result.blocks) == 5
    assert all("\n\n" not in b.text for b in result.blocks)


def test_garbage_file_fails_gracefully(write) -> None:
    result = parse(write(b"this is not a zip archive at all"))

    assert result.status is ExtractionStatus.FAILED
    assert [e.code for e in result.errors] == ["corrupt_document"]
    assert "EPUB" in result.errors[0].message


def test_zip_without_epub_structure_fails_gracefully(write, tmp_path: Path) -> None:
    path = tmp_path / "broken.epub"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")

    result = parse(path)

    assert result.status is ExtractionStatus.FAILED
    assert result.errors[0].code == "corrupt_document"


def test_truncated_epub_fails_gracefully(write) -> None:
    data = make_epub_book(TWO_CHAPTERS)

    result = parse(write(data[: len(data) // 2]))

    assert result.status is ExtractionStatus.FAILED
    assert result.errors[0].code == "corrupt_document"


def test_epub_without_text_content_fails_as_empty(write) -> None:
    result = parse(write(make_epub_book([("c.xhtml", "C", "<div></div><p>   </p><img src='x.png'/>")])))

    assert result.status is ExtractionStatus.FAILED
    assert [e.code for e in result.errors] == ["empty_document"]


def test_a_broken_chapter_does_not_lose_the_rest_of_the_book(write, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.parsing import epub_parser

    original = epub_parser._ItemExtractor.run

    def flaky(self, soup):  # type: ignore[no-untyped-def]
        if self.location["href"] == "ch1.xhtml":
            raise ValueError("chapter exploded")
        return original(self, soup)

    monkeypatch.setattr(epub_parser._ItemExtractor, "run", flaky)

    result = parse(write(make_epub_book(TWO_CHAPTERS)))

    assert result.status is ExtractionStatus.PARTIAL
    assert [b.text for b in result.blocks] == ["The Middle", "Things happened next."]
    assert result.warnings[0].code == "chapter_parse_failed"
    assert result.warnings[0].details["item_id"]


def test_progress_is_reported_per_spine_item(write) -> None:
    events: list[tuple[str, int, int]] = []

    EpubParser().parse(
        write(make_epub_book(TWO_CHAPTERS)),
        document_id=uuid.uuid4(),
        original_filename="b.epub",
        on_progress=lambda stage, cur, total: events.append((stage.value, cur, total)),
    )

    assert events[-1] == ("parsing", 3, 3)  # nav + 2 chapters
    assert all(stage == "parsing" for stage, _, _ in events)
