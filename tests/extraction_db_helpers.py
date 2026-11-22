"""Helpers for the extraction pipeline tests that use the test database and a mocked LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.extraction.model_config import LlmModelConfig, ModelRegistry
from app.extraction.providers import LlmResponse
from app.extraction.settings import ExtractionSettings
from app.knowledge.types import KnowledgeType
from app.services.knowledge_extraction_service import KnowledgeExtractionService
from database.models import Book, Document, Passage, StructuralNode
from tests.extraction_helpers import make_config
from tests.knowledge_helpers import BUILDERS
from tests.test_models import make_book, make_document

TARGET = re.compile(r'<passage id="([0-9a-f-]+)" role="target"')
CATEGORY = re.compile(r"Extract (\w+) knowledge")
RELATIONSHIP_PASSAGE = re.compile(r'<passage id="([0-9a-f-]+)" pages=')


@dataclass
class Library:
    """One book, one document. Reading order (document_sequence 0..7) and structure:

    front (front_matter): f1
    ch1 (chapter): c1a, c1b;   s1 (section in ch1): s1a, s1b;   ss1 (subsection in s1): ss1a
    ch2 (chapter): c2a, c2b
    """

    book: Book
    document: Document
    nodes: dict[str, StructuralNode]
    passages: dict[str, Passage] = field(default_factory=dict)

    @property
    def ordered(self) -> list[Passage]:
        return sorted(self.passages.values(), key=lambda p: p.document_sequence)


STRUCTURE = [
    ("front", None, "front_matter", "Preface"),
    ("ch1", None, "chapter", "Entropy"),
    ("s1", "ch1", "section", "The second law"),
    ("ss1", "s1", "subsection", "Statistical view"),
    ("ch2", None, "chapter", "Engines"),
]
PASSAGES = [("f1", "front"), ("c1a", "ch1"), ("c1b", "ch1"), ("s1a", "s1"), ("s1b", "s1"), ("ss1a", "ss1"), ("c2a", "ch2"), ("c2b", "ch2")]


def store_library(session: Session, title: str = "Thermodynamics", *, text_size: int = 0) -> Library:
    book = make_book(title)
    book.publisher, book.publication_year, book.subtitle = "Test Press", 1999, "A primer"
    document = make_document(book)
    session.add_all([book, document])
    session.flush()
    nodes: dict[str, StructuralNode] = {}
    for sequence, (key, parent, kind, node_title) in enumerate(STRUCTURE):
        node = StructuralNode(
            book_id=book.id, document_id=document.id, parent_id=nodes[parent].id if parent else None,
            node_type=kind, title=node_title, level=0 if parent is None else nodes[parent].level + 1, sequence=sequence,
        )
        session.add(node)
        session.flush()
        nodes[key] = node
    library = Library(book, document, nodes)
    for sequence, (name, node_key) in enumerate(PASSAGES):
        passage = Passage(
            book_id=book.id, document_id=document.id, structural_node_id=nodes[node_key].id,
            text=f"Passage {name}. " + "Entropy is a measure of the disorder of a system. " + "Filler sentence. " * text_size,
            sequence=sequence, document_sequence=sequence, page_start=sequence + 1, page_end=sequence + 1,
            page_status="exact", token_count=10,
        )
        session.add(passage)
        library.passages[name] = passage
    session.commit()
    return library


def add_second_document(session: Session, library: Library, *, older: bool) -> list[Passage]:
    """A second document of the same book with two passages, created before or after the first one."""
    document = make_document(library.book)
    document.created_at = library.document.created_at + timedelta(hours=-1 if older else 1)
    session.add(document)
    session.flush()
    node = StructuralNode(book_id=library.book.id, document_id=document.id, node_type="chapter", title="Other edition", level=0, sequence=0)
    session.add(node)
    session.flush()
    passages = [
        Passage(
            book_id=library.book.id, document_id=document.id, structural_node_id=node.id, text=f"Other {i}.", sequence=i,
            document_sequence=i, page_start=i + 1, page_end=i + 1, page_status="exact", token_count=5,
        )
        for i in range(2)
    ]
    session.add_all(passages)
    session.commit()
    return passages


# -- the mocked model ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    target_id: str
    category: KnowledgeType | None  # None for a relationship call
    number: int  # how many calls the model has had so far, starting at 1
    units: tuple[dict, ...] = ()  # the knowledge units shown in a relationship call

    @property
    def is_relationship(self) -> bool:
        return self.category is None

    def unit(self, title: str, *, in_passage: bool = True) -> str:
        """The id of the unit with this title in a relationship call (from the passage, or from earlier context)."""
        return next(u["id"] for u in self.units if u["title"] == title and u["in_passage"] == in_passage)


class FunctionProvider:
    """Answers each request with `answer(call)`: a JSON string, an LlmResponse, or an exception to raise."""

    def __init__(self, config: LlmModelConfig, answer) -> None:
        self.config, self.answer, self.requests = config, answer, []

    def complete(self, request):
        self.requests.append(request)
        content = request.messages[0].content
        if "Knowledge units:" in content:
            units = tuple(json.loads(content.split("Knowledge units:\n", 1)[1]))
            call = Call(RELATIONSHIP_PASSAGE.search(content).group(1), None, len(self.requests), units)
        else:
            call = Call(TARGET.search(content).group(1), KnowledgeType(CATEGORY.search(content).group(1)), len(self.requests))
        reply = self.answer(call)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, LlmResponse):
            return reply
        return LlmResponse(text=reply, model=self.config.model, input_tokens=100, output_tokens=40, finish_reason="stop")


def item(call: Call, **fields: Any) -> dict[str, Any]:
    """One extracted item for the call's passage; `content` entries override the category's valid sample."""
    content = BUILDERS[call.category](**fields.pop("content", {}))
    return {
        "content": content,
        "sources": [{"passage_id": call.target_id, "role": "primary"}],
        "basis": "explicit",
        "confidence": 0.8,
        "is_uncertain": False,
    } | fields


def reply(*items: dict[str, Any]) -> str:
    return json.dumps({"items": list(items)})


def one_item(call: Call) -> str:
    if call.is_relationship:
        return json.dumps({"relationships": []})
    return reply(item(call))


def relationship(call: Call, source: str, target: str, kind: str = "related_to", *, target_in_passage: bool = True, **fields: Any) -> dict[str, Any]:
    """One relationship between two units of a relationship call, named by title."""
    return {
        "source_id": call.unit(source), "target_id": call.unit(target, in_passage=target_in_passage), "type": kind,
        "explanation": "The passage says so.", "confidence": 0.8,
    } | fields


def relationships_reply(*relationships: dict[str, Any]) -> str:
    return json.dumps({"relationships": list(relationships), "notes": None})


def with_relationships(knowledge, relations):
    """An answer function: `knowledge(call)` for knowledge calls, `relations(call)` for relationship calls."""
    return lambda call: relations(call) if call.is_relationship else knowledge(call)


def make_service(
    session: Session,
    answer=one_item,
    *,
    configs: list[LlmModelConfig] | None = None,
    default: str | None = "local-test",
    overrides: dict[str, str] | None = None,
    environ: dict[str, str] | None = None,
    **settings: Any,
) -> tuple[KnowledgeExtractionService, dict[str, FunctionProvider]]:
    configs = configs or [make_config()]
    providers = {c.id: FunctionProvider(c, answer) for c in configs}
    service = KnowledgeExtractionService(
        session,
        ModelRegistry(configs, default_id=default, overrides=overrides),
        ExtractionSettings(retry_backoff_seconds=0, **settings),
        environ=environ if environ is not None else {"TEST_LLM_KEY": "test-value"},
        provider_factory=lambda c: providers[c.id],
        sleep=lambda _s: None,
    )
    return service, providers


def only_provider(providers: dict[str, FunctionProvider]) -> FunctionProvider:
    return next(iter(providers.values()))



# -- Phase 1 equations -----------------------------------------------------------------------------


def add_equation(session: Session, library: Library, passage: str, **fields: Any):
    """A stored Phase 1 equation placed in a passage (listed in the passage's metadata, as Phase 1 does)."""
    import uuid

    from database.models import Equation

    values = {
        "document_id": library.document.id,
        "sequence_number": sum(1 for p in library.passages.values() if (p.passage_metadata or {}).get("equation_ids")) + fields.pop("n", 0),
        "identity_hash": uuid.uuid4().hex + uuid.uuid4().hex[:32],
        "latex": r"dS = \frac{dQ}{T}",
        "original_latex": r"dS = \frac { dQ } { T }",
        "plain_text": "dS = dQ / T",
        "equation_type": "display",
        "extraction_method": "test",
        "page_number": library.passages[passage].page_start,
        "surrounding_text": "Entropy change of a reversible process.",
        "crop_reference": f"equations/{library.document.id}/{uuid.uuid4().hex[:8]}.png",
        "validation_status": "valid",
    } | fields
    equation = Equation(**values)
    session.add(equation)
    session.flush()
    target = library.passages[passage]
    ids = [*(target.passage_metadata or {}).get("equation_ids", []), str(equation.id)]
    target.passage_metadata = {**(target.passage_metadata or {}), "equation_ids": ids}
    session.commit()
    return equation
