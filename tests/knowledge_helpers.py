"""Builders for Phase 2 knowledge tests: valid content of each type, and a small stored book."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.knowledge.schemas import KnowledgeUnitCreate
from app.knowledge.types import KnowledgeType
from database.models import Equation, Passage, StructuralNode
from tests.test_models import make_book, make_document


def concept_data(**overrides: Any) -> dict[str, Any]:
    return {
        "knowledge_type": "concept",
        "name": "Entropy",
        "definition": "A measure of the disorder of a system.",
        "explanation": "Higher entropy means more ways to arrange the parts.",
        "characteristics": ["extensive", "state function"],
        "related_concepts": ["Temperature"],
        "examples": ["A melting ice cube"],
    } | overrides


def principle_data(**overrides: Any) -> dict[str, Any]:
    return {
        "knowledge_type": "principle",
        "name": "Second law",
        "statement": "The entropy of an isolated system never decreases.",
        "explanation": "Heat does not flow from cold to hot by itself.",
        "conditions": ["isolated system"],
        "limitations": ["statistical at very small scales"],
        "supporting_evidence": ["Heat engines never reach 100% efficiency"],
    } | overrides


def method_data(**overrides: Any) -> dict[str, Any]:
    return {
        "knowledge_type": "method",
        "name": "Calorimetry",
        "purpose": "Measure the heat exchanged in a process.",
        "preconditions": ["insulated container"],
        "steps": ["Weigh the sample", "Heat it", "Record the temperature change"],
        "inputs": ["sample mass"],
        "outputs": ["heat capacity"],
        "expected_results": ["a stable final temperature"],
        "limitations": ["heat loss to the surroundings"],
    } | overrides


def framework_data(**overrides: Any) -> dict[str, Any]:
    return {
        "knowledge_type": "framework",
        "name": "Thermodynamic cycle",
        "purpose": "Describe how a heat engine converts heat to work.",
        "components": [
            {"name": "Hot reservoir", "description": "Supplies heat"},
            {"name": "Working fluid"},
            {"name": "Cold reservoir", "description": "Receives waste heat"},
        ],
        "component_relationships": [
            {"source": "Hot reservoir", "target": "Working fluid", "relation": "heats"},
            {"source": "Working fluid", "target": "Cold reservoir", "relation": "rejects heat to"},
        ],
        "application": "Designing engines and refrigerators.",
        "limitations": ["ideal cycles only"],
    } | overrides


def evidence_data(**overrides: Any) -> dict[str, Any]:
    return {
        "knowledge_type": "evidence",
        "claim": "No engine is perfectly efficient.",
        "description": "Measurements of 40 engines over ten years.",
        "findings": ["Best efficiency was 42%"],
        "supporting_examples": ["The diesel engine"],
        "context": "Laboratory study, 1998",
    } | overrides


def example_data(**overrides: Any) -> dict[str, Any]:
    return {
        "knowledge_type": "example",
        "name": "Melting ice",
        "scenario": "An ice cube melts in a warm room.",
        "explanation": "Heat flows in and entropy rises.",
        "related_to": [{"knowledge_type": "principle", "name": "Second law"}],
        "expected_outcome": "The ice becomes water.",
    } | overrides


def equation_data(**overrides: Any) -> dict[str, Any]:
    return {
        "knowledge_type": "equation",
        "name": "Entropy change",
        "original_expression": "dS = dQ / T",
        "latex": r"dS = \frac{dQ}{T}",
        "variables": [
            {"symbol": "S", "definition": "entropy", "unit": "J/K"},
            {"symbol": "Q", "definition": "heat"},
        ],
        "parameters": [{"symbol": "T", "kind": "parameter", "description": "absolute temperature"}],
        "assumptions": ["reversible process"],
        "explanation": "Entropy change equals heat over temperature.",
        "extraction_status": "complete",
        "reconstruction_status": "not_needed",
    } | overrides


BUILDERS = {
    KnowledgeType.CONCEPT: concept_data,
    KnowledgeType.PRINCIPLE: principle_data,
    KnowledgeType.METHOD: method_data,
    KnowledgeType.FRAMEWORK: framework_data,
    KnowledgeType.EVIDENCE: evidence_data,
    KnowledgeType.EXAMPLE: example_data,
    KnowledgeType.EQUATION: equation_data,
}


def unit_data(content: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Arguments for KnowledgeUnitCreate with one source passage and no database ties."""
    return {
        "book_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "content": content,
        "sources": [{"passage_id": uuid.uuid4()}],
    } | overrides


@dataclass
class StoredBook:
    """A book with one document, a chapter with a section, and five passages."""

    book: Any
    document: Any
    chapter: StructuralNode
    section: StructuralNode
    passages: list[Passage]

    def unit(self, content: dict[str, Any], passages: int = 1, **overrides: Any) -> KnowledgeUnitCreate:
        data = {
            "book_id": self.book.id,
            "document_id": self.document.id,
            "content": content,
            "sources": [{"passage_id": passage.id} for passage in self.passages[:passages]],
            "chapter_node_id": self.chapter.id,
            "section_node_id": self.section.id,
            "page_start": 3,
            "page_end": 4,
            "page_status": "exact",
        } | overrides
        return KnowledgeUnitCreate(**data)


def store_book(session: Session, title: str = "Thermodynamics") -> StoredBook:
    book = make_book(title)
    document = make_document(book)
    session.add_all([book, document])
    session.flush()
    chapter = StructuralNode(
        book_id=book.id, document_id=document.id, node_type="chapter", title="Entropy", level=0, sequence=0,
        page_start=1, page_end=10,
    )
    session.add(chapter)
    session.flush()
    section = StructuralNode(
        book_id=book.id, document_id=document.id, parent_id=chapter.id, node_type="section",
        title="The second law", level=1, sequence=1, page_start=3, page_end=5,
    )
    session.add(section)
    session.flush()
    passages = [
        Passage(
            book_id=book.id, document_id=document.id, structural_node_id=section.id, text=f"Passage {i} text.",
            sequence=i, document_sequence=i, page_start=3 + i, page_end=3 + i, page_status="exact", token_count=10,
        )
        for i in range(5)
    ]
    session.add_all(passages)
    session.commit()
    return StoredBook(book, document, chapter, section, passages)


def store_equation(session: Session, stored: StoredBook) -> Equation:
    equation = Equation(
        document_id=stored.document.id, sequence_number=0, identity_hash=uuid.uuid4().hex, latex=r"dS = dQ/T",
        extraction_method="test",
    )
    session.add(equation)
    session.commit()
    return equation
