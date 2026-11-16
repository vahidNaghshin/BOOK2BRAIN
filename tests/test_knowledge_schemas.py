"""Phase 2 knowledge schemas: validation of each category and of the common envelope. No database."""

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from app.knowledge.content import (
    CONTENT_MODELS,
    AnyKnowledgeContent,
    ConceptContent,
    EquationContent,
    EvidenceContent,
    ExampleContent,
    FrameworkContent,
    MethodContent,
    PrincipleContent,
)
from app.knowledge.relationships import RELATIONSHIP_RULES, check_relationship_types, is_allowed
from app.knowledge.schemas import (
    ExtractionRunCreate,
    ExtractionRunRead,
    KnowledgeRelationshipCreate,
    KnowledgeUnitCreate,
    Uncertainty,
)
from app.knowledge.types import (
    ExtractionRunStatus,
    KnowledgeType,
    PageStatus,
    RelationshipType,
    ReviewStatus,
    SourceRole,
    ValidationStatus,
)
from database.models import (
    EXTRACTION_RUN_STATUSES,
    KNOWLEDGE_TYPES,
    KNOWLEDGE_VALIDATION_STATUSES,
    PAGE_STATUSES,
    RELATIONSHIP_TYPES,
    REVIEW_STATUSES,
    SOURCE_ROLES,
)
from tests.knowledge_helpers import (
    BUILDERS,
    concept_data,
    equation_data,
    evidence_data,
    example_data,
    framework_data,
    method_data,
    principle_data,
    unit_data,
)

CONTENT_ADAPTER = TypeAdapter(AnyKnowledgeContent)

CLASSES = {
    KnowledgeType.CONCEPT: ConceptContent,
    KnowledgeType.PRINCIPLE: PrincipleContent,
    KnowledgeType.METHOD: MethodContent,
    KnowledgeType.FRAMEWORK: FrameworkContent,
    KnowledgeType.EVIDENCE: EvidenceContent,
    KnowledgeType.EXAMPLE: ExampleContent,
    KnowledgeType.EQUATION: EquationContent,
}


def errors_of(exc: pytest.ExceptionInfo[ValidationError]) -> str:
    return " | ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.value.errors())


# -- every category ---------------------------------------------------------------------------------


@pytest.mark.parametrize("knowledge_type", list(KnowledgeType))
def test_each_category_is_created_from_valid_data(knowledge_type: KnowledgeType) -> None:
    content = CONTENT_ADAPTER.validate_python(BUILDERS[knowledge_type]())

    assert isinstance(content, CLASSES[knowledge_type])
    assert content.knowledge_type == knowledge_type
    assert content.title and content.body


@pytest.mark.parametrize("knowledge_type", list(KnowledgeType))
def test_each_category_is_wrapped_in_a_knowledge_unit(knowledge_type: KnowledgeType) -> None:
    unit = KnowledgeUnitCreate(**unit_data(BUILDERS[knowledge_type]()))

    assert unit.knowledge_type == knowledge_type
    assert isinstance(unit.content, CLASSES[knowledge_type])


def test_content_models_cover_exactly_the_knowledge_types() -> None:
    assert set(CONTENT_MODELS) == set(KnowledgeType)
    assert {t.value for t in KnowledgeType} == set(KNOWLEDGE_TYPES)


@pytest.mark.parametrize("knowledge_type", list(KnowledgeType))
def test_content_survives_a_json_round_trip(knowledge_type: KnowledgeType) -> None:
    original = CONTENT_ADAPTER.validate_python(BUILDERS[knowledge_type]())

    again = CONTENT_ADAPTER.validate_json(original.model_dump_json())

    assert again == original


def test_an_unknown_or_missing_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CONTENT_ADAPTER.validate_python(concept_data(knowledge_type="theorem"))
    data = concept_data()
    del data["knowledge_type"]
    with pytest.raises(ValidationError):
        CONTENT_ADAPTER.validate_python(data)


def test_a_field_of_another_category_is_rejected() -> None:
    with pytest.raises(ValidationError, match="steps"):
        CONTENT_ADAPTER.validate_python(concept_data(steps=["one"]))


# -- required and optional fields -------------------------------------------------------------------

REQUIRED = {
    KnowledgeType.CONCEPT: ["name", "definition"],
    KnowledgeType.PRINCIPLE: ["name", "statement"],
    KnowledgeType.METHOD: ["name", "purpose", "steps"],
    KnowledgeType.FRAMEWORK: ["name", "purpose", "components"],
    KnowledgeType.EVIDENCE: ["claim", "description"],
    KnowledgeType.EXAMPLE: ["name", "scenario"],
    KnowledgeType.EQUATION: ["name", "extraction_status"],
}


@pytest.mark.parametrize(
    ("knowledge_type", "field"), [(t, f) for t, fields in REQUIRED.items() for f in fields]
)
def test_required_fields_cannot_be_left_out(knowledge_type: KnowledgeType, field: str) -> None:
    data = BUILDERS[knowledge_type]()
    del data[field]

    with pytest.raises(ValidationError) as exc:
        CONTENT_ADAPTER.validate_python(data)

    assert field in errors_of(exc)


@pytest.mark.parametrize("knowledge_type", [t for t in KnowledgeType if t != KnowledgeType.EQUATION])
def test_only_required_fields_are_enough(knowledge_type: KnowledgeType) -> None:
    required = {"knowledge_type": knowledge_type.value} | {f: BUILDERS[knowledge_type]()[f] for f in REQUIRED[knowledge_type]}

    content = CONTENT_ADAPTER.validate_python(required)

    for name, info in type(content).model_fields.items():
        if name not in REQUIRED[knowledge_type] and name != "knowledge_type":
            assert getattr(content, name) in (None, [])  # optional text is None, optional lists are empty


@pytest.mark.parametrize("knowledge_type", list(KnowledgeType))
def test_blank_required_text_is_rejected(knowledge_type: KnowledgeType) -> None:
    field = REQUIRED[knowledge_type][0]

    with pytest.raises(ValidationError):
        CONTENT_ADAPTER.validate_python(BUILDERS[knowledge_type]() | {field: "   "})


def test_text_is_stripped() -> None:
    assert ConceptContent(name="  Entropy\n", definition=" x ").name == "Entropy"


def test_a_list_item_cannot_be_blank_and_lists_are_not_none() -> None:
    with pytest.raises(ValidationError):
        ConceptContent(name="A", definition="B", characteristics=["fine", " "])
    with pytest.raises(ValidationError):
        ConceptContent(name="A", definition="B", characteristics=None)


def test_name_length_is_limited() -> None:
    with pytest.raises(ValidationError):
        ConceptContent(name="x" * 301, definition="B")


def test_default_lists_are_not_shared_between_instances() -> None:
    first = ConceptContent(name="A", definition="B")
    first.examples.append("leaked?")

    assert ConceptContent(name="A", definition="B").examples == []


# -- category rules ---------------------------------------------------------------------------------


def test_a_method_needs_at_least_one_step() -> None:
    with pytest.raises(ValidationError, match="steps"):
        MethodContent(**method_data(steps=[]))


def test_a_method_keeps_its_steps_in_order() -> None:
    assert MethodContent(**method_data()).steps[0] == "Weigh the sample"


def test_a_framework_needs_components() -> None:
    with pytest.raises(ValidationError, match="components"):
        FrameworkContent(**framework_data(components=[], component_relationships=[]))


def test_framework_component_names_must_be_unique_ignoring_case() -> None:
    with pytest.raises(ValidationError, match="unique"):
        FrameworkContent(**framework_data(components=[{"name": "Fluid"}, {"name": "fluid"}], component_relationships=[]))


def test_framework_relationships_must_refer_to_its_components() -> None:
    bad = [{"source": "Hot reservoir", "target": "Cold storage", "relation": "heats"}]

    with pytest.raises(ValidationError, match="not a component"):
        FrameworkContent(**framework_data(component_relationships=bad))


def test_a_framework_component_cannot_relate_to_itself() -> None:
    bad = [{"source": "Working fluid", "target": "working fluid", "relation": "is"}]

    with pytest.raises(ValidationError, match="itself"):
        FrameworkContent(**framework_data(component_relationships=bad))


def test_component_descriptions_are_optional() -> None:
    content = FrameworkContent(**framework_data())

    assert [c.description for c in content.components] == ["Supplies heat", None, "Receives waste heat"]


def test_an_example_can_relate_to_a_concept_principle_or_method_only() -> None:
    ok = ExampleContent(**example_data(related_to=[{"knowledge_type": "method", "name": "Calorimetry"}]))
    assert ok.related_to[0].knowledge_type == "method"

    with pytest.raises(ValidationError):
        ExampleContent(**example_data(related_to=[{"knowledge_type": "equation", "name": "x"}]))


def test_evidence_title_is_its_claim_shortened() -> None:
    assert EvidenceContent(**evidence_data()).title == "No engine is perfectly efficient."
    long = EvidenceContent(**evidence_data(claim="word " * 200))
    assert len(long.title) <= 300 and long.title.endswith("…")


# -- equations: complete, incomplete, ambiguous -----------------------------------------------------


def test_a_complete_equation_keeps_original_and_latex() -> None:
    content = EquationContent(**equation_data())

    assert content.original_expression == "dS = dQ / T"
    assert content.latex == r"dS = \frac{dQ}{T}"
    assert not content.is_incomplete


def test_an_equation_with_only_its_original_expression_is_allowed() -> None:
    content = EquationContent(
        **equation_data(latex=None, reconstruction_status="not_attempted", variables=[], parameters=[])
    )

    assert content.latex is None and content.body == "dS = dQ / T"


def test_an_equation_with_only_latex_is_allowed() -> None:
    assert EquationContent(**equation_data(original_expression=None)).body == r"dS = \frac{dQ}{T}"


def test_an_equation_without_any_expression_is_rejected() -> None:
    with pytest.raises(ValidationError, match="original expression or a LaTeX"):
        EquationContent(**equation_data(original_expression=None, latex=None))


def test_a_truncated_equation_is_stored_as_read_and_marked_incomplete() -> None:
    content = EquationContent(
        name="Energy",
        original_expression="E = m c",
        extraction_status="partial",
        status_notes="The exponent after 'c' is cut off at the page edge.",
    )

    assert content.original_expression == "E = m c"
    assert content.latex is None
    assert content.is_incomplete
    assert content.reconstruction_status == "not_attempted"


def test_an_ambiguous_equation_can_carry_a_partial_reconstruction() -> None:
    content = EquationContent(
        name="Ratio", original_expression="a / b c", latex=r"\frac{a}{b} c",
        extraction_status="ambiguous", reconstruction_status="partial",
    )

    assert content.is_incomplete


def test_an_equation_variable_can_lack_a_definition() -> None:
    content = EquationContent(**equation_data(variables=[{"symbol": "k"}]))

    assert content.variables[0].definition is None


def test_equation_status_must_be_stated() -> None:
    data = equation_data()
    del data["extraction_status"]
    with pytest.raises(ValidationError, match="extraction_status"):
        EquationContent(**data)


def test_an_incomplete_expression_cannot_be_marked_as_needing_no_reconstruction() -> None:
    with pytest.raises(ValidationError, match="not_needed"):
        EquationContent(**equation_data(extraction_status="partial", reconstruction_status="not_needed"))


@pytest.mark.parametrize("status", ["reconstructed", "partial"])
def test_a_reconstruction_needs_a_latex_form(status: str) -> None:
    with pytest.raises(ValidationError, match="requires a LaTeX"):
        EquationContent(**equation_data(latex=None, extraction_status="partial", reconstruction_status=status))


def test_a_failed_reconstruction_counts_as_incomplete_even_if_extraction_was_complete() -> None:
    assert EquationContent(**equation_data(reconstruction_status="failed")).is_incomplete


def test_a_symbol_cannot_be_defined_twice() -> None:
    with pytest.raises(ValidationError, match="once"):
        EquationContent(**equation_data(parameters=[{"symbol": "s", "kind": "constant"}]))


def test_a_parameter_must_be_a_parameter_or_a_constant() -> None:
    with pytest.raises(ValidationError):
        EquationContent(**equation_data(parameters=[{"symbol": "T", "kind": "variable"}]))


def test_an_incomplete_equation_unit_is_marked_uncertain_automatically() -> None:
    unit = KnowledgeUnitCreate(
        **unit_data(equation_data(extraction_status="partial", reconstruction_status="failed", status_notes="Cut off."))
    )

    assert unit.uncertainty == Uncertainty(is_uncertain=True, notes="Cut off.")


def test_an_incomplete_equation_unit_gets_a_default_uncertainty_note() -> None:
    unit = KnowledgeUnitCreate(**unit_data(equation_data(extraction_status="failed", latex=None, reconstruction_status="failed")))

    assert unit.uncertainty.is_uncertain and "failed" in unit.uncertainty.notes


def test_an_incomplete_equation_cannot_be_validated_as_valid() -> None:
    with pytest.raises(ValidationError, match="incomplete equation"):
        KnowledgeUnitCreate(**unit_data(equation_data(extraction_status="partial", reconstruction_status="failed"), validation_status="valid"))


def test_an_incomplete_equation_may_be_pending_uncertain_or_invalid() -> None:
    for status in ("pending", "uncertain", "invalid"):
        unit = KnowledgeUnitCreate(
            **unit_data(equation_data(extraction_status="ambiguous", reconstruction_status="failed"), validation_status=status)
        )
        assert unit.validation_status == status


def test_a_complete_equation_is_not_marked_uncertain() -> None:
    assert not KnowledgeUnitCreate(**unit_data(equation_data())).uncertainty.is_uncertain


# -- the common envelope ----------------------------------------------------------------------------


def test_defaults_of_a_new_unit() -> None:
    unit = KnowledgeUnitCreate(**unit_data(concept_data()))

    assert isinstance(unit.id, uuid.UUID)
    assert unit.confidence is None
    assert unit.uncertainty == Uncertainty()
    assert unit.validation_status == ValidationStatus.PENDING
    assert unit.review_status == ReviewStatus.UNREVIEWED
    assert unit.page_status == PageStatus.UNKNOWN and unit.page_start is None
    assert unit.metadata == {}
    assert unit.sources[0].role == SourceRole.SUPPORTING


def test_each_unit_gets_its_own_identifier() -> None:
    assert KnowledgeUnitCreate(**unit_data(concept_data())).id != KnowledgeUnitCreate(**unit_data(concept_data())).id


def test_an_identifier_can_be_supplied_and_must_be_a_uuid() -> None:
    fixed = uuid.uuid4()
    assert KnowledgeUnitCreate(**unit_data(concept_data(), id=fixed)).id == fixed
    with pytest.raises(ValidationError):
        KnowledgeUnitCreate(**unit_data(concept_data(), id="not-a-uuid"))


def test_a_unit_needs_book_document_content_and_a_source() -> None:
    for missing in ("book_id", "document_id", "content", "sources"):
        data = unit_data(concept_data())
        del data[missing]
        with pytest.raises(ValidationError, match=missing):
            KnowledgeUnitCreate(**data)


def test_a_unit_needs_at_least_one_source_passage() -> None:
    with pytest.raises(ValidationError, match="sources"):
        KnowledgeUnitCreate(**unit_data(concept_data(), sources=[]))


def test_a_unit_can_have_many_source_passages_with_quotes() -> None:
    ids = [uuid.uuid4() for _ in range(3)]
    unit = KnowledgeUnitCreate(
        **unit_data(
            concept_data(),
            sources=[
                {"passage_id": ids[0], "role": "primary", "quote": "Entropy is a measure"},
                {"passage_id": ids[1]},
                {"passage_id": ids[2], "quote": "  see also  "},
            ],
        )
    )

    assert [s.passage_id for s in unit.sources] == ids
    assert unit.sources[0].role == SourceRole.PRIMARY and unit.sources[2].quote == "see also"


def test_a_passage_cannot_be_a_source_twice() -> None:
    same = uuid.uuid4()
    with pytest.raises(ValidationError, match="only be listed once"):
        KnowledgeUnitCreate(**unit_data(concept_data(), sources=[{"passage_id": same}, {"passage_id": same}]))


def test_only_one_source_can_be_primary() -> None:
    two = [{"passage_id": uuid.uuid4(), "role": "primary"}, {"passage_id": uuid.uuid4(), "role": "primary"}]
    with pytest.raises(ValidationError, match="primary"):
        KnowledgeUnitCreate(**unit_data(concept_data(), sources=two))


def test_a_source_role_must_be_known() -> None:
    with pytest.raises(ValidationError):
        KnowledgeUnitCreate(**unit_data(concept_data(), sources=[{"passage_id": uuid.uuid4(), "role": "main"}]))


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, None])
def test_valid_confidence_values(value: float | None) -> None:
    assert KnowledgeUnitCreate(**unit_data(concept_data(), confidence=value)).confidence == value


@pytest.mark.parametrize("value", [-0.01, 1.01, 7, "high"])
def test_invalid_confidence_values(value: object) -> None:
    with pytest.raises(ValidationError):
        KnowledgeUnitCreate(**unit_data(concept_data(), confidence=value))


def test_uncertainty_notes_need_the_flag() -> None:
    with pytest.raises(ValidationError, match="only allowed"):
        Uncertainty(is_uncertain=False, notes="not sure")
    assert Uncertainty(is_uncertain=True, notes="Two readings possible.").is_uncertain


def test_statuses_must_be_known() -> None:
    with pytest.raises(ValidationError):
        KnowledgeUnitCreate(**unit_data(concept_data(), validation_status="great"))
    with pytest.raises(ValidationError):
        KnowledgeUnitCreate(**unit_data(concept_data(), review_status="done"))


def test_page_range_rules() -> None:
    ok = KnowledgeUnitCreate(**unit_data(concept_data(), page_start=4, page_end=6, page_status="exact"))
    assert (ok.page_start, ok.page_end) == (4, 6)

    for bad in (
        {"page_start": 6, "page_end": 4, "page_status": "exact"},
        {"page_start": 0, "page_status": "exact"},
        {"page_start": 4},  # pages given, status left "unknown"
        {"page_status": "exact"},  # a status without pages
    ):
        with pytest.raises(ValidationError):
            KnowledgeUnitCreate(**unit_data(concept_data(), **bad))


def test_only_an_equation_can_link_to_an_equation_record() -> None:
    assert KnowledgeUnitCreate(**unit_data(equation_data(), equation_id=uuid.uuid4())).equation_id
    with pytest.raises(ValidationError, match="Only an equation"):
        KnowledgeUnitCreate(**unit_data(concept_data(), equation_id=uuid.uuid4()))


def test_metadata_accepts_json_values_only() -> None:
    unit = KnowledgeUnitCreate(**unit_data(concept_data(), metadata={"tags": ["a"], "n": 1, "nested": {"k": None}}))
    assert unit.metadata["tags"] == ["a"]
    with pytest.raises(ValidationError):
        KnowledgeUnitCreate(**unit_data(concept_data(), metadata={"bad": object()}))


def test_unknown_envelope_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="surprise"):
        KnowledgeUnitCreate(**unit_data(concept_data(), surprise=1))


# -- relationships ----------------------------------------------------------------------------------


def relationship_data(**overrides):
    return {
        "book_id": uuid.uuid4(),
        "source_unit_id": uuid.uuid4(),
        "target_unit_id": uuid.uuid4(),
        "relationship_type": "supports",
    } | overrides


def test_a_relationship_is_created_with_defaults() -> None:
    relationship = KnowledgeRelationshipCreate(**relationship_data())

    assert relationship.relationship_type == RelationshipType.SUPPORTS
    assert relationship.explanation is None and relationship.confidence is None and relationship.source_passage_ids == []


def test_a_relationship_can_cite_several_passages_and_explain_itself() -> None:
    ids = [uuid.uuid4(), uuid.uuid4()]
    relationship = KnowledgeRelationshipCreate(
        **relationship_data(explanation=" The study backs the claim. ", confidence=0.8, source_passage_ids=ids)
    )

    assert relationship.explanation == "The study backs the claim." and relationship.source_passage_ids == ids


def test_a_unit_cannot_relate_to_itself() -> None:
    same = uuid.uuid4()
    with pytest.raises(ValidationError, match="itself"):
        KnowledgeRelationshipCreate(**relationship_data(source_unit_id=same, target_unit_id=same))


def test_relationship_values_are_validated() -> None:
    for bad in ({"relationship_type": "loves"}, {"confidence": 2}, {"source_unit_id": "x"}):
        with pytest.raises(ValidationError):
            KnowledgeRelationshipCreate(**relationship_data(**bad))
    passage = uuid.uuid4()
    with pytest.raises(ValidationError, match="only be listed once"):
        KnowledgeRelationshipCreate(**relationship_data(source_passage_ids=[passage, passage]))


@pytest.mark.parametrize(
    ("relationship_type", "source", "target"),
    [
        ("supports", "evidence", "principle"),  # evidence supports a principle
        ("illustrates", "example", "method"),  # example illustrates a method
        ("implements", "method", "framework"),  # method implements a framework
        ("describes", "equation", "concept"),  # equation describes a concept
        ("part_of", "concept", "framework"),
        ("depends_on", "method", "concept"),
        ("related_to", "equation", "example"),
    ],
)
def test_meaningful_relationships_are_allowed(relationship_type: str, source: str, target: str) -> None:
    check_relationship_types(relationship_type, source, target)
    assert is_allowed(relationship_type, source, target)


@pytest.mark.parametrize(
    ("relationship_type", "source", "target"),
    [
        ("supports", "principle", "evidence"),  # the wrong way round
        ("illustrates", "concept", "example"),
        ("implements", "framework", "method"),
        ("describes", "concept", "equation"),
        ("part_of", "framework", "concept"),
        ("supports", "evidence", "equation"),
    ],
)
def test_nonsensical_relationships_are_refused(relationship_type: str, source: str, target: str) -> None:
    with pytest.raises(ValueError, match="cannot be"):
        check_relationship_types(relationship_type, source, target)


def test_every_relationship_type_has_a_rule() -> None:
    assert set(RELATIONSHIP_RULES) == set(RelationshipType)


# -- extraction runs --------------------------------------------------------------------------------


def run_read_data(**overrides):
    now = datetime.now(timezone.utc)
    return {
        "book_id": uuid.uuid4(),
        "status": "pending",
        "passages_processed": 0,
        "units_extracted": 0,
        "failure_count": 0,
        "created_at": now,
        "updated_at": now,
    } | overrides


def test_an_extraction_run_can_cover_a_whole_book_or_one_section() -> None:
    whole = ExtractionRunCreate(book_id=uuid.uuid4())
    assert whole.document_id is None and whole.structural_node_id is None and whole.parameters == {}

    section = ExtractionRunCreate(
        book_id=uuid.uuid4(), document_id=uuid.uuid4(), structural_node_id=uuid.uuid4(),
        model_name="any-model-v1", prompt_version="2026-11-a", parameters={"temperature": 0},
    )
    assert section.model_name == "any-model-v1"


def test_a_section_needs_its_document() -> None:
    with pytest.raises(ValidationError, match="document_id"):
        ExtractionRunCreate(book_id=uuid.uuid4(), structural_node_id=uuid.uuid4())


def test_run_counts_cannot_be_negative() -> None:
    for field in ("passages_processed", "units_extracted", "failure_count"):
        with pytest.raises(ValidationError):
            ExtractionRunRead(**run_read_data(**{field: -1}))


def test_a_finished_run_has_a_completion_time_and_a_running_one_does_not() -> None:
    started = datetime(2026, 11, 16, 9, tzinfo=timezone.utc)
    done = datetime(2026, 11, 16, 10, tzinfo=timezone.utc)
    ExtractionRunRead(**run_read_data(status="completed", started_at=started, completed_at=done))
    ExtractionRunRead(**run_read_data(status="processing", started_at=started))

    with pytest.raises(ValidationError, match="completed_at"):
        ExtractionRunRead(**run_read_data(status="completed", started_at=started))
    with pytest.raises(ValidationError, match="completed_at"):
        ExtractionRunRead(**run_read_data(status="processing", started_at=started, completed_at=done))
    with pytest.raises(ValidationError, match="before"):
        ExtractionRunRead(**run_read_data(status="completed", started_at=done, completed_at=started))


def test_a_failed_run_must_say_why() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError, match="error_message"):
        ExtractionRunRead(**run_read_data(status="failed", started_at=now, completed_at=now))
    failed = ExtractionRunRead(
        **run_read_data(status="failed", started_at=now, completed_at=now, error_code="model_unavailable", error_message="Down.")
    )
    assert failed.error_code == "model_unavailable"


# -- vocabularies stay in step with the database ----------------------------------------------------


def test_vocabularies_match_the_database_constraints() -> None:
    assert {v.value for v in RelationshipType} == set(RELATIONSHIP_TYPES)
    assert {v.value for v in ValidationStatus} == set(KNOWLEDGE_VALIDATION_STATUSES)
    assert {v.value for v in ReviewStatus} == set(REVIEW_STATUSES)
    assert {v.value for v in ExtractionRunStatus} == set(EXTRACTION_RUN_STATUSES)
    assert {v.value for v in SourceRole} == set(SOURCE_ROLES)
    assert {v.value for v in PageStatus} == set(PAGE_STATUSES)
