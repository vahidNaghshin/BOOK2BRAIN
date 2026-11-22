"""Equation preservation: what the source gave, kept apart from the model's reading, with its status and context."""

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.knowledge.content import EquationContent
from app.knowledge.schemas import KnowledgeRelationshipCreate
from app.knowledge.types import EquationCaptureStatus, ExpressionStatus, KnowledgeOrigin, ReviewFlagCode
from app.services.equation_service import EquationService, classify_capture, flag_code_for
from app.services.knowledge_service import KnowledgeService
from app.services.relationship_service import RelationshipService
from app.services.review_flag_service import ReviewFlagService
from database.models import Equation, EquationReconstruction, EquationSource, KnowledgeUnit
from tests.extraction_db_helpers import Library, add_equation, item, make_service, one_item, reply, store_library
from tests.knowledge_helpers import equation_data

C = EquationCaptureStatus
TEXT = "Before the equation. dS = dQ / T. After the equation."


@pytest.fixture
def library(session: Session) -> Library:
    lib = store_library(session)
    lib.passages["ss1a"].text = TEXT
    session.commit()
    return lib


def extract(session, library, content=None, scope="ss1", **options):
    """Extract equations from the subsection (one passage) with a model that returns one equation reading."""

    def answer(call):
        return reply(item(call, content=content or {})) if content is not None else one_item(call)

    service, _ = make_service(session, answer, **options)
    status = service.extract_section(library.nodes[scope].id, ["equation"], **{k: v for k, v in options.items() if k == "force"})
    return service, status


def capture_of(session, unit_id):
    return session.get(EquationSource, unit_id)


def equation_units(session):
    return list(session.scalars(select(KnowledgeUnit).where(KnowledgeUnit.knowledge_type == "equation").order_by(KnowledgeUnit.created_at)))


def flag_codes(session, library, unit):
    return {f.code for f in ReviewFlagService(session).list_flags(library.book.id, unit_id=unit.id)}


# ---------------------------------------------------------------- the capture status (pure)


@pytest.mark.parametrize(
    "fields, status",
    [
        (dict(original_text="dS = dQ / T", source_latex=r"\frac{dQ}{T}", image_reference="x.png", model_status=ExpressionStatus.COMPLETE, matched_to_source=True), C.COMPLETE),
        (dict(original_text="dS = dQ / T", source_latex=None, image_reference="x.png", model_status=ExpressionStatus.COMPLETE, matched_to_source=True), C.TEXT_ONLY),
        (dict(original_text=None, source_latex=None, image_reference="x.png", model_status=ExpressionStatus.COMPLETE, matched_to_source=True), C.IMAGE_ONLY),
        (dict(original_text="dS = dQ /", source_latex=None, image_reference=None, model_status=ExpressionStatus.PARTIAL, matched_to_source=True), C.INCOMPLETE),
        (dict(original_text="dS", source_latex="dS", image_reference=None, model_status=ExpressionStatus.FAILED, matched_to_source=True), C.INCOMPLETE),
        (dict(original_text="x", source_latex="x", image_reference=None, model_status=ExpressionStatus.AMBIGUOUS, matched_to_source=True), C.AMBIGUOUS),
        (dict(original_text="x", source_latex="x", image_reference=None, model_status=ExpressionStatus.COMPLETE, matched_to_source=True, source_validation="uncertain"), C.AMBIGUOUS),
        (dict(original_text="x", source_latex="x", image_reference=None, model_status=ExpressionStatus.COMPLETE, matched_to_source=False), C.NEEDS_REVIEW),
        (dict(original_text="x", source_latex="x", image_reference=None, model_status=ExpressionStatus.COMPLETE, matched_to_source=True, source_validation="invalid"), C.NEEDS_REVIEW),
        (dict(original_text=None, source_latex=None, image_reference=None, model_status=ExpressionStatus.COMPLETE, matched_to_source=True), C.NEEDS_REVIEW),
    ],
)
def test_the_capture_status_says_what_was_captured_not_whether_it_is_right(fields, status):
    result, notes = classify_capture(**fields)
    assert result == status and (notes or status == C.COMPLETE)


def test_which_statuses_raise_which_flags():
    assert {s: flag_code_for(s) for s in C} == {
        C.COMPLETE: None, C.TEXT_ONLY: None, C.INCOMPLETE: ReviewFlagCode.INCOMPLETE_EQUATION, C.IMAGE_ONLY: ReviewFlagCode.INCOMPLETE_EQUATION,
        C.AMBIGUOUS: ReviewFlagCode.AMBIGUOUS_EQUATION, C.NEEDS_REVIEW: ReviewFlagCode.EQUATION_NEEDS_REVIEW,
    }


# ---------------------------------------------------------------- the schema


def test_an_equation_with_only_an_image_must_say_so():
    ok = EquationContent(name="Equation 1", extraction_status="failed", status_notes="Only an image of the equation is available.")
    assert ok.original_expression is None and ok.latex is None and ok.is_incomplete and "image" in ok.body
    with pytest.raises(ValidationError, match="needs its original expression"):
        EquationContent(name="Equation 1", extraction_status="failed")  # nothing, and no explanation
    with pytest.raises(ValidationError, match="needs its original expression"):
        EquationContent(name="Equation 1", extraction_status="complete", status_notes="x")  # complete means something was read


# ---------------------------------------------------------------- a complete equation


def test_a_complete_equation_is_preserved_with_its_source_context_and_image(session, library):
    eq = add_equation(session, library, "ss1a")
    extract(session, library, equation_data())
    [unit] = equation_units(session)
    capture = capture_of(session, unit.id)

    assert capture.capture_status == "complete" and capture.capture_notes == []
    assert (capture.equation_id, unit.equation_id) == (eq.id, eq.id)
    assert (capture.original_text, capture.source_latex) == (eq.plain_text, eq.latex)
    assert capture.image_reference == eq.crop_reference and capture.surrounding_text == eq.surrounding_text
    assert (capture.context_before, capture.context_after) == ("Before the equation.", ". After the equation.")
    assert capture.passage_id == library.passages["ss1a"].id and capture.structural_node_id == library.passages["ss1a"].structural_node_id
    assert (capture.page_start, capture.page_end) == (6, 6)
    assert capture.source_location == {"page": eq.page_number}
    assert flag_codes(session, library, unit) == set()
    assert unit.origin == "ai_extracted" and unit.sources[0].reference_type == "equation"


def test_the_surrounding_text_comes_from_the_passage_not_from_the_model(session, library):
    add_equation(session, library, "ss1a")
    extract(session, library, equation_data(explanation="A completely invented explanation."))
    [unit] = equation_units(session)
    capture = capture_of(session, unit.id)
    assert "invented" not in (capture.context_before or "") + (capture.context_after or "") + (capture.surrounding_text or "")
    assert capture.context_before == "Before the equation." and capture.context_after.endswith("After the equation.")


def test_the_original_is_never_replaced_by_what_the_model_wrote(session, library):
    eq = add_equation(session, library, "ss1a")
    extract(session, library, equation_data(original_expression="dS=dQ/T", latex=r"dS = \frac{dQ}{T}"))
    [unit] = equation_units(session)
    assert capture_of(session, unit.id).original_text == eq.plain_text == "dS = dQ / T"
    assert unit.details["original_expression"] == "dS=dQ/T"  # the model's reading is kept, separately
    session.expire_all()
    stored = session.get(Equation, eq.id)
    assert (stored.plain_text, stored.latex, stored.crop_reference) == (eq.plain_text, eq.latex, eq.crop_reference)


# ---------------------------------------------------------------- incomplete, ambiguous, text only, needs review


def test_an_incomplete_equation_is_kept_and_flagged(session, library):
    add_equation(session, library, "ss1a")
    extract(session, library, equation_data(original_expression="dS = dQ /", latex=None, variables=[], parameters=[], assumptions=[], extraction_status="partial", reconstruction_status="not_attempted", status_notes="The denominator is cut off."))
    [unit] = equation_units(session)
    capture = capture_of(session, unit.id)
    assert capture.capture_status == "incomplete" and capture.capture_notes == ["Part of the expression is missing or cut off."]
    assert unit.is_uncertain and unit.validation_status != "valid" and unit.details["status_notes"] == "The denominator is cut off."
    assert flag_codes(session, library, unit) >= {ReviewFlagCode.INCOMPLETE_EQUATION}


def test_an_ambiguous_equation_is_flagged_as_such(session, library):
    add_equation(session, library, "ss1a")
    extract(session, library, equation_data(extraction_status="ambiguous", reconstruction_status="not_attempted", status_notes="Could be T or t."))
    [unit] = equation_units(session)
    assert capture_of(session, unit.id).capture_status == "ambiguous"
    assert ReviewFlagCode.AMBIGUOUS_EQUATION in flag_codes(session, library, unit)


def test_an_equation_without_latex_is_text_only(session, library):
    add_equation(session, library, "ss1a", latex="", original_latex=None)
    extract(session, library, equation_data(latex=None))
    [unit] = equation_units(session)
    capture = capture_of(session, unit.id)
    assert (capture.capture_status, capture.source_latex, capture.original_text) == ("text_only", None, "dS = dQ / T")
    assert flag_codes(session, library, unit) == set()  # a valid state, not a problem


def test_a_reading_that_matches_nothing_in_the_source_needs_review(session, library):
    library.passages["ss1a"].text = "A passage about heat with no equation in it."
    session.commit()
    extract(session, library, equation_data(original_expression="E = mc^2", latex="E = mc^2"))
    [unit] = equation_units(session)
    capture = capture_of(session, unit.id)
    assert capture.capture_status == "needs_review" and capture.original_text is None and capture.equation_id is None
    assert "could not be matched" in capture.capture_notes[0]
    assert ReviewFlagCode.EQUATION_NEEDS_REVIEW in flag_codes(session, library, unit)


def test_an_equation_the_source_extraction_marked_invalid_needs_review(session, library):
    add_equation(session, library, "ss1a", validation_status="invalid")
    extract(session, library, equation_data())
    [unit] = equation_units(session)
    capture = capture_of(session, unit.id)
    assert capture.capture_status == "needs_review" and "marked invalid" in capture.capture_notes[0]


# ---------------------------------------------------------------- image only and unread equations


def test_an_equation_the_model_could_not_read_is_kept_as_a_source_derived_unit(session, library):
    eq = add_equation(session, library, "ss1a", latex="", original_latex=None, plain_text=None, validation_status="invalid")
    service, status = extract(session, library, None)  # the model returns an equation item for the passage too
    # the stored equation has no text at all, so the model's item cannot be matched to it: both are kept
    units = equation_units(session)
    derived = [u for u in units if u.origin == "source_derived"]
    assert len(derived) == 1 and status.run.units_extracted == len(units)
    [unit] = derived
    capture = capture_of(session, unit.id)
    assert capture.capture_status == "image_only" and capture.image_reference == eq.crop_reference and capture.equation_id == eq.id
    assert unit.equation_id == eq.id and "Only an image" in unit.content and unit.details["extraction_status"] == "failed"
    assert unit.is_uncertain and unit.validation_status != "valid"
    assert flag_codes(session, library, unit) >= {ReviewFlagCode.INCOMPLETE_EQUATION}
    assert unit.sources[0].reference_type == "equation" and unit.extraction_execution_id is None
    assert unit.unit_metadata["source_derived"]


def test_image_only_equations_are_kept_even_when_the_model_finds_nothing(session, library):
    eq = add_equation(session, library, "ss1a", latex="", original_latex=None, plain_text=None)
    service, _ = make_service(session, lambda call: '{"items": []}')
    service.extract_section(library.nodes["ss1"].id, ["equation"])
    [unit] = equation_units(session)
    assert unit.origin == "source_derived" and capture_of(session, unit.id).capture_status == "image_only"
    assert EquationService(session).get_capture(unit.id).image_reference == eq.crop_reference


def test_text_that_phase_1_captured_without_latex_is_kept_when_the_model_finds_nothing(session, library):
    add_equation(session, library, "ss1a", latex="", original_latex=None)
    service, _ = make_service(session, lambda call: '{"items": []}')
    service.extract_section(library.nodes["ss1"].id, ["equation"])
    [unit] = equation_units(session)
    assert unit.details["original_expression"] == "dS = dQ / T" and capture_of(session, unit.id).capture_status == "text_only"
    assert unit.details["extraction_status"] == "complete"  # Phase 1 marked it valid


def test_several_equations_in_one_passage_are_matched_and_none_is_dropped(session, library):
    first = add_equation(session, library, "ss1a")
    second = add_equation(session, library, "ss1a", n=1, latex=r"F = ma", original_latex=r"F = m a", plain_text="F = ma")
    extract(session, library, equation_data(name="Newton", original_expression="F = ma", latex="F = ma"))
    units = equation_units(session)
    by_equation = {u.equation_id: u for u in units}
    assert set(by_equation) == {first.id, second.id} and len(units) == 2
    assert by_equation[second.id].origin == "ai_extracted" and by_equation[second.id].title == "Newton"
    assert by_equation[first.id].origin == "source_derived"  # the one the model did not read


def test_an_ambiguous_match_among_several_equations_is_reported_not_guessed(session, library):
    add_equation(session, library, "ss1a")
    add_equation(session, library, "ss1a", n=1, latex="x", original_latex="x", plain_text="x")
    library.passages["ss1a"].text = "Nothing recognisable here."
    session.commit()
    extract(session, library, equation_data(original_expression="E = mc^2", latex="E = mc^2"))
    units = equation_units(session)
    model_unit = next(u for u in units if u.origin == "ai_extracted")
    capture = capture_of(session, model_unit.id)
    assert capture.capture_status == "needs_review" and capture.equation_id is None
    assert any("holds 2 stored equations" in n for n in capture.capture_notes)
    assert sum(u.origin == "source_derived" for u in units) == 2  # both stored equations are kept


# ---------------------------------------------------------------- idempotency, association, queries


def test_repeating_the_extraction_creates_no_second_equation_unit(session, library):
    add_equation(session, library, "ss1a")
    add_equation(session, library, "ss1a", n=1, latex="x", original_latex="x", plain_text="x")
    service, _ = make_service(session, lambda call: reply(item(call, content=equation_data())))
    service.extract_section(library.nodes["ss1"].id, ["equation"])
    count = len(equation_units(session))
    service.extract_section(library.nodes["ss1"].id, ["equation"], force=True)
    assert len(equation_units(session)) == count and session.scalars(select(EquationSource)).all().__len__() == count


def test_an_equation_stays_associated_with_the_knowledge_it_describes(session, library):
    add_equation(session, library, "ss1a")
    service, _ = make_service(session, one_item)
    service.extract_section(library.nodes["ss1"].id, ["concept", "equation"])
    units = {u.knowledge_type: u for u in session.scalars(select(KnowledgeUnit))}
    RelationshipService(session).create(
        KnowledgeRelationshipCreate(
            book_id=library.book.id, source_unit_id=units["equation"].id, target_unit_id=units["concept"].id, relationship_type="describes",
            explanation="The equation expresses entropy change.", source_passage_ids=[library.passages["ss1a"].id],
        )
    )
    detail = KnowledgeService(session).detail(units["equation"].id)
    [view] = detail.relationships.outgoing
    assert (view.relationship.relationship_type.value, view.other_title) == ("describes", "Entropy")
    assert detail.equation.capture_status == EquationCaptureStatus.COMPLETE and detail.equation.equation_id == units["equation"].equation_id
    assert KnowledgeService(session).detail(units["concept"].id).equation is None


def test_equations_can_be_listed_by_status(session, library):
    add_equation(session, library, "ss1a")
    add_equation(session, library, "ss1a", n=1, latex="", original_latex=None, plain_text=None)
    extract(session, library, equation_data())
    service = EquationService(session)
    assert service.counts_by_status(library.book.id) == {"complete": 1, "image_only": 1}
    assert [c.capture_status for c in service.list_captures(library.book.id, "image_only")] == [C.IMAGE_ONLY]
    assert len(service.list_captures(library.book.id)) == 2 and service.get_capture(__import__("uuid").uuid4()) is None


def test_phase_1_equations_and_passages_are_not_changed(session, library):
    eq = add_equation(session, library, "ss1a")
    before = (eq.latex, eq.original_latex, eq.plain_text, eq.crop_reference, eq.surrounding_text, eq.validation_status, library.passages["ss1a"].text)
    extract(session, library, equation_data())
    session.expire_all()
    stored = session.get(Equation, eq.id)
    assert (stored.latex, stored.original_latex, stored.plain_text, stored.crop_reference, stored.surrounding_text, stored.validation_status, session.get(type(library.passages["ss1a"]), library.passages["ss1a"].id).text) == before


# ---------------------------------------------------------------- room for a later reconstruction


def test_a_future_reconstruction_has_its_own_record_apart_from_the_original(session, library):
    add_equation(session, library, "ss1a")
    extract(session, library, equation_data())
    [unit] = equation_units(session)
    original = (capture_of(session, unit.id).original_text, dict(unit.details))
    reconstruction = EquationReconstruction(
        knowledge_unit_id=unit.id, reconstructed_expression="dS = dQ_rev / T", reconstructed_latex=r"dS = \frac{\delta Q_{rev}}{T}", confidence=0.6,
        model_configuration_id="m1", provider="ollama", model_id="x:1b", model_version="1", prompt_name="equation_reconstruction",
        prompt_version="0.1", generation_parameters={"temperature": 0}, math_validation_status="inconclusive",
    )
    session.add(reconstruction)
    session.add(EquationReconstruction(knowledge_unit_id=unit.id, reconstructed_latex="x = y"))  # several models can reconstruct one equation
    session.commit()
    session.expire_all()
    assert (capture_of(session, unit.id).original_text, dict(session.get(KnowledgeUnit, unit.id).details)) == original
    stored = session.scalars(select(EquationReconstruction).order_by(EquationReconstruction.created_at)).all()
    assert [r.review_status for r in stored] == ["unreviewed", "unreviewed"] and stored[0].math_validation_status == "inconclusive"
    assert stored[1].math_validation_status == "not_validated"


@pytest.mark.parametrize("fields", [{}, {"reconstructed_latex": "x", "confidence": 1.5}, {"reconstructed_latex": "x", "math_validation_status": "proven"}, {"reconstructed_latex": "x", "review_status": "maybe"}])
def test_a_reconstruction_record_is_validated(session, library, fields):
    add_equation(session, library, "ss1a")
    extract(session, library, equation_data())
    [unit] = equation_units(session)
    session.add(EquationReconstruction(knowledge_unit_id=unit.id, **fields))
    with pytest.raises(IntegrityError):
        session.commit()


def test_every_unit_origin_is_distinguishable():
    assert {o.value for o in KnowledgeOrigin} == {"ai_extracted", "ai_interpretation", "ai_reconstructed", "source_derived", "human"}
