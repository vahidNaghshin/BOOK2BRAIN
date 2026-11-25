"""Quality metrics: compare what an extraction produced with the manually reviewed reference.

Everything here is pure and deterministic: no database, no model. Each metric records its counts and
its definition, so a value can be re-derived and nothing is a bare score. A metric that cannot be
measured is `None`, never zero.

What the metrics are and are not. Matching is lexical (names, key facts, symbols): it measures
whether the expected knowledge is present, in the right place and traceable, and it flags content
that the reviewer said must not appear. It cannot judge whether a paraphrase preserves an author's meaning;
`key_fact_coverage` is a proxy for that and the report says so. Semantic accuracy is finally a
reviewer's judgement, and the report keeps the matched pairs side by side for that purpose.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from app.evaluation.dataset import EvaluationDataset, ReferencePassage, ReferenceUnit
from app.evaluation.observation import Observation, ObservedUnit

NAME_MATCH = 0.5  # share of name words that must agree for a unit to be recognised by its name
FACT_MATCH = 0.6  # share of key facts that must be present for a unit to be recognised by its content
NAME_NEEDS_FACTS = 0.34  # with the right name, at least this share of the key facts must also be present
SAME_PASSAGE_BONUS = 0.25
EQUATION_MATCH = 0.6  # share of an equation's symbols that must be found for it to be recognised
EQUATION_PRESERVED = 0.8  # share needed to call the original expression preserved

FLAGGING_CODES = {"incomplete_equation", "ambiguous_equation", "equation_needs_review", "uncertain_extraction", "low_confidence"}
UNSUPPORTED_CODES = {"unsupported_claim", "source_quote_mismatch"}
UNCLEAR_CAPTURE = {"incomplete", "ambiguous", "needs_review", "image_only", "text_only"}


# -- results ---------------------------------------------------------------------------------------


class Metric(BaseModel):
    key: str
    label: str
    dimension: str
    value: float | None
    numerator: float | None = None
    denominator: float | None = None
    unit: Literal["ratio", "count", "seconds", "tokens", "usd"] = "ratio"
    definition: str
    note: str | None = None


class UnitMatch(BaseModel):
    passage: str
    reference: str
    category: str
    observed_id: str
    observed_title: str
    score: float
    name_score: float
    fact_coverage: float | None
    facts_missing: list[str]
    category_exact: bool
    source_correct: bool
    pages_ok: bool | None
    forbidden_hits: list[str]


class MissedUnit(BaseModel):
    passage: str
    reference: str
    category: str
    names: list[str]
    required: bool
    reason: Literal["not_extracted", "not_attempted"]
    merged_into: str | None = None  # an observed unit that seems to contain it


class ExtraUnit(BaseModel):
    passage: str
    observed_id: str
    category: str
    title: str
    kind: Literal["redundant", "unmatched"]
    counted: bool  # whether it counts against precision
    of_reference: str | None = None


class EquationOutcome(BaseModel):
    passage: str
    reference: str
    covered: bool
    found: bool
    observed_id: str | None = None
    token_coverage: float | None = None
    original_preserved: bool | None = None
    source_preserved: bool | None = None
    image_retained: bool | None = None
    page_retained: bool | None = None
    capture_status: str | None = None
    status_ok: bool | None = None
    flagged: bool | None = None
    flag_ok: bool | None = None
    handled_correctly: bool | None = None


class RelationshipOutcome(BaseModel):
    passage: str
    source: str
    target: str
    expected_types: list[str]
    found: bool
    stored_type: str | None = None
    type_ok: bool | None = None
    required: bool
    covered: bool


class UnexpectedRelationship(BaseModel):
    source: str
    target: str
    type: str
    origin: str
    kind: Literal["not_in_reference", "unjudged"]


class RunInfo(BaseModel):
    run_id: str
    status: str
    model_configuration_id: str | None
    provider: str | None
    model: str | None
    generation: dict
    parameters: dict
    prompt_versions: list[str]
    duration_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None


class Evaluation(BaseModel):
    dataset: str
    dataset_version: str
    dataset_digest: str
    review_status: str
    metrics: list[Metric]
    runs: list[RunInfo]
    matches: list[UnitMatch]
    missed: list[MissedUnit]
    extras: list[ExtraUnit]
    equations: list[EquationOutcome]
    relationships: list[RelationshipOutcome]
    unexpected_relationships: list[UnexpectedRelationship]
    passages_covered: int
    passages_total: int

    def metric(self, key: str) -> Metric:
        return next(m for m in self.metrics if m.key == key)

    def value(self, key: str) -> float | None:
        return self.metric(key).value


# -- text helpers ----------------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lower case, accents and ligatures folded, punctuation and hyphens as spaces."""
    text = unicodedata.normalize("NFKD", text).casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"\w+", text))


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def stems(text: str) -> list[str]:
    """Words with a plural 's' removed, so 'models' and 'EBM' / 'EBMs' compare equal."""
    return [_stem(w) for w in normalize(text).split()]


def words(text: str) -> set[str]:
    return set(stems(text))


def math_form(text: str) -> str:
    """An expression with layout removed and Greek letters spelled out: 'F_{w}(x, y)' and 'Fw(x,y)' agree."""
    out: list[str] = []
    for char in unicodedata.normalize("NFKC", text):
        if char in " \t\r\n{}\\_^$":
            continue
        name = unicodedata.name(char, "")
        if name.startswith("GREEK") and "LETTER" in name:
            out.append(name.rsplit(" ", 1)[-1].lower())
        else:
            out.append(char.casefold())
    return "".join(out)


def name_score(title: str, names: list[str]) -> float:
    title_form, title_words = " ".join(stems(title)), words(title)
    best = 0.0
    for name in names:
        form, name_words = " ".join(stems(name)), words(name)
        if not name_words:
            continue
        # A name found inside the title counts in full only when it is most of the title: the one-word name
        # "Inference" must not recognise "Energy function used only for inference".
        if form and f" {form} " in f" {title_form} " and 2 * len(name_words) >= len(title_words):
            return 1.0
        overlap = len(name_words & title_words) / len(name_words | title_words)
        best = max(best, overlap)
    return best


def covered_facts(text: str, facts: list[list[str]]) -> tuple[int, list[str]]:
    haystack = f" {normalize(text)} "
    missing = [" / ".join(alternatives) for alternatives in facts if not any(f" {normalize(a)} " in haystack or normalize(a) in haystack for a in alternatives)]
    return len(facts) - len(missing), missing


def token_coverage(tokens: list[str], text: str) -> float:
    form = math_form(text)
    return sum(math_form(t) in form for t in tokens) / len(tokens)


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


# -- matching --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    score: float
    name_score: float
    fact_coverage: float | None
    facts_missing: tuple[str, ...]
    obs: ObservedUnit
    passage: ReferencePassage
    ref: ReferenceUnit


def _candidate(obs: ObservedUnit, passage: ReferencePassage, ref: ReferenceUnit) -> _Candidate | None:
    if obs.category not in {c.value for c in ref.categories}:
        return None
    ns = name_score(obs.title, ref.names)
    coverage, missing = (None, [])
    if ref.facts:
        found, missing = covered_facts(obs.text, ref.facts)
        coverage = found / len(ref.facts)
    # A name alone is not enough when the reference lists facts: "Training energy-based models" must not
    # recognise a unit about "energy-based models" that says nothing of training.
    recognised_by_name = ns >= NAME_MATCH and (coverage is None or coverage >= NAME_NEEDS_FACTS)
    recognised_by_facts = coverage is not None and len(ref.facts) >= 2 and coverage >= FACT_MATCH
    if not (recognised_by_name or recognised_by_facts):
        return None
    score = ns if coverage is None else 0.5 * ns + 0.5 * coverage
    if obs.primary_passage == passage.key:
        score += SAME_PASSAGE_BONUS
    return _Candidate(score, ns, coverage, tuple(missing), obs, passage, ref)


def _assign(candidates: list[_Candidate]) -> tuple[dict[str, _Candidate], dict[str, str]]:
    """One observed unit per reference unit, best score first. Returns {ref key: candidate} and {obs id: ref key}
    for observed units that are recognised but redundant (the reference unit already has a better match)."""
    taken_obs: set[str] = set()
    by_ref: dict[tuple[str, str], _Candidate] = {}
    for cand in sorted(candidates, key=lambda c: (-c.score, c.obs.id)):
        ref_id = (cand.passage.key, cand.ref.key)
        if ref_id in by_ref or cand.obs.id in taken_obs:
            continue
        by_ref[ref_id] = cand
        taken_obs.add(cand.obs.id)
    redundant: dict[str, str] = {}
    for cand in sorted(candidates, key=lambda c: (-c.score, c.obs.id)):
        if cand.obs.id in taken_obs or cand.obs.id in redundant:
            continue
        ref_id = (cand.passage.key, cand.ref.key)
        if ref_id in by_ref and cand.obs.primary_passage == cand.passage.key:
            redundant[cand.obs.id] = f"{ref_id[0]}/{ref_id[1]}"
    return {f"{p}/{r}": c for (p, r), c in by_ref.items()}, redundant


def _is_covered(observation: Observation, passage: ReferencePassage, categories) -> bool:
    if not observation.executions:
        return True
    return any((passage.key, c.value) in observation.covered for c in categories)


# -- the evaluation --------------------------------------------------------------------------------


def evaluate(dataset: EvaluationDataset, observation: Observation) -> Evaluation:
    metrics: list[Metric] = []

    def add(key, label, dimension, value, definition, *, numerator=None, denominator=None, unit="ratio", note=None):
        metrics.append(
            Metric(key=key, label=label, dimension=dimension, value=value, numerator=numerator, denominator=denominator,
                   unit=unit, definition=definition, note=note)
        )

    def add_ratio(key, label, dimension, numerator, denominator, definition, note=None):
        add(key, label, dimension, ratio(numerator, denominator), definition, numerator=numerator, denominator=denominator, note=note)

    units_by_id = {u.id: u for u in observation.units}
    plain_observed = [u for u in observation.units if u.category != "equation"]
    plain_refs = [(p, r) for p, r in dataset.reference_units if r.category.value != "equation"]

    # 1. match observed units to reference units (equations are matched separately, by their symbols)
    candidates = [c for obs in plain_observed for p, r in plain_refs if (c := _candidate(obs, p, r))]
    matched, redundant = _assign(candidates)
    matched_obs = {c.obs.id: key for key, c in matched.items()}

    matches: list[UnitMatch] = []
    for key, cand in matched.items():
        forbidden = [f for f in cand.ref.forbidden if normalize(f) in normalize(cand.obs.text)]
        pages_ok = None
        if cand.obs.page_start is not None:
            low, high = cand.passage.pages
            last = cand.obs.page_end if cand.obs.page_end is not None else cand.obs.page_start
            pages_ok = low <= cand.obs.page_start and last <= high
        matches.append(
            UnitMatch(
                passage=cand.passage.key, reference=cand.ref.key, category=cand.obs.category, observed_id=cand.obs.id,
                observed_title=cand.obs.title, score=round(cand.score, 3), name_score=round(cand.name_score, 3),
                fact_coverage=None if cand.fact_coverage is None else round(cand.fact_coverage, 3),
                facts_missing=list(cand.facts_missing), category_exact=cand.obs.category == cand.ref.category.value,
                source_correct=cand.obs.primary_passage == cand.passage.key, pages_ok=pages_ok, forbidden_hits=forbidden,
            )
        )

    # 2. misses
    missed: list[MissedUnit] = []
    referenced_required = 0
    found_required = 0
    per_category: dict[str, list[int]] = {}
    multi_total = multi_found = 0
    possible_merges = 0
    for passage, ref in plain_refs:
        key = f"{passage.key}/{ref.key}"
        covered = _is_covered(observation, passage, [ref.category])  # asked for under its own category
        if key in matched:
            if ref.required:
                referenced_required += 1
                found_required += 1
                tally = per_category.setdefault(ref.category.value, [0, 0])
                tally[0] += 1
                tally[1] += 1
            continue
        if not covered:
            missed.append(MissedUnit(passage=passage.key, reference=ref.key, category=ref.category.value, names=ref.names,
                                     required=ref.required, reason="not_attempted"))
            continue
        merged_into = None
        for other_key, cand in matched.items():
            if cand.passage.key == passage.key and all(normalize(n) in normalize(cand.obs.text) for n in ref.names[:1]):
                merged_into = cand.obs.title
                break
        if merged_into and ref.required:
            possible_merges += 1
        missed.append(MissedUnit(passage=passage.key, reference=ref.key, category=ref.category.value, names=ref.names,
                                 required=ref.required, reason="not_extracted", merged_into=merged_into))
        if ref.required:
            referenced_required += 1
            per_category.setdefault(ref.category.value, [0, 0])[1] += 1
    for passage in dataset.passages:
        wanted = [u for u in passage.units if u.required and u.category.value != "equation" and _is_covered(observation, passage, [u.category])]
        if len(wanted) >= 2:
            multi_total += len(wanted)
            multi_found += sum(f"{passage.key}/{u.key}" in matched for u in wanted)

    # 3. extras: observed units no reference unit explains
    passage_by_key = {p.key: p for p in dataset.passages}
    extras: list[ExtraUnit] = []
    for obs in plain_observed:
        if obs.id in matched_obs:
            continue
        passage = passage_by_key.get(obs.primary_passage or "")
        judged = bool(passage and (passage.exhaustive or passage.expect_nothing))
        if obs.id in redundant:
            extras.append(ExtraUnit(passage=obs.primary_passage or "", observed_id=obs.id, category=obs.category, title=obs.title,
                                    kind="redundant", counted=True, of_reference=redundant[obs.id]))
        else:
            extras.append(ExtraUnit(passage=obs.primary_passage or "", observed_id=obs.id, category=obs.category, title=obs.title,
                                    kind="unmatched", counted=judged))

    true_positive = len(matched)
    counted_extras = sum(e.counted for e in extras)
    unjudged_extras = sum(not e.counted for e in extras)

    # -- A. completeness
    add_ratio("unit_recall", "Required units found (recall)", "A. Completeness", found_required, referenced_required,
              "Required reference units (not equations) that an observed unit matches, among those whose own category was extracted on their passage (a unit found under an accepted alternative category still counts).")
    for category, (found, total) in sorted(per_category.items()):
        add_ratio(f"unit_recall.{category}", f"Recall: {category}", "A. Completeness", found, total, f"Required {category} units found.")
    add_ratio("multi_unit_recall", "Recall in passages with several required units", "A. Completeness", multi_found, multi_total,
              "Required units found, counting only passages whose reference lists two or more.")
    add_ratio("unit_precision", "Extracted units explained by the reference (precision)", "A. Completeness", true_positive,
              true_positive + counted_extras,
              "Units matching a reference unit (required or optional) divided by those plus redundant units and unmatched units in passages whose reference is exhaustive.",
              note=f"{unjudged_extras} unmatched unit(s) in non-exhaustive passages are not counted either way: a reviewer must judge them." if unjudged_extras else None)

    empty = [p for p in dataset.passages if p.expect_nothing and _is_covered(observation, p, list(dataset.categories))]
    extra_in = {e.passage for e in extras}
    add_ratio("empty_passage_accuracy", "Passages without knowledge left without unexpected units", "A. Completeness",
              sum(p.key not in extra_in for p in empty), len(empty),
              "Passages the reviewer marked as holding no required knowledge in which every extracted unit is one the reference accepts (usually: none).")
    add("possible_merges", "Required units that seem merged into another unit", "A. Completeness", float(possible_merges),
        "Missed required units whose first name appears in the text of a unit matched to a different reference unit of the same passage.",
        numerator=float(possible_merges), unit="count", note="Candidates for a reviewer: similar but distinct units may have been merged.")

    # -- B. semantic accuracy (lexical proxies)
    total_facts = found_facts = 0
    for match in matches:
        ref = next(u for u in passage_by_key[match.passage].units if u.key == match.reference)
        if ref.facts:
            total_facts += len(ref.facts)
            found_facts += len(ref.facts) - len(match.facts_missing)
    add_ratio("key_fact_coverage", "Key facts present in matched units (lexical proxy)", "B. Semantic accuracy", found_facts, total_facts,
              "Reference facts (each satisfied by any of its alternative words) found in the text of the matched unit. Presence of words, not proof of correct meaning.",
              note="A reviewer must still read the matched pairs; this does not show that meaning was preserved.")
    hits = sum(bool(m.forbidden_hits) for m in matches)
    add("forbidden_phrase_units", "Matched units containing a phrase the reviewer ruled out", "B. Semantic accuracy", float(hits),
        "Matched units whose text contains a forbidden phrase of their reference unit.", numerator=float(hits), unit="count")
    flagged_unsupported = sum(bool(set(u.flags) & UNSUPPORTED_CODES) for u in observation.units)
    add("unsupported_flagged", "Units the pipeline flagged as unsupported", "B. Semantic accuracy", float(flagged_unsupported),
        "Units with an open unsupported-claim or quote-mismatch flag.", numerator=float(flagged_unsupported), unit="count")

    # relationships

    # -- C. source accuracy
    correct_source = sum(m.source_correct for m in matches)
    add_ratio("primary_source_accuracy", "Matched units citing the right passage", "C. Source accuracy", correct_source, len(matches),
              "Matched units whose primary source is the reference passage.")
    paged = [m for m in matches if m.pages_ok is not None]
    add_ratio("page_accuracy", "Matched units with pages inside the passage's pages", "C. Source accuracy", sum(bool(m.pages_ok) for m in paged), len(paged),
              "Matched units with page information whose range lies within the reference pages of the passage.")
    in_scope = observation.units
    add_ratio("provenance_verified", "Units whose sources verified", "C. Source accuracy", sum(u.source_status == "verified" for u in in_scope), len(in_scope),
              "Units with source_status 'verified' (passage exists, quotes found in the passage, pages inside it).")
    traced = [u for u in in_scope if u.extracted_from is not None]
    add_ratio("target_source_rate", "Units whose primary source is the passage that was extracted", "C. Source accuracy",
              sum(u.primary_passage == u.extracted_from for u in traced), len(traced),
              "Units whose primary source is the passage being extracted (the prompt says to extract only from it). A unit that cites a neighbouring passage as its primary source was read from context.")
    quotes = sum(u.quotes for u in in_scope)
    add_ratio("quote_verification", "Quotes found in their passage", "C. Source accuracy", sum(u.quotes_verified for u in in_scope), quotes,
              "Source quotes whose words were found in the cited passage text.")

    # -- D. structure
    executions = observation.executions
    answered = [a for a in observation.attempts if a.category != "relationship" and a.status == "completed"]
    add_ratio("schema_valid_calls", "Answers that matched their schema", "D. Structural validity",
              sum(a.validation_outcome != "schema_invalid" for a in answered), len(answered),
              "Model answers (not relationship calls; failed provider calls have no answer) that parsed and matched the Pydantic schema, counting every retry.")
    add_ratio("first_attempt_success", "Extractions that succeeded on the first call", "D. Structural validity",
              sum(e.attempts == 1 and e.status != "failed" for e in executions), len(executions),
              "Executions (passage and category) finished with a single call and not failed.")
    add_ratio("category_accuracy", "Matched units in the expected category", "D. Structural validity", sum(m.category_exact for m in matches), len(matches),
              "Matched units whose category is the reference's primary category (not only an accepted alternative).")

    # -- E. duplication and consistency
    redundant_count = sum(e.kind == "redundant" for e in extras)
    total_observed = len(observation.units)
    add_ratio("duplicate_rate", "Redundant units among stored units", "E. Duplication", redundant_count, total_observed,
              "Stored units that match a reference unit already matched by a better unit of the same passage, over all stored units in the dataset.")
    suppressed = sum(r.duplicates for r in observation.runs)
    add("duplicates_suppressed", "Duplicates the pipeline refused to store", "E. Duplication", float(suppressed),
        "Duplicate units skipped by the pipeline during the evaluated runs.", numerator=float(suppressed), unit="count")

    # -- F. equations
    eq_outcomes = _equations(dataset, observation, passage_by_key)
    equation_units = {f"{e.passage}/{e.reference}": e.observed_id for e in eq_outcomes if e.observed_id}
    rel_outcomes, unexpected, rel_metrics = _relationships(dataset, observation, matched, equation_units, units_by_id)
    for item in rel_metrics:
        metrics.append(item)
    scored = [e for e in eq_outcomes if e.covered or e.found]
    add_ratio("equation_found", "Reference equations found", "F. Equations", sum(e.found for e in scored), len(scored),
              "Reference equations for which an equation unit of the same passage holds at least 60% of the expected symbols.")
    found_eq = [e for e in scored if e.found]
    add_ratio("equation_original_preserved", "Original expression preserved", "F. Equations", sum(bool(e.original_preserved) for e in found_eq), len(found_eq),
              "Found equations whose original_expression holds at least 80% of the expected symbols.")
    with_source = [e for e in found_eq if e.source_preserved is not None]
    add_ratio("equation_source_preserved", "Phase 1 source text preserved", "F. Equations", sum(bool(e.source_preserved) for e in with_source), len(with_source),
              "Found equations whose stored source text or LaTeX is the Phase 1 one (never the model's).")
    with_image = [e for e in found_eq if e.image_retained is not None]
    add_ratio("equation_image_retained", "Image reference retained", "F. Equations", sum(bool(e.image_retained) for e in with_image), len(with_image),
              "Found equations that should have an image whose crop reference is kept.")
    with_page = [e for e in found_eq if e.page_retained is not None]
    add_ratio("equation_page_retained", "Page retained", "F. Equations", sum(bool(e.page_retained) for e in with_page), len(with_page),
              "Found equations whose source page lies within the reference pages.")
    add_ratio("equation_status_correct", "Capture status as expected", "F. Equations", sum(bool(e.status_ok) for e in found_eq), len(found_eq),
              "Found equations whose stored capture status is one the reviewer accepts.")
    add_ratio("equation_handled", "Equations captured or appropriately flagged", "F. Equations", sum(bool(e.handled_correctly) for e in scored), len(scored),
              "Reference equations that were found with an accepted status, and flagged when the reviewer said the source is incomplete or ambiguous. Equations that are missing count against.")

    # -- operations
    _operations(observation, metrics)

    covered_passages = (
        len({key for key, _ in observation.covered} & {p.key for p in dataset.passages}) if observation.executions else len(dataset.passages)
    )
    return Evaluation(
        dataset=dataset.name, dataset_version=dataset.version, dataset_digest=dataset.digest, review_status=dataset.review.status,
        metrics=metrics,
        runs=[RunInfo(run_id=r.run_id, status=r.status, model_configuration_id=r.model_configuration_id, provider=r.provider, model=r.model,
                      generation=r.generation, parameters=r.parameters, prompt_versions=r.prompt_versions, duration_seconds=r.duration_seconds,
                      input_tokens=r.input_tokens, output_tokens=r.output_tokens, cost_usd=r.cost_usd) for r in observation.runs],
        matches=sorted(matches, key=lambda m: (m.passage, m.reference)),
        missed=missed, extras=extras, equations=eq_outcomes, relationships=rel_outcomes, unexpected_relationships=unexpected,
        passages_covered=covered_passages, passages_total=len(dataset.passages),
    )


# -- relationships ---------------------------------------------------------------------------------


def _relationships(dataset, observation, matched, equation_units, units_by_id):
    outcomes: list[RelationshipOutcome] = []
    unit_of_ref = {key: cand.obs.id for key, cand in matched.items()} | equation_units
    ref_of_unit = {unit_id: key for key, unit_id in unit_of_ref.items()}
    stored = {}
    for rel in observation.relationships:
        stored.setdefault((rel.source_id, rel.target_id), []).append(rel)
        if rel.type in {"related_to", "contradicts"}:  # symmetric: either direction counts
            stored.setdefault((rel.target_id, rel.source_id), []).append(rel)
    relation_passages = observation.relationship_passages
    judged_pairs: set[tuple[str, str]] = set()
    for passage in dataset.passages:
        asked = not observation.executions or passage.key in relation_passages
        for relation in passage.relationships:
            source_key, target_key = f"{passage.key}/{relation.source}", f"{passage.key}/{relation.target}"
            source_id, target_id = unit_of_ref.get(source_key), unit_of_ref.get(target_key)
            found = type_ok = None
            stored_type = None
            if asked and source_id and target_id:
                judged_pairs.add((source_id, target_id))
                options = stored.get((source_id, target_id), [])
                found = bool(options)
                if options:
                    stored_type = options[0].type
                    type_ok = any(o.type in {t.value for t in relation.types} for o in options)
            outcomes.append(
                RelationshipOutcome(passage=passage.key, source=relation.source, target=relation.target,
                                    expected_types=[t.value for t in relation.types], found=bool(found), stored_type=stored_type,
                                    type_ok=type_ok, required=relation.required, covered=bool(asked and source_id and target_id))
            )
    measurable = [o for o in outcomes if o.covered and o.required]
    found_required = sum(o.found for o in measurable)
    typed = sum(bool(o.type_ok) for o in measurable)
    expected_pairs = {
        (unit_of_ref[f"{p.key}/{r.source}"], unit_of_ref[f"{p.key}/{r.target}"])
        for p in dataset.passages for r in p.relationships
        if f"{p.key}/{r.source}" in unit_of_ref and f"{p.key}/{r.target}" in unit_of_ref
    }
    expected_pairs |= {(b, a) for a, b in expected_pairs}
    unexpected: list[UnexpectedRelationship] = []

    def label(unit_id: str) -> str:
        return units_by_id[unit_id].title if unit_id in units_by_id else unit_id

    judged_observed = 0
    matched_rel = 0
    for rel in observation.relationships:
        both = rel.source_id in ref_of_unit and rel.target_id in ref_of_unit
        if both:
            judged_observed += 1
            if (rel.source_id, rel.target_id) in expected_pairs:
                matched_rel += 1
            else:
                unexpected.append(UnexpectedRelationship(source=label(rel.source_id), target=label(rel.target_id), type=rel.type,
                                                         origin=rel.origin, kind="not_in_reference"))
        else:
            unexpected.append(UnexpectedRelationship(source=label(rel.source_id), target=label(rel.target_id), type=rel.type,
                                                     origin=rel.origin, kind="unjudged"))
    metrics = [
        _metric("relationship_recall", "Required relationships found", "B. Semantic accuracy", found_required, len(measurable),
                "Required reference relationships whose two units were both found, and that a stored relationship connects (either direction for symmetric types)."),
        _metric("relationship_type_accuracy", "Found relationships of an accepted type", "B. Semantic accuracy", typed, found_required,
                "Of the found relationships, those whose type the reviewer accepts."),
        _metric("relationship_precision", "Relationships between recognised units that the reference expects", "B. Semantic accuracy", matched_rel, judged_observed,
                "Stored relationships between two recognised units that the reference lists. Relationships touching unrecognised units cannot be judged and are listed separately."),
    ]
    return outcomes, unexpected, metrics


def _metric(key, label, dimension, numerator, denominator, definition, note=None) -> Metric:
    return Metric(key=key, label=label, dimension=dimension, value=ratio(numerator, denominator), numerator=numerator,
                  denominator=denominator, definition=definition, note=note)


# -- equations -------------------------------------------------------------------------------------


def _equations(dataset, observation, passage_by_key) -> list[EquationOutcome]:
    outcomes: list[EquationOutcome] = []
    for passage in dataset.passages:
        refs = [u for u in passage.units if u.category.value == "equation"]
        pool = [u for u in observation.units if u.category == "equation" and u.primary_passage == passage.key and u.equation]
        taken: set[str] = set()
        for ref in refs:
            spec = ref.equation
            covered = _is_covered(observation, passage, [ref.category]) if observation.executions else True
            best, best_score = None, 0.0
            for unit in pool:
                if unit.id in taken:
                    continue
                eq = unit.equation
                text = " ".join([eq.original_expression, eq.latex, unit.title, eq.source_text or "", eq.source_latex or ""])
                score = token_coverage(spec.tokens, text)
                if any(status.value == "image_only" for status in spec.capture_status) and unit.origin == "source_derived":
                    score = max(score, 1.0)
                if score > best_score:
                    best, best_score = unit, score
            if best is None or best_score < EQUATION_MATCH:
                outcomes.append(EquationOutcome(passage=passage.key, reference=ref.key, covered=covered, found=False,
                                                handled_correctly=False if covered else None))
                continue
            taken.add(best.id)
            eq = best.equation
            reported = " ".join([eq.original_expression, eq.latex])
            preserved = token_coverage(spec.tokens, reported) >= EQUATION_PRESERVED
            source_ok: bool | None
            if spec.phase1 is not None:
                wanted = math_form(spec.phase1.plain_text or spec.phase1.latex or "")
                # an image-only equation has no source text to preserve
                source_ok = (wanted in math_form(f"{eq.source_text or ''}{eq.source_latex or ''}")) if wanted else None
            elif spec.in_phase1:
                source_ok = bool(eq.source_text or eq.source_latex)
            else:
                source_ok = None
            image = None if not spec.has_image else bool(eq.image_reference)
            low, high = passage_by_key[passage.key].pages
            page = None if eq.page_start is None else low <= eq.page_start <= high
            status_ok = eq.capture_status in {s.value for s in spec.capture_status}
            flagged = bool(set(best.flags) & FLAGGING_CODES) or best.is_uncertain or best.validation_status != "valid" or (eq.capture_status in UNCLEAR_CAPTURE)
            flag_ok = flagged if spec.must_be_flagged else True
            outcomes.append(
                EquationOutcome(passage=passage.key, reference=ref.key, covered=covered, found=True, observed_id=best.id,
                                token_coverage=round(best_score, 3), original_preserved=preserved, source_preserved=source_ok,
                                image_retained=image, page_retained=page, capture_status=eq.capture_status, status_ok=status_ok,
                                flagged=flagged, flag_ok=flag_ok, handled_correctly=status_ok and flag_ok)
            )
    return outcomes


# -- operations ------------------------------------------------------------------------------------


def _operations(observation: Observation, metrics: list[Metric]) -> None:
    executions = observation.executions
    runs = observation.runs

    def add(key, label, value, definition, unit="count", numerator=None, denominator=None, note=None):
        metrics.append(Metric(key=key, label=label, dimension="Operations", value=value, numerator=numerator,
                              denominator=denominator, unit=unit, definition=definition, note=note))

    calls = sum(e.attempts for e in executions)
    retries = sum(e.retries for e in executions)
    failed = sum(e.status == "failed" for e in executions)
    review = sum(e.status == "needs_review" for e in executions)
    add("executions", "Extractions (passage x category, plus relationship steps)", float(len(executions)), "Executions recorded by the evaluated runs.")
    add("model_calls", "Model calls", float(calls), "Calls to the model, including retries.")
    add("retry_rate", "Retries per extraction", ratio(retries, len(executions)), "Retries (corrections and resends) divided by executions.",
        unit="ratio", numerator=float(retries), denominator=float(len(executions)))
    add("failure_rate", "Failed extractions", ratio(failed, len(executions)), "Executions that ended failed.", unit="ratio",
        numerator=float(failed), denominator=float(len(executions)))
    add("needs_review_rate", "Extractions needing review", ratio(review, len(executions)), "Executions that ended needs_review.",
        unit="ratio", numerator=float(review), denominator=float(len(executions)))
    seconds = sum(e.duration_ms for e in executions) / 1000 if executions else None
    add("processing_seconds", "Model time (sum of extractions)", seconds, "Sum of the recorded duration of every execution.", unit="seconds")
    passages = len({e.passage for e in executions if e.passage})
    add("seconds_per_passage", "Model time per passage", ratio(seconds or 0, passages), "Model time divided by the passages extracted.", unit="seconds")
    known = [r.duration_seconds for r in runs if r.duration_seconds is not None]
    add("run_seconds", "Wall-clock duration of the runs", sum(known) if known else None, "Sum of the runs' start-to-finish time.", unit="seconds")
    tokens_in = [r.input_tokens for r in runs if r.input_tokens is not None]
    tokens_out = [r.output_tokens for r in runs if r.output_tokens is not None]
    add("input_tokens", "Input tokens", float(sum(tokens_in)) if tokens_in else None, "Tokens sent, as reported by the provider (unavailable if it reports none).", unit="tokens")
    add("output_tokens", "Output tokens", float(sum(tokens_out)) if tokens_out else None, "Tokens received, as reported by the provider.", unit="tokens")
    costs = [r.cost_usd for r in runs if r.cost_usd is not None]
    add("estimated_cost", "Estimated cost", float(sum(costs)) if costs else None,
        "Estimated from the configured token prices; unavailable if the model has no prices.", unit="usd")
