"""Evaluation reports: a Markdown document for people and JSON for tools, from `Evaluation` results."""

from __future__ import annotations

import json
from collections import defaultdict

from app.evaluation.metrics import Evaluation, Metric

LIMITS = """\
* The reference was prepared before any extraction was inspected, but it is one reviewer's reading. Where its status is `draft` it still needs a second, human review.
* Matching is lexical: names, key words and symbols. It shows whether expected knowledge is present, correctly placed and traceable. It cannot prove that a paraphrase keeps the author's meaning; the matched pairs below are listed for a reviewer to read.
* Precision counts unmatched units only in passages whose reference is exhaustive; elsewhere they are listed for judgement, not scored.
* A value of `n/a` means the denominator was zero or the provider did not report the figure. It is never shown as zero.
* Small datasets give coarse ratios: read the counts next to every percentage.
"""


def format_value(metric: Metric) -> str:
    if metric.value is None:
        return "n/a"
    if metric.unit == "ratio":
        counts = ""
        if metric.numerator is not None and metric.denominator is not None:
            counts = f" ({metric.numerator:g}/{metric.denominator:g})"
        return f"{metric.value:.1%}{counts}"
    if metric.unit == "seconds":
        return f"{metric.value:.1f} s"
    if metric.unit == "usd":
        return f"${metric.value:.4f}"
    return f"{metric.value:,.0f}"


def run_label(evaluation: Evaluation, index: int = 0) -> str:
    runs = evaluation.runs
    if not runs:
        return f"run {index + 1}"
    return ", ".join(dict.fromkeys(r.model_configuration_id or r.model or "unknown" for r in runs))


def _by_dimension(metrics: list[Metric]) -> dict[str, list[Metric]]:
    grouped: dict[str, list[Metric]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.dimension].append(metric)
    return dict(sorted(grouped.items()))


def _runs_table(evaluation: Evaluation) -> list[str]:
    lines = ["| Run | Status | Configuration | Provider / model | Prompt versions | Parameters | Duration | Tokens in / out | Est. cost |", "|---|---|---|---|---|---|---|---|---|"]
    for run in evaluation.runs:
        scope = {k: v for k, v in run.parameters.items() if k in ("categories", "context_passages", "extract_relationships", "limit", "offset")}
        shown = json.dumps({"generation": run.generation, **scope}, ensure_ascii=False, sort_keys=True)
        lines.append(
            f"| `{run.run_id[:8]}` | {run.status} | {run.model_configuration_id or 'n/a'} | {run.provider or 'n/a'} / {run.model or 'n/a'} | "
            f"{', '.join(run.prompt_versions) or 'n/a'} | `{shown}` | "
            f"{'n/a' if run.duration_seconds is None else f'{run.duration_seconds:.1f} s'} | {run.input_tokens if run.input_tokens is not None else 'n/a'} / "
            f"{run.output_tokens if run.output_tokens is not None else 'n/a'} | {'n/a' if run.cost_usd is None else f'${run.cost_usd:.4f}'} |"
        )
    return lines


def render_evaluation(evaluation: Evaluation, *, heading_level: int = 2) -> str:
    h = "#" * heading_level
    out = [
        f"{h} {evaluation.dataset} v{evaluation.dataset_version} — {run_label(evaluation)}",
        "",
        f"Reference status: **{evaluation.review_status}** · dataset fingerprint `{evaluation.dataset_digest}` · "
        f"passages extracted: {evaluation.passages_covered} of {evaluation.passages_total}",
        "",
        *(_runs_table(evaluation) if evaluation.runs else ["No run information."]),
        "",
    ]
    for dimension, metrics in _by_dimension(evaluation.metrics).items():
        out += [f"{h}# {dimension}", "", "| Metric | Result | Note |", "|---|---|---|"]
        out += [f"| {m.label} | {format_value(m)} | {m.note or ''} |" for m in metrics]
        out.append("")

    required_missed = [m for m in evaluation.missed if m.required and m.reason == "not_extracted"]
    out += [f"{h}# Required units not found", ""]
    if required_missed:
        out += ["| Passage | Unit | Category | Names | Seems merged into |", "|---|---|---|---|---|"]
        out += [f"| {m.passage} | {m.reference} | {m.category} | {'; '.join(m.names[:2])} | {m.merged_into or ''} |" for m in required_missed]
    else:
        out.append("None.")
    not_attempted = [m for m in evaluation.missed if m.reason == "not_attempted"]
    if not_attempted:
        out += ["", f"{len(not_attempted)} reference unit(s) belong to a category that was not extracted on their passage and are not scored."]
    out.append("")

    out += [f"{h}# Extracted units without a reference match", ""]
    if evaluation.extras:
        out += ["| Passage | Category | Title | Kind | Counted against precision |", "|---|---|---|---|---|"]
        out += [f"| {e.passage} | {e.category} | {e.title} | {e.kind}{' of ' + e.of_reference if e.of_reference else ''} | {'yes' if e.counted else 'no — needs a reviewer'} |" for e in evaluation.extras]
    else:
        out.append("None.")
    out.append("")

    out += [f"{h}# Matched pairs for review", "", "| Passage | Reference | Extracted title | Recognised by | Facts present | Missing facts | Source | Pages |", "|---|---|---|---|---|---|---|---|"]
    for m in evaluation.matches:
        out.append(
            f"| {m.passage} | {m.reference} | {m.observed_title} | {'name' if m.name_score >= 0.5 else 'content only: read it'} | "
            f"{'n/a' if m.fact_coverage is None else f'{m.fact_coverage:.0%}'} | {'; '.join(m.facts_missing) or '—'}"
            f"{' ⚠ forbidden: ' + '; '.join(m.forbidden_hits) if m.forbidden_hits else ''} | "
            f"{'correct' if m.source_correct else 'WRONG passage'} | {'n/a' if m.pages_ok is None else ('ok' if m.pages_ok else 'outside')} |"
        )
    out.append("")

    if evaluation.equations:
        out += [f"{h}# Equations", "", "| Passage | Reference | Found | Status | Original kept | Source kept | Image | Page | Flagged | Handled correctly |", "|---|---|---|---|---|---|---|---|---|---|"]

        def mark(value):
            return "n/a" if value is None else ("yes" if value else "no")

        for e in evaluation.equations:
            if not e.covered and not e.found:
                out.append(f"| {e.passage} | {e.reference} | not scored: equations were not extracted on this passage | | | | | | | |")
                continue
            out.append(
                f"| {e.passage} | {e.reference} | {mark(e.found)}{'' if e.covered else ' (read while extracting a neighbouring passage)'} | {e.capture_status or 'n/a'} | {mark(e.original_preserved)} | "
                f"{mark(e.source_preserved)} | {mark(e.image_retained)} | {mark(e.page_retained)} | {mark(e.flagged)} | {mark(e.handled_correctly)} |"
            )
        out.append("")

    judged = [r for r in evaluation.relationships if r.covered]
    if judged or evaluation.unexpected_relationships:
        out += [f"{h}# Relationships", ""]
        if judged:
            out += ["| Passage | Expected | Required | Found | Stored type | Type accepted |", "|---|---|---|---|---|---|"]
            out += [
                f"| {r.passage} | {r.source} → {r.target} ({'/'.join(r.expected_types)}) | {'yes' if r.required else 'no'} | {'yes' if r.found else 'no'} | "
                f"{r.stored_type or ''} | {'' if r.type_ok is None else ('yes' if r.type_ok else 'no')} |"
                for r in judged
            ]
        if evaluation.unexpected_relationships:
            out += ["", "Stored relationships the reference does not list:", "", "| Source | Type | Target | Origin | Why not scored |", "|---|---|---|---|---|"]
            out += [
                f"| {u.source} | {u.type} | {u.target} | {u.origin} | {'both units recognised, pair not in the reference' if u.kind == 'not_in_reference' else 'touches a unit with no reference match'} |"
                for u in evaluation.unexpected_relationships
            ]
        out.append("")
    return "\n".join(out)


COMPARISON_KEYS = [
    "unit_recall", "unit_precision", "key_fact_coverage", "forbidden_phrase_units", "relationship_recall", "primary_source_accuracy",
    "page_accuracy", "provenance_verified", "quote_verification", "schema_valid_calls", "first_attempt_success", "category_accuracy",
    "duplicate_rate", "equation_found", "equation_original_preserved", "equation_status_correct", "equation_handled", "model_calls",
    "retry_rate", "failure_rate", "needs_review_rate", "seconds_per_passage", "run_seconds", "input_tokens", "output_tokens", "estimated_cost",
]


def render_comparison(evaluations: list[Evaluation], labels: list[str] | None = None, *, heading_level: int = 2) -> str:
    """One table, metrics down, runs across. Nothing is ranked: a higher number is not declared better."""
    h = "#" * heading_level
    labels = labels or [run_label(e, i) for i, e in enumerate(evaluations)]
    out = [
        f"{h} Comparison on {evaluations[0].dataset} v{evaluations[0].dataset_version}", "",
        "Same passages, categories, prompt versions and schemas for every column. No column is ranked: a larger or dearer model is not assumed better, and a metric is only comparable where both denominators are comparable.", "",
        "| Metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels),
    ]
    for key in COMPARISON_KEYS:
        try:
            cells = [format_value(e.metric(key)) for e in evaluations]
            label = evaluations[0].metric(key).label
        except StopIteration:
            continue
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def render_report(evaluations: list[Evaluation], *, title: str, labels: list[str] | None = None, repeated: bool = False) -> str:
    out = [f"# {title}", ""]
    if len(evaluations) > 1:
        out += [render_comparison(evaluations, labels), ""]
    for index, evaluation in enumerate(evaluations):
        out += [render_evaluation(evaluation), ""]
    out += ["## How to read this report", "", LIMITS]
    seen: dict[str, str] = {}
    for evaluation in evaluations:
        for metric in evaluation.metrics:
            seen.setdefault(metric.key, f"| {metric.label} | {metric.definition} |")
    out += ["", "## Metric definitions", "", "| Metric | Counting method |", "|---|---|", *seen.values(), ""]
    return "\n".join(out)


def to_json(evaluations: list[Evaluation], labels: list[str] | None = None) -> str:
    payload = [{"label": (labels[i] if labels else run_label(e, i)), **e.model_dump(mode="json")} for i, e in enumerate(evaluations)]
    return json.dumps(payload, indent=2, ensure_ascii=False)
