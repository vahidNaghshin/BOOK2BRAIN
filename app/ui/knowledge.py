"""Knowledge screen: browse what the model extracted, see the evidence, and record review decisions."""

import logging
import uuid

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from app.knowledge.types import KnowledgeType
from app.repositories.extraction_repository import ExtractionRepository
from app.services.errors import InvalidReviewError, KnowledgeNotFoundError
from app.services.knowledge_service import KnowledgeDetail, KnowledgeFilter, KnowledgeOverview, KnowledgeService, KnowledgeSummary
from app.ui.formatting import format_timestamp
from app.ui.navigation import goto
from app.ui.selection import render_document_selector, safe_load_library
from database.connection import session_scope

logger = logging.getLogger(__name__)

PAGE_SIZE = 15
ALL = "All"
REVIEW_LABELS = {"unreviewed": "Unreviewed", "approved": "Approved", "rejected": "Rejected", "needs_changes": "Needs changes"}
VALIDATION_LABELS = {"valid": "Valid", "uncertain": "Uncertain", "invalid": "Invalid", "pending": "Pending"}
FLASH_KEY = "knowledge_flash"


def render_knowledge_page() -> None:
    st.header("Knowledge")
    st.caption("What the model extracted: read it with its source passage, see how it was produced, and approve or reject it.")
    books = safe_load_library()
    if books is None:
        return
    selection = render_document_selector(books, predicate=lambda d: True, key="knowledge")
    if selection is None:
        st.info("No documents yet. Upload a book first.")
        return
    book_entry, entry = selection
    book, document = book_entry.book, entry.document
    base = KnowledgeFilter(book_id=book.id, document_id=document.id)

    try:
        with session_scope() as session:
            service = KnowledgeService(session)
            summary = service.summary(base)
            overview = service.overview(base)
            runs = ExtractionRepository(session).list_runs(book.id, limit=20)
    except SQLAlchemyError:
        logger.exception("Could not load knowledge")
        st.error("Could not load knowledge. Check the database connection and that migrations have been applied.")
        return
    if summary.total == 0:
        st.info("Nothing has been extracted from this document yet.")
        if st.button("Go to Extraction", key="knowledge_go_extraction"):
            goto("Extraction", document_id=document.id)
        return

    _render_summary(summary)
    _render_overview(overview)
    filters = _render_filters(base, summary, runs)
    page_unit = _render_list(filters)
    if page_unit is not None:
        st.divider()
        _render_detail(page_unit)


# -- list ------------------------------------------------------------------------------------------


def _render_summary(summary: KnowledgeSummary) -> None:
    columns = st.columns(5)
    columns[0].metric("Knowledge units", summary.total)
    columns[1].metric("Need attention", summary.needing_attention, help="Uncertain, or not validated.")
    columns[2].metric("Approved", summary.by_review.get("approved", 0))
    columns[3].metric("Rejected", summary.by_review.get("rejected", 0))
    columns[4].metric("Unreviewed", summary.by_review.get("unreviewed", 0))


def _count_table(counts: dict[str, int], heading: str, empty: str) -> None:
    st.markdown(f"**{heading}**")
    if counts:
        st.dataframe(
            [{"": _label(key), "Count": total} for key, total in sorted(counts.items(), key=lambda item: (-item[1], item[0]))], hide_index=True
        )
    else:
        st.caption(empty)


def _render_overview(overview: KnowledgeOverview) -> None:
    attempts = sum(overview.attempt_outcomes.values())
    problems = attempts - overview.attempt_outcomes.get("schema_valid", 0)
    label = f"Overview: {len(overview.failed_extractions)} failed extraction(s), {sum(overview.open_flags.values())} open review flag(s)"
    with st.expander(label, expanded=bool(overview.failed_extractions)):
        columns = st.columns(4)
        with columns[0]:
            _count_table(overview.by_type, "Units by category", "No units.")
        with columns[1]:
            _count_table(overview.equation_status, "Equations by capture status", "No equations.")
        with columns[2]:
            _count_table(overview.relationships_by_type, "Relationships by type", "No relationships.")
        with columns[3]:
            _count_table(overview.open_flags, "Open review flags", "No open flags.")
        st.markdown("**Model calls**")
        if attempts:
            st.caption(f"{attempts} call(s); {problems} did not end as a clean, schema-valid answer. " + ", ".join(f"{_label(k)}: {v}" for k, v in sorted(overview.attempt_outcomes.items())))
        else:
            st.caption("No model calls are recorded.")
        st.markdown("**Failed extractions** (every attempt was used and nothing could be stored)")
        if overview.failed_extractions:
            st.dataframe(
                [
                    {
                        "Page": f.page_start or "-", "Category": f.category, "Attempts": f.attempts, "Reason": f.failure_code or "",
                        "Message": (f.failure_message or "")[:160], "Model": f.model or "", "Prompt": f.prompt or "",
                    }
                    for f in overview.failed_extractions
                ],
                hide_index=True,
            )
        else:
            st.caption("None.")


def _render_filters(base: KnowledgeFilter, summary: KnowledgeSummary, runs) -> KnowledgeFilter:
    first = st.columns(4)
    kinds = [ALL, *[t.value for t in KnowledgeType if summary.by_type.get(t.value)]]
    kind = first[0].selectbox(
        "Category", kinds, key="knowledge_type", format_func=lambda v: v if v == ALL else f"{v.capitalize()} ({summary.by_type[v]})"
    )
    validation = first[1].selectbox(
        "Validation", [ALL, *summary.by_validation], key="knowledge_validation", format_func=lambda v: VALIDATION_LABELS.get(v, v)
    )
    review = first[2].selectbox("Review", [ALL, *REVIEW_LABELS], key="knowledge_review", format_func=lambda v: REVIEW_LABELS.get(v, v))
    run_options = {str(r.id): r for r in runs}
    run = first[3].selectbox(
        "Extraction run", [ALL, *run_options], key="knowledge_run",
        format_func=lambda v: v if v == ALL else f"{format_timestamp(run_options[v].started_at or run_options[v].created_at)} · {run_options[v].model_name or 'model not recorded'}",
    )
    second = st.columns([3, 1])
    search = second[0].text_input("Search title and text", key="knowledge_search")
    attention = second[1].checkbox("Only uncertain or unvalidated", key="knowledge_attention")
    return KnowledgeFilter(
        book_id=base.book_id,
        document_id=base.document_id,
        knowledge_type=None if kind == ALL else kind,
        validation_status=None if validation == ALL else validation,
        review_status=None if review == ALL else review,
        run_id=None if run == ALL else uuid.UUID(run),
        uncertain_only=attention,
        search=search or None,
    )


def _render_list(filters: KnowledgeFilter) -> uuid.UUID | None:
    signature = str(filters)
    if st.session_state.get("knowledge_filter_signature") != signature:
        st.session_state["knowledge_filter_signature"] = signature
        st.session_state["knowledge_page"] = 1
    with session_scope() as session:
        service = KnowledgeService(session)
        probe = service.list_units(filters, page=1, page_size=PAGE_SIZE)
        page_number = 1
        if probe.pages > 1:
            page_number = int(
                st.number_input(f"Page (of {probe.pages})", min_value=1, max_value=probe.pages, step=1, key="knowledge_page")
            )
        page = service.list_units(filters, page=page_number, page_size=PAGE_SIZE) if page_number != 1 else probe
    if not page.items:
        st.info("No knowledge matches these filters.")
        return None
    st.caption(f"{page.total} unit(s) · showing {len(page.items)} on page {page.page} of {page.pages}")
    st.table(
        [
            {
                "Category": item.knowledge_type,
                "Title": item.title,
                "Page": item.page_start or "-",
                "Validation": VALIDATION_LABELS.get(item.validation_status, item.validation_status),
                "Uncertain": "yes" if item.is_uncertain else "",
                "Review": REVIEW_LABELS.get(item.review_status, item.review_status),
            }
            for item in page.items
        ]
    )
    labels = {str(i.id): f"[{i.knowledge_type}] {i.title} · p.{i.page_start or '?'}" for i in page.items}
    if st.session_state.get("knowledge_unit") not in labels:
        st.session_state.pop("knowledge_unit", None)
    chosen = st.selectbox("Open a unit", list(labels), format_func=lambda v: labels[v], key="knowledge_unit")
    return uuid.UUID(chosen)


# -- one unit --------------------------------------------------------------------------------------


def _label(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _render_value(key: str, value) -> None:
    if value in (None, "", [], {}):
        return
    st.markdown(f"**{_label(key)}**")
    if key == "latex" and isinstance(value, str):
        try:
            st.latex(value)
        except Exception:  # unparseable LaTeX is shown as text instead
            pass
        st.code(value, language="latex")
    elif key == "original_expression" and isinstance(value, str):
        st.code(value, language="text")
    elif isinstance(value, str):
        st.markdown(value)
    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        st.markdown("\n".join(f"- {v}" for v in value))
    elif isinstance(value, list) and all(isinstance(v, dict) for v in value):
        st.table([{_label(k): ("" if v is None else v) for k, v in row.items()} for row in value])
    elif isinstance(value, dict):
        st.json(value)
    else:
        st.write(value)


def _load_detail(unit_id: uuid.UUID) -> KnowledgeDetail | None:
    try:
        with session_scope() as session:
            return KnowledgeService(session).detail(unit_id)
    except KnowledgeNotFoundError:
        st.warning("This knowledge unit no longer exists.")
        return None


def _render_detail(unit_id: uuid.UUID) -> None:
    detail = _load_detail(unit_id)
    if detail is None:
        return
    unit = detail.unit
    flash = st.session_state.pop(FLASH_KEY, None)
    if flash:
        st.success(flash)

    st.subheader(unit.content.title)
    place = " › ".join(t for t in (detail.chapter_title, detail.section_title) if t)
    pages = "-" if unit.page_start is None else (f"p.{unit.page_start}" if unit.page_end in (None, unit.page_start) else f"pp.{unit.page_start}-{unit.page_end}")
    st.caption(f"{unit.knowledge_type.value.capitalize()} · {pages}" + (f" · {place}" if place else ""))
    st.write(
        f"Validation: **{VALIDATION_LABELS.get(unit.validation_status.value, unit.validation_status.value)}** · "
        f"Review: **{REVIEW_LABELS.get(unit.review_status.value, unit.review_status.value)}**"
        + (f" · Model confidence: **{unit.confidence:.0%}**" if unit.confidence is not None else "")
        + (f" · Basis: **{unit.metadata.get('basis')}**" if unit.metadata.get("basis") else "")
    )
    if unit.uncertainty.is_uncertain:
        st.warning(f"The model flagged this as uncertain. {unit.uncertainty.notes or ''}")
    if detail.review_note:
        st.info(f"Review note: {detail.review_note}" + (f" ({format_timestamp_text(detail.reviewed_at)})" if detail.reviewed_at else ""))

    st.markdown("##### Extracted content")
    for key, value in unit.content.model_dump(mode="json", exclude={"knowledge_type"}).items():
        _render_value(key, value)

    st.markdown("##### Source")
    _render_sources(detail)
    if detail.same_title_as:
        _render_same_title(detail)
    if detail.equation is not None:
        _render_equation(detail)
    _render_relationships(detail)
    _render_flags(detail)
    _render_execution(detail)
    _render_review(detail)


SOURCE_LABELS = {"unverified": "not verified yet", "verified": "verified", "invalid": "problems found"}
ORIGIN_LABELS = {
    "ai_extracted": "extracted by a model", "source_derived": "copied from the source (no model)",
    "ai_interpretation": "model interpretation", "ai_reconstructed": "model reconstruction", "human": "human",
}
RELATIONSHIP_LABELS = {"extracted": "evidence from the book", "ai_suggested": "AI suggestion, needs review", "manual": "manual"}
CAPTURE_LABELS = {
    "complete": "Complete", "incomplete": "Incomplete", "ambiguous": "Ambiguous", "image_only": "Image only",
    "text_only": "Text only (no LaTeX)", "needs_review": "Needs review",
}


def _render_equation(detail: KnowledgeDetail) -> None:
    capture = detail.equation
    st.markdown("##### Equation as captured from the source")
    st.write(f"Capture status: **{CAPTURE_LABELS.get(capture.capture_status.value, capture.capture_status.value)}**")
    for note in capture.capture_notes:
        st.caption(f"• {note}")
    st.caption("This is what the source gave, kept apart from the model's reading above. It is never overwritten.")
    if capture.original_text:
        st.markdown("**Original text**")
        st.code(capture.original_text, language="text")
    if capture.source_latex:
        st.markdown("**LaTeX from the source extraction**")
        st.code(capture.source_latex, language="latex")
    if capture.image_reference:
        st.markdown(f"**Image**: `{capture.image_reference}`")
    if capture.context_before or capture.context_after or capture.surrounding_text:
        with st.expander("Surrounding text"):
            for label, text in (("Before", capture.context_before), ("After", capture.context_after), ("Around (stored with the equation)", capture.surrounding_text)):
                if text:
                    st.markdown(f"**{label}**")
                    st.markdown(text)


def _render_relationships(detail: KnowledgeDetail) -> None:
    related = detail.relationships.all if detail.relationships else []
    if not related:
        return
    st.markdown("##### Related knowledge")
    rows = []
    for view in related:
        rel = view.relationship
        kind = rel.relationship_type.value.replace("_", " ")
        rows.append(
            {
                "Relationship": f"this {kind} → {view.other_title}" if view.direction == "outgoing" else f"{view.other_title} {kind} → this",
                "Other unit": f"[{view.other_type.value}] {view.other_title}",
                "Basis": RELATIONSHIP_LABELS.get(rel.origin.value, rel.origin.value),
                "Review": REVIEW_LABELS.get(rel.review_status.value, rel.review_status.value),
                "Why": rel.explanation or "",
                "Evidence": rel.evidence_quote or "",
            }
        )
    st.table(rows)


def _render_flags(detail: KnowledgeDetail) -> None:
    if not detail.flags:
        return
    st.markdown("##### Review flags")
    for flag in detail.flags:
        icon = {"error": "🔴", "warning": "🟠", "info": "🔵"}.get(flag.severity.value, "")
        columns = st.columns([5, 1])
        columns[0].markdown(f"{icon} **{flag.code.value.replace('_', ' ')}** · raised by {flag.raised_by}\n\n{flag.message}")
        if columns[1].button("Resolve", key=f"knowledge_flag_{flag.id}"):
            with session_scope() as session:
                KnowledgeService(session).resolve_flag(flag.id, "Resolved in the Knowledge screen.")
            st.rerun()


def format_timestamp_text(value: str) -> str:
    return value.replace("T", " ")[:16]


def _render_sources(detail: KnowledgeDetail) -> None:
    unit_id = detail.unit.id
    columns = st.columns([3, 1])
    columns[0].caption(
        f"Source check: **{SOURCE_LABELS.get(detail.source_status, detail.source_status)}** · "
        f"origin: {ORIGIN_LABELS.get(detail.origin, detail.origin)}"
    )
    if columns[1].button("Verify sources", key=f"knowledge_verify_{unit_id}"):
        try:
            with session_scope() as session:
                report = KnowledgeService(session).verify_sources(unit_id)
        except KnowledgeNotFoundError as exc:
            st.error(str(exc))
        else:
            st.session_state[FLASH_KEY] = (
                f"Sources verified: {len(report.issues)} problem(s) found, {report.flags_raised} flag(s) raised."
                if report.issues else "Sources verified: every reference checks out."
            )
            st.rerun()
    if not detail.sources:
        st.warning("The source passages of this unit no longer exist (the document's passages were regenerated). Its page range and the source snapshot are kept.")
        if detail.source_snapshot:
            with st.expander("Source snapshot taken when the unit was stored"):
                st.table(
                    [
                        {
                            "Role": snap.get("role"), "Chapter": snap.get("chapter") or "-", "Section": snap.get("section") or "-",
                            "Pages": "-".join(str(p) for p in dict.fromkeys(snap.get("pages") or []) if p) or "-",
                            "Excerpt": ("“" + snap.get("excerpt", "") + "”") if snap.get("excerpt_is_quote") else snap.get("excerpt", "")[:160],
                        }
                        for snap in detail.source_snapshot
                    ]
                )
    for source in detail.sources:
        pages = "pages unknown" if source.page_start is None else (f"p.{source.page_start}" if source.page_end in (None, source.page_start) else f"pp.{source.page_start}-{source.page_end}")
        mark = {"verified": "✅", "invalid": "⚠️"}.get(source.verification_status, "▫️")
        st.markdown(f"**{source.role.capitalize()} passage** · {pages} · {mark} {source.verification_status}")
        for issue in source.verification_issues:
            st.caption(f"⚠️ {issue.get('message')}")
        if source.quote:
            st.markdown(f"> {source.quote}")
            if source.quote_found is True:
                st.caption("✅ The quote appears in the passage.")
            elif source.quote_found is False:
                st.caption("⚠️ The quote was not found word for word in the passage. Check it against the text.")
        if source.text is None:
            st.caption("The passage text is no longer available.")
            continue
        with st.expander("Passage text", expanded=source.role == "primary" and not source.quote):
            text = source.text
            if source.quote and source.quote in text:
                text = text.replace(source.quote, f"**{source.quote}**", 1)
            st.markdown(text)


def _render_same_title(detail: KnowledgeDetail) -> None:
    with session_scope() as session:
        others = KnowledgeService(session).items(detail.same_title_as)
    if others:
        st.caption(
            "The same title was also extracted from other passages (separate occurrences, not merged): "
            + ", ".join(f"p.{o.page_start or '?'}" for o in others)
        )


def _render_execution(detail: KnowledgeDetail) -> None:
    execution = detail.execution
    with st.expander("How it was extracted"):
        if execution is None:
            st.caption("No extraction record is linked to this unit.")
            return
        st.write(
            f"Model: **{execution.model_configuration_id or '-'}** ({execution.provider or '-'}, {execution.model_id or '-'}"
            + (f", version {execution.model_version}" if execution.model_version else "")
            + f") · Prompt: **{execution.prompt or '-'}**"
        )
        st.caption(
            f"{execution.attempts} attempt(s) · {execution.retries} retries · {execution.validation_failures} validation failure(s) · "
            f"{execution.duration_ms / 1000:.1f}s · {execution.input_tokens or 0} input / {execution.output_tokens or 0} output tokens"
            + (f" · ${execution.estimated_cost_usd:.4f}" if execution.estimated_cost_usd is not None else "")
        )
        if execution.review_reasons:
            st.markdown("**Why it was flagged for review**\n" + "\n".join(f"- {r}" for r in execution.review_reasons))
        if execution.context and execution.context.get("reduced"):
            st.markdown(f"**Context was reduced** to fit the model: {execution.context.get('sent')} of {execution.context.get('requested')} neighbouring passages were sent.")
        if detail.attempts:
            st.markdown("**Attempts**")
            st.table(
                [
                    {
                        "#": a.attempt_number, "Kind": a.kind, "Call": a.status, "Validation": a.validation_outcome.replace("_", " "),
                        "Model": f"{a.model_id}", "Prompt": a.prompt, "Seconds": f"{a.duration_ms / 1000:.1f}",
                        "Tokens in/out": f"{a.input_tokens or 0}/{a.output_tokens or 0}",
                    }
                    for a in detail.attempts
                ]
            )
            with st.expander("Model configuration as it was for these attempts"):
                st.json(detail.attempts[0].model_configuration or {})
        if execution.issues:
            st.markdown("**Checks and warnings**")
            st.table([{"Stage": i.get("stage"), "Severity": i.get("severity"), "Code": i.get("code"), "Message": i.get("message")} for i in execution.issues])
        st.caption(f"Extraction run {execution.run_id} · execution {execution.id}")


def _render_review(detail: KnowledgeDetail) -> None:
    unit_id = detail.unit.id
    st.markdown("##### Review")
    note = st.text_area("Note (optional)", value=detail.review_note or "", key=f"knowledge_note_{unit_id}", height=80)
    actions = [("approved", "Approve", "primary"), ("rejected", "Reject", "secondary"), ("needs_changes", "Needs changes", "secondary"), ("unreviewed", "Reset", "secondary")]
    columns = st.columns(len(actions))
    for column, (status, label, kind) in zip(columns, actions, strict=True):
        if column.button(label, key=f"knowledge_{status}_{unit_id}", type=kind, disabled=detail.unit.review_status.value == status and note.strip() == (detail.review_note or "")):
            try:
                with session_scope() as session:
                    KnowledgeService(session).review(unit_id, status, note)
            except (KnowledgeNotFoundError, InvalidReviewError) as exc:
                st.error(str(exc))
                return
            st.session_state[FLASH_KEY] = f"Marked as {REVIEW_LABELS[status].lower()}."
            st.rerun()
