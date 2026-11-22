"""Knowledge extraction screen: choose a book, a model and a scope, start a run, follow its progress."""

import logging
import uuid

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.knowledge.types import KnowledgeType
from app.repositories.passage_repository import CHAPTER_TYPES, SECTION_TYPES, PassageRepository
from app.services.errors import ExtractionRunError
from app.services.extraction_job_runner import ExtractionJobRunner
from app.services.knowledge_extraction_service import (
    KnowledgeExtractionService,
    ModelOption,
    NothingToExtractError,
    RunStatus,
    create_extraction_service,
)
from app.ui.formatting import format_timestamp
from app.ui.navigation import DOCUMENT_KEY, goto
from app.ui.selection import render_document_selector, safe_load_library
from database.connection import session_scope

logger = logging.getLogger(__name__)

POLL_SECONDS = 2
RUN_KEY = "extraction_run_id"
STATUS_LABELS = {
    "pending": "Waiting to start",
    "processing": "Processing",
    "completed": "Completed",
    "needs_review": "Completed, needs review",
    "failed": "Failed",
    "cancelled": "Cancelled",
}
SCOPES = ["Whole document", "A chapter", "A section"]


@st.cache_resource
def get_job_runner() -> ExtractionJobRunner:
    return ExtractionJobRunner()


def _service(session: Session) -> KnowledgeExtractionService:
    return create_extraction_service(session)


def _model_label(option: ModelOption) -> str:
    return f"{option.id} · {option.model} ({option.deployment})"


def render_extraction_page() -> None:
    st.header("Knowledge extraction")
    st.caption("Extract concepts, principles, methods and other knowledge from the passages of a processed book with an LLM.")
    books = safe_load_library()
    if books is None:
        return
    selection = render_document_selector(
        books, predicate=lambda d: d.document.processing_status == "completed", key="extraction"
    )
    if selection is None:
        st.info("No processed documents yet. Upload a book and process it first; its status is shown in the Library.")
        return
    book_entry, entry = selection
    book, document = book_entry.book, entry.document

    try:
        with session_scope() as session:
            service = _service(session)
            models = service.available_models()
            outline = PassageRepository(session).outline(document.id)
            runs = service.list_runs(book.id, limit=10)
    except SQLAlchemyError:
        logger.exception("Could not load the extraction screen")
        st.error("Could not load extraction data. Check the database connection and that migrations have been applied.")
        return

    active = next((r for r in runs if r.is_active), None)
    _render_form(book.id, document.id, models, outline, active)
    st.divider()
    _render_runs(book.id, runs)


# -- starting a run --------------------------------------------------------------------------------


def _render_form(book_id, document_id, models: list[ModelOption], outline, active: RunStatus | None) -> None:
    ready = [m for m in models if m.ready]
    unavailable = [m for m in models if not m.ready]

    st.subheader("New extraction")
    if not ready:
        st.error("No model is ready to use. See the unavailable models below, or the README section on configuring the LLM provider.")
    else:
        by_id = {m.id: m for m in ready}
        st.selectbox(
            "Model",
            options=list(by_id),
            format_func=lambda value: _model_label(by_id[value]),
            key="extraction_model",
            help="The model used for every category of this run. It is recorded with every result.",
        )
    if unavailable:
        with st.expander(f"Unavailable models ({len(unavailable)})"):
            for option in unavailable:
                st.write(f"**{option.id}** · {option.model}: {'inactive' if not option.active else option.reason}")

    scope_label = st.radio("Scope", SCOPES, horizontal=True, key="extraction_scope")
    node_id = _choose_node(document_id, outline, scope_label)

    categories = st.multiselect(
        "Knowledge categories",
        options=[t.value for t in KnowledgeType],
        default=[t.value for t in KnowledgeType],
        format_func=lambda value: value.capitalize(),
        key="extraction_categories",
    )
    with st.expander("Options"):
        limit = st.number_input("Limit to the first N passages (0 = all)", min_value=0, value=0, step=1, key="extraction_limit")
        context = st.number_input(
            "Neighbouring passages offered as context on each side", min_value=0, max_value=5, value=1, key="extraction_context",
            help="Reduced automatically when they do not fit the model's context window.",
        )
        force = st.checkbox(
            "Extract again where a result already exists", value=False, key="extraction_force",
            help="Without this, passages and categories that already have a usable result are skipped. Duplicate knowledge is never stored twice.",
        )
        relationships = st.checkbox(
            "Also find relationships between the extracted units", value=False, key="extraction_relationships",
            help="One extra model call per passage. A relationship the model can quote evidence for is stored with that evidence; "
            "anything else is stored as a suggestion for review. Needs at least two units to relate.",
        )

    kwargs = {
        "model_id": st.session_state.get("extraction_model") if ready else None,
        "limit": int(limit) or None,
        "force": force,
        "context_passages": int(context),
        "extract_relationships": relationships,
    }
    scope_args = (book_id, document_id, node_id, scope_label)
    plan_text = _plan(scope_args, categories, kwargs) if ready and categories and (scope_label == SCOPES[0] or node_id) else None

    if active is not None:
        st.info("An extraction of this book is running. Follow it below; you can start another when it has finished.")
    clicked = st.button(
        "Start extraction", type="primary", key="extraction_start",
        disabled=not ready or not categories or active is not None or plan_text is None,
    )
    if clicked:
        _start(scope_args, categories, kwargs)


def _choose_node(document_id, outline, scope_label: str) -> uuid.UUID | None:
    if scope_label == SCOPES[0]:
        return None
    types = CHAPTER_TYPES if scope_label == SCOPES[1] else SECTION_TYPES
    nodes = [(node, passages) for node, passages in outline if node.node_type in types]
    if not nodes:
        st.warning(f"This document has no {'chapters' if scope_label == SCOPES[1] else 'sections'} with passages.")
        return None
    by_id = {str(node.id): (node, passages) for node, passages in nodes}
    chosen = st.selectbox(
        "Chapter" if scope_label == SCOPES[1] else "Section",
        options=list(by_id),
        format_func=lambda value: f"{'\u00a0\u00a0' * by_id[value][0].level}{by_id[value][0].title} ({by_id[value][1]} passages)",
        key=f"extraction_node_{document_id}_{scope_label}",
    )
    return uuid.UUID(chosen)


def _scope(service: KnowledgeExtractionService, scope_args):
    book_id, document_id, node_id, scope_label = scope_args
    if scope_label == SCOPES[1]:
        return service.chapter_scope(node_id)
    if scope_label == SCOPES[2]:
        return service.section_scope(node_id)
    return service.book_scope(book_id, document_id)


def _plan(scope_args, categories: list[str], kwargs: dict) -> str | None:
    """Show what the run would do. Returns None (after showing why) when it cannot start."""
    try:
        with session_scope() as session:
            service = _service(session)
            plan = service.plan(
                _scope(service, scope_args), categories, model_id=kwargs["model_id"], limit=kwargs["limit"], force=kwargs["force"],
                extract_relationships=kwargs["extract_relationships"],
            )
    except ExtractionRunError as exc:
        st.error(str(exc))
        return None
    except SQLAlchemyError:
        logger.exception("Could not plan the extraction")
        st.error("Could not read the passages. Check the database connection.")
        return None
    columns = st.columns(3)
    columns[0].metric("Passages", plan.total_passages)
    columns[1].metric("Model calls", plan.model_calls)
    columns[2].metric("Already extracted", plan.skipped)
    if kwargs["extract_relationships"]:
        st.caption(f"Plus up to {plan.relationship_calls} calls to find relationships.")
    st.caption(f"Model: **{plan.model_label}**")
    if plan.total_passages == 0:
        st.warning("No passages found. Process the document's structure and passages first.")
        return None
    if plan.model_calls == 0 and plan.relationship_calls == 0:
        st.info("Everything in this scope has already been extracted. Tick the option to extract it again.")
        return None
    if plan.model_calls > 50:
        st.warning(f"This makes {plan.model_calls} model calls, which can take a while and cost money with hosted models.")
    return "ok"


def _start(scope_args, categories: list[str], kwargs: dict) -> None:
    try:
        with session_scope() as session:
            scope = _scope(_service(session), scope_args)
        run_id = get_job_runner().start(scope, categories, **kwargs)
    except NothingToExtractError as exc:
        st.info(str(exc))
        return
    except ExtractionRunError as exc:
        st.error(str(exc))
        return
    st.session_state[RUN_KEY] = str(run_id)
    st.rerun()


# -- following a run -------------------------------------------------------------------------------


def _render_runs(book_id, runs: list[RunStatus]) -> None:
    st.subheader("Progress")
    if not runs:
        st.caption("No extraction has been run for this book yet.")
        return
    shown = next((r for r in runs if str(r.run.id) == st.session_state.get(RUN_KEY)), None) or runs[0]
    # Poll while the shown run is active; a full rerun (see _render_status) turns polling off again.
    st.fragment(run_every=POLL_SECONDS if shown.is_active else None)(_render_status)(shown.run.id, was_active=shown.is_active)

    with st.expander("Earlier runs"):
        st.table(
            [
                {
                    "Started": format_timestamp(r.run.started_at or r.run.created_at),
                    "Scope": r.run.scope_type or "-",
                    "Model": r.model_label or "-",
                    "Status": STATUS_LABELS.get(r.status, r.status),
                    "Passages": f"{r.run.passages_processed}/{r.run.total_passages}",
                    "Units": r.run.units_extracted,
                    "Needs review": r.run.needs_review_count,
                    "Failed passages": r.failed_passages,
                }
                for r in runs
            ]
        )
    _render_knowledge(book_id)
    if st.button("Browse and review the extracted knowledge", key="extraction_go_knowledge"):
        goto("Knowledge", document_id=uuid.UUID(st.session_state[DOCUMENT_KEY]) if st.session_state.get(DOCUMENT_KEY) else None)


def _load_status(run_id: uuid.UUID) -> RunStatus:
    with session_scope() as session:
        return _service(session).get_run_status(run_id)


def _render_status(run_id: uuid.UUID, *, was_active: bool) -> None:
    try:
        status = _load_status(run_id)
    except (ExtractionRunError, SQLAlchemyError):
        logger.exception("Could not read extraction run %s", run_id)
        st.error("Could not read the extraction run.")
        return
    if was_active and not status.is_active:
        st.rerun()  # finished: refresh the whole page and stop polling

    run = status.run
    label = STATUS_LABELS.get(status.status, status.status)
    (st.error if status.status == "failed" else st.warning if status.status in ("needs_review", "cancelled") else st.info if status.is_active else st.success)(f"**{label}**")
    st.write(f"Model: **{status.model_label or 'not recorded'}** · prompt version {run.prompt_version or '-'} · {run.scope_type or 'book'} scope")
    st.progress(status.progress, text=f"{run.passages_processed} of {run.total_passages} passages")

    columns = st.columns(4)
    columns[0].metric("Knowledge units", run.units_extracted)
    columns[1].metric("Need review", run.needs_review_count)
    columns[2].metric("Failed passages", status.failed_passages)
    columns[3].metric("Duplicates skipped", run.duplicate_count)
    st.caption(
        f"{run.attempt_count} model attempts · {run.retry_count} retries · {run.validation_failure_count} validation failures · "
        f"{run.input_tokens or 0} input / {run.output_tokens or 0} output tokens"
        + (f" · estimated cost ${run.estimated_cost_usd:.4f}" if run.estimated_cost_usd is not None else "")
        + (f" · {status.duration_seconds:.0f}s" if status.duration_seconds is not None else "")
    )
    if run.error_message:
        st.error(f"{run.error_code}: {run.error_message}")
    if status.is_active and st.button("Cancel extraction", key=f"extraction_cancel_{run.id}"):
        with session_scope() as session:
            _service(session).cancel_run(run.id)
        st.rerun()
    _render_passages(run.id, status)
    if status.can_resume and st.button("Resume extraction", key=f"extraction_resume_{run.id}"):
        try:
            get_job_runner().resume(run.id)
        except ExtractionRunError as exc:
            st.error(str(exc))
        else:
            st.session_state[RUN_KEY] = str(run.id)
            st.rerun()


PASSAGE_ICONS = {"pending": "⏳", "processing": "🔄", "completed": "✅", "needs_review": "🟠", "failed": "🔴"}


def _render_passages(run_id: uuid.UUID, status: RunStatus) -> None:
    """Where every passage of the run stands, and what resuming would do."""
    counts = status.passage_counts
    if not counts:
        return
    with session_scope() as session:
        service = _service(session)
        views = service.get_passage_status(run_id)
        info = service.get_resume_info(run_id)
    summary = " · ".join(f"{PASSAGE_ICONS.get(k, '')} {v} {k.replace('_', ' ')}" for k, v in counts.items() if v)
    with st.expander(f"Passages: {summary}"):
        if info.can_resume:
            st.info(f"Resuming continues with {info.failed} failed, {info.pending} not reached and {info.processing} interrupted passage(s). {info.completed} finished passage(s) are not repeated.")
        elif info.reason:
            st.caption(info.reason)
        st.table(
            [
                {
                    "#": v.sequence + 1,
                    "Page": v.page_start or "-",
                    "Status": f"{PASSAGE_ICONS.get(v.status, '')} {v.status.replace('_', ' ')}",
                    "Categories": ", ".join(f"{k} {st_}" for k, st_ in sorted(v.categories.items())),
                    "Attempts": v.attempts,
                    "Units": v.units_extracted,
                    "Error": f"{v.error_code}: {v.error_message}" if v.error_code else "",
                }
                for v in views
            ]
        )


def _render_knowledge(book_id) -> None:
    with session_scope() as session:
        service = _service(session)
        total = service.runs.count_knowledge(book_id)
        latest = service.get_book_knowledge(book_id, limit=15)
    with st.expander(f"Extracted knowledge ({total} units)"):
        if not latest:
            st.caption("Nothing extracted yet.")
            return
        st.table(
            [
                {
                    "Type": unit.knowledge_type.value,
                    "Title": unit.content.title,
                    "Page": unit.page_start or "-",
                    "Validation": unit.validation_status.value,
                    "Uncertain": "yes" if unit.uncertainty.is_uncertain else "",
                }
                for unit in latest
            ]
        )
        if total > len(latest):
            st.caption(f"Showing the first {len(latest)} in reading order.")
