"""The processing pipeline: one entry point from an uploaded document to searchable passages.

    document_registration -> parsing -> ocr (scanned pages only) -> normalizing -> text_extraction
    -> equation_extraction -> structural_detection -> paragraph_identification
    -> passage_segmentation -> persistence -> validation -> completed

The text stages are `DocumentProcessingService` (Day 3), the structure stages are
`StructureService` (Day 4), and the last stage is `IntegrityValidator`. This class only
coordinates them: it keeps the run's `stage` and per-stage results up to date, records which
stage failed and why, and marks the run and document **completed only after validation passes**.

A failure in a structure stage keeps the extracted text. Retrying then *resumes the same run*
from the stage that failed, without parsing the document again; a failure while parsing starts
a new run (the old one stays as history).
"""

from __future__ import annotations

import logging
import traceback
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.parsing.equation.equation_service import EquationService
from app.parsing.parser_factory import ParserFactory
from app.repositories.processing_repository import ProcessingRepository
from app.schemas import ProcessingInfo, ProcessingRunRead
from app.services.document_processing_service import DocumentProcessingService, RunProgressCallback
from app.services.errors import (
    DocumentNotFoundError,
    ProcessingNotAllowedError,
    ProcessingServiceError,
    StructureServiceError,
)
from app.services.integrity_service import IntegrityValidator
from app.services.storage_service import StorageBackend
from app.services.structure_service import StructureService
from app.structure.models import Severity
from database.models import (
    RESUMABLE_STAGES,
    Document,
    ExtractedBlock,
    ProcessingRun,
    ProcessingRunStatus,
    ProcessingStage,
    ProcessingStatus,
    StructureAnalysis,
)
from database.models.base import utcnow

logger = logging.getLogger(__name__)

# What a user is told when a stage fails (the technical detail is kept in the processing error).
STAGE_FAILURE_MESSAGES: dict[str, str] = {
    ProcessingStage.DOCUMENT_REGISTRATION.value: "Document registration failed.",
    ProcessingStage.PARSING.value: "Document parsing failed.",
    ProcessingStage.OCR.value: "Text recognition (OCR) on scanned pages failed.",
    ProcessingStage.NORMALIZING.value: "Text extraction returned no usable content.",
    ProcessingStage.TEXT_EXTRACTION.value: "Text extraction returned no usable content.",
    ProcessingStage.EQUATION_EXTRACTION.value: "Equation extraction failed.",
    ProcessingStage.STRUCTURAL_DETECTION.value: "Structural detection failed.",
    ProcessingStage.PARAGRAPH_IDENTIFICATION.value: "Paragraph identification failed.",
    ProcessingStage.PASSAGE_SEGMENTATION.value: "Passage segmentation failed.",
    ProcessingStage.PERSISTENCE.value: "Database persistence failed.",
    ProcessingStage.VALIDATION.value: "Data integrity validation failed.",
}
_MAX_ERROR_MESSAGE = 2000


class PipelineStageError(Exception):
    """A stage failed in a way the pipeline understands. The message is safe to show to users."""

    def __init__(
        self, stage: ProcessingStage, message: str, *, error_type: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.error_type = error_type
        self.details = details or {}


def stage_failure_message(stage: str | None) -> str:
    return STAGE_FAILURE_MESSAGES.get(stage or "", "Processing failed.")


class ProcessingPipeline:
    def __init__(
        self,
        session: Session,
        storage: StorageBackend,
        settings: Settings | None = None,
        parser_factory: ParserFactory | None = None,
        equation_service: EquationService | None = None,
        structure_service: StructureService | None = None,
    ) -> None:
        self.session = session
        self.storage = storage
        self.settings = settings or get_settings()
        self.text = DocumentProcessingService(session, storage, self.settings, parser_factory, equation_service)
        self.structure = structure_service or StructureService(session, self.settings)
        self.integrity = IntegrityValidator(session, storage)
        self.runs = ProcessingRepository(session)
        self._stage = ProcessingStage.DOCUMENT_REGISTRATION
        self._on_progress: RunProgressCallback | None = None

    # -- queries -------------------------------------------------------------------------

    def get_info(self, document_id: uuid.UUID) -> ProcessingInfo:
        return self.text.get_info(document_id)

    # -- commands ------------------------------------------------------------------------

    def process_document(
        self, document_id: uuid.UUID, on_progress: RunProgressCallback | None = None
    ) -> ProcessingRunRead:
        """Start (or resume) processing and run it to the end, synchronously."""
        return self.execute(self.start(document_id), on_progress)

    def retry(self, document_id: uuid.UUID) -> uuid.UUID:
        """Queue another attempt for a failed document (see `start`)."""
        return self.start(document_id)

    def start(self, document_id: uuid.UUID) -> uuid.UUID:
        """Queue processing and return the run id (status: pending).

        A failed run whose text is already saved is *resumed* (same run, `retry_count` + 1) so the
        document is not parsed again. Any other start creates a new run.
        """
        document = self.runs.lock_document(document_id)
        if document is None:
            self.session.rollback()
            raise DocumentNotFoundError("The document no longer exists.")

        if document.processing_status in (ProcessingStatus.PENDING.value, ProcessingStatus.PROCESSING.value):
            active = self.runs.latest_active_run(document_id)
            if active is not None and self.text._is_stale(active):
                self.text._abandon(active, document)
                document.processing_status = ProcessingStatus.FAILED.value
                self.session.flush()

        run = self.runs.latest_run(document_id)
        if document.processing_status == ProcessingStatus.FAILED.value and self._resumable(run, document_id):
            assert run is not None
            run.status = ProcessingRunStatus.PENDING.value
            run.retry_count = (run.retry_count or 0) + 1
            run.error_message = None
            run.completed_at = None
            document.processing_status = ProcessingStatus.PENDING.value
            self.session.commit()
            logger.info(
                "Resuming run %s of document %s from stage %s (attempt %d)",
                run.id, document_id, run.stage, run.retry_count,
            )
            return run.id

        prior_runs = self.runs.count_runs(document_id)
        run_id = self.text.start(document_id)
        if prior_runs:
            self.runs.set_retry_count(run_id, prior_runs)
            self.session.commit()
        return run_id

    def execute(self, run_id: uuid.UUID, on_progress: RunProgressCallback | None = None) -> ProcessingRunRead:
        """Run a queued run. Problems with the document end in a failed run, not an exception."""
        run = self.runs.get_run(run_id)
        if run is None:
            raise ProcessingServiceError("The processing run does not exist.")
        if run.status != ProcessingRunStatus.PENDING.value:
            raise ProcessingNotAllowedError("This processing run has already been started.")
        document = self.session.get(Document, run.document_id)
        if document is None:
            raise DocumentNotFoundError("The document no longer exists.")
        document_id = document.id
        resuming = run.text_completed_at is not None
        failed_stage = run.stage  # where a resumed run stopped last time
        self._on_progress = on_progress

        self._begin(run, document, resuming)
        try:
            self._register(run_id, document)
            if not resuming:
                outcome = self.text.execute(run_id, on_progress, finalize=False)
                if outcome.status == ProcessingRunStatus.FAILED.value:
                    return outcome  # the text stage recorded its own failure, with its stage
                self._stage = ProcessingStage.TEXT_EXTRACTION
                self._record_text_results(run_id, outcome)
            self._run_structure(document_id, run_id, resuming and failed_stage == ProcessingStage.VALIDATION.value)
            self._validate(document_id, run_id)
            self._complete(run_id, document_id)
        except PipelineStageError as exc:
            self._fail(run_id, document_id, exc.stage, exc.error_type, str(exc), exc.details, None)
        except StructureServiceError as exc:
            self._fail(run_id, document_id, self._stage, "structure_failed", None, {"reason": str(exc)}, exc)
        except Exception as exc:
            self._fail(run_id, document_id, self._stage, "unexpected_error", None, {}, exc)
        except BaseException as exc:  # e.g. the UI session ended: don't leave the run "processing"
            self._fail(run_id, document_id, self._stage, "interrupted", None, {}, exc)
            raise
        return self.text._read_run(self.text._load_run(run_id))

    # -- stages --------------------------------------------------------------------------

    def _begin(self, run: ProcessingRun, document: Document, resuming: bool) -> None:
        # A new run stays "pending" through registration: the text stage marks it "processing"
        # itself when it starts. A resumed run is already past that, so it starts processing now.
        if resuming:
            run.status = ProcessingRunStatus.PROCESSING.value
            run.started_at = run.started_at or utcnow()
            run.progress_current = None
            run.progress_total = None
            document.processing_status = ProcessingStatus.PROCESSING.value
        self.session.commit()
        self._enter(run.id, ProcessingStage.DOCUMENT_REGISTRATION)

    def _enter(self, run_id: uuid.UUID, stage: ProcessingStage) -> None:
        """Record that a stage has started (visible to other sessions at once).

        Entering a stage means the one before it finished, so a stage still marked running is
        closed as completed.
        """
        self._stage = stage
        run = self.text._load_run(run_id)
        run.stage = stage.value
        now = utcnow().isoformat()
        results = {
            name: {**entry, "status": "completed", "finished_at": now} if entry.get("status") == "running" else entry
            for name, entry in (run.stage_results or {}).items()
        }
        results[stage.value] = {"status": "running", "started_at": now}
        run.stage_results = results
        self.session.commit()
        if self._on_progress is not None:
            self._on_progress(stage.value, 0, 1)

    def _finish_stage(self, run_id: uuid.UUID, stage: ProcessingStage, status: str = "completed", **info: Any) -> None:
        run = self.text._load_run(run_id)
        results = dict(run.stage_results or {})
        entry = dict(results.get(stage.value) or {})
        entry.update({"status": status, "finished_at": utcnow().isoformat(), **info})
        results[stage.value] = entry
        run.stage_results = results
        self.session.commit()

    def _register(self, run_id: uuid.UUID, document: Document) -> None:
        """The document must exist with a stored file before anything else runs."""
        if not self.storage.exists(document.file_path):
            raise PipelineStageError(
                ProcessingStage.DOCUMENT_REGISTRATION,
                "The original file is missing from storage. Upload the book again.",
                error_type="source_file_missing",
                details={"file_path": document.file_path},
            )
        self._finish_stage(
            run_id,
            ProcessingStage.DOCUMENT_REGISTRATION,
            document_id=str(document.id),
            book_id=str(document.book_id),
            file_format=document.file_format,
            sha256=document.sha256_hash,
        )

    def _record_text_results(self, run_id: uuid.UUID, outcome: ProcessingRunRead) -> None:
        summary = outcome.summary
        if summary is None:
            return
        self._finish_stage(
            run_id,
            ProcessingStage.PARSING,
            extraction_method=summary.extraction_method,
            extraction_status=summary.status.value,
            page_count=summary.page_count,
        )
        if summary.ocr is not None:
            self._finish_stage(
                run_id,
                ProcessingStage.OCR,
                status="partial" if summary.ocr.pages_failed else "completed",
                pages_processed=len(summary.ocr.pages_processed),
                pages_failed=summary.ocr.pages_failed,
            )
        self._finish_stage(
            run_id,
            ProcessingStage.TEXT_EXTRACTION,
            block_count=summary.block_count,
            character_count=summary.character_count,
            warnings=len(summary.warnings),
        )
        if self.settings.equation_extraction_enabled:
            # Equations have their own status; they never decide whether the run succeeds.
            self._finish_stage(run_id, ProcessingStage.EQUATION_EXTRACTION, equation_status=outcome.equation_status)

    def _run_structure(self, document_id: uuid.UUID, run_id: uuid.UUID, validation_only: bool) -> None:
        if validation_only and self._structure_saved(document_id):
            return  # only validation failed last time: the saved structure is reused as it is
        if not self._blocks_saved(document_id):
            raise PipelineStageError(
                ProcessingStage.STRUCTURAL_DETECTION,
                "No extracted text is stored for this document.",
                error_type="no_extracted_text",
            )

        analysis = self.structure.detect_structure(
            document_id, on_stage=lambda name: self._enter(run_id, ProcessingStage(name)), require_completed=False
        )
        summary, passages = analysis.structure_summary or {}, analysis.passage_summary or {}
        self._finish_stage(run_id, ProcessingStage.PERSISTENCE)
        self._finish_stage(
            run_id, ProcessingStage.STRUCTURAL_DETECTION, node_count=summary.get("node_count"), node_types=summary.get("node_types")
        )
        self._finish_stage(run_id, ProcessingStage.PARAGRAPH_IDENTIFICATION, paragraph_count=summary.get("paragraph_count"))
        self._finish_stage(
            run_id, ProcessingStage.PASSAGE_SEGMENTATION, passage_count=passages.get("passage_count"), total_tokens=passages.get("total_tokens")
        )

    def _validate(self, document_id: uuid.UUID, run_id: uuid.UUID) -> None:
        self._enter(run_id, ProcessingStage.VALIDATION)
        issues = self.integrity.validate(document_id, run_id)
        errors = [i for i in issues if i.severity is Severity.ERROR]
        report = {
            "errors": len(errors),
            "warnings": sum(1 for i in issues if i.severity is Severity.WARNING),
            "issues": [i.to_dict() for i in issues if i.severity is not Severity.INFO],
        }
        if errors:
            self._finish_stage(run_id, ProcessingStage.VALIDATION, status="failed", **report)
            raise PipelineStageError(
                ProcessingStage.VALIDATION,
                "Data integrity validation failed: " + "; ".join(i.message for i in errors[:3]),
                error_type="integrity_validation_failed",
                details={"issues": [i.to_dict() for i in errors]},
            )
        self._finish_stage(run_id, ProcessingStage.VALIDATION, **report)

    def _complete(self, run_id: uuid.UUID, document_id: uuid.UUID) -> None:
        """The only place a run becomes completed: after every stage, including validation, passed."""
        run = self.text._load_run(run_id)
        run.status = ProcessingRunStatus.COMPLETED.value
        run.stage = ProcessingStage.COMPLETED.value
        run.completed_at = utcnow()
        run.error_message = None
        run.progress_current = None
        run.progress_total = None
        results = dict(run.stage_results or {})
        results[ProcessingStage.COMPLETED.value] = {"status": "completed", "finished_at": utcnow().isoformat()}
        run.stage_results = results
        self.text._load_document(document_id).processing_status = ProcessingStatus.COMPLETED.value
        self.session.commit()
        logger.info("Processing run %s of document %s completed", run_id, document_id)

    # -- failure ---------------------------------------------------------------------------

    def _fail(
        self,
        run_id: uuid.UUID,
        document_id: uuid.UUID,
        stage: ProcessingStage,
        error_type: str,
        message: str | None,
        details: dict[str, Any],
        exc: BaseException | None,
    ) -> None:
        """Mark the run and document failed at `stage`, keeping everything already saved."""
        if exc is not None:
            logger.exception("Processing run %s failed in stage %s", run_id, stage.value)
        else:
            logger.warning("Processing run %s failed in stage %s: %s", run_id, stage.value, message)
        user_message = (message or stage_failure_message(stage.value))[:_MAX_ERROR_MESSAGE]
        try:
            self.session.rollback()
            run = self.text._load_run(run_id)
            run.status = ProcessingRunStatus.FAILED.value
            run.stage = stage.value
            run.completed_at = utcnow()
            run.error_message = user_message
            results = dict(run.stage_results or {})
            entry = dict(results.get(stage.value) or {})
            entry.update({"status": "failed", "finished_at": utcnow().isoformat()})
            results[stage.value] = entry
            run.stage_results = results
            detail = {**details, "stage": stage.value}
            if exc is not None:
                detail["exception"] = type(exc).__name__
                detail["traceback"] = "".join(traceback.format_exception(exc))[-6000:]
            self.runs.add_error(
                document_id=document_id,
                run_id=run_id,
                error_type=error_type,
                message=user_message,
                details=detail,
                stage=stage.value,
                retry_attempt=run.retry_count or 0,
            )
            self.text._load_document(document_id).processing_status = ProcessingStatus.FAILED.value
            self.session.commit()
        except Exception:
            self.session.rollback()
            logger.exception("Could not record the failure of processing run %s", run_id)
            raise

    # -- helpers ---------------------------------------------------------------------------

    def _resumable(self, run: ProcessingRun | None, document_id: uuid.UUID) -> bool:
        """A failed run can resume when its text is saved and it failed in a later stage."""
        return bool(
            run is not None
            and run.status == ProcessingRunStatus.FAILED.value
            and run.text_completed_at is not None
            and run.stage in {stage.value for stage in RESUMABLE_STAGES}
            and self._blocks_saved(document_id)
        )

    def _blocks_saved(self, document_id: uuid.UUID) -> bool:
        stmt = select(func.count()).select_from(ExtractedBlock).where(ExtractedBlock.document_id == document_id)
        return bool(self.session.scalar(stmt))

    def _structure_saved(self, document_id: uuid.UUID) -> bool:
        analysis = self.session.scalar(select(StructureAnalysis).where(StructureAnalysis.document_id == document_id))
        return bool(analysis and analysis.structure_status == "completed" and analysis.passage_status == "completed")
