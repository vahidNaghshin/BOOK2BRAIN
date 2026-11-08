import hashlib
import io
import logging
import shutil
import tempfile
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.parsing.equation.equation_models import (
    EquationBlock,
    EquationExtractionStatus,
)
from app.parsing.equation.equation_service import EquationOutcome, EquationService
from app.parsing.errors import ParserError
from app.parsing.models import (
    ContentBlock,
    ExtractionStatus,
    ParseStage,
    ParsingIssue,
    ParsingResult,
)
from app.parsing.parser_factory import ParserFactory
from app.parsing.text_normalizer import normalize_blocks
from app.repositories.processing_repository import (
    EquationRepository,
    ExtractedBlockRepository,
    ProcessingRepository,
)
from app.schemas import (
    EquationRead,
    ExtractedBlockRead,
    ProcessingErrorRead,
    ProcessingInfo,
    ProcessingRunRead,
    ProcessingSummary,
)
from app.services.errors import (
    DocumentNotFoundError,
    ProcessingNotAllowedError,
    ProcessingServiceError,
    StorageError,
)
from app.services.storage_service import TEMPORARY_DIRNAME, StorageBackend
from database.models import (
    Document,
    EquationStatus,
    ProcessingRun,
    ProcessingRunStatus,
    ProcessingStage,
    ProcessingStatus,
)
from database.models.base import utcnow

logger = logging.getLogger(__name__)

# (stage, current, total) reported to the caller while a run executes.
RunProgressCallback = Callable[[str, int, int], None]
_PROGRESS_WRITE_INTERVAL_SECONDS = 0.5
_MAX_ERROR_MESSAGE = 2000


@contextmanager
def local_copy(storage: StorageBackend, key: str, suffix: str, temp_root: Path) -> Iterator[Path]:
    """Copy a stored file to a temporary local path, so parsers work with any storage backend."""
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root, prefix="processing-") as directory:
        path = Path(directory) / f"source{suffix}"
        with storage.open(key) as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)
        yield path


class DocumentProcessingService:
    """Turns an uploaded document into stored, normalized, source-traceable text blocks.

    Status flow: uploaded -> pending -> processing -> completed | failed. While processing,
    the run's `stage` is parsing, ocr or normalizing. Results are persisted in one
    transaction, so a failed run never leaves partial blocks behind.
    """

    def __init__(
        self,
        session: Session,
        storage: StorageBackend,
        settings: Settings | None = None,
        parser_factory: ParserFactory | None = None,
        equation_service: EquationService | None = None,
    ) -> None:
        self.session = session
        self.storage = storage
        self.settings = settings or get_settings()
        self.parser_factory = parser_factory or ParserFactory(self.settings)
        self.equation_service = equation_service or EquationService(self.settings)
        self.runs = ProcessingRepository(session)
        self.blocks = ExtractedBlockRepository(session)
        self.equations = EquationRepository(session)
        self._last_progress_write = 0.0
        self._last_stage: str | None = None

    # -- queries -------------------------------------------------------------------

    def get_info(self, document_id: uuid.UUID) -> ProcessingInfo:
        document = self._load_document(document_id)
        run = self.runs.latest_run(document_id)
        active = run is not None and run.status in (
            ProcessingRunStatus.PENDING.value,
            ProcessingRunStatus.PROCESSING.value,
        )
        return ProcessingInfo(
            document_id=document_id,
            document_status=document.processing_status,
            latest_run=self._read_run(run) if run else None,
            errors=[ProcessingErrorRead.model_validate(e) for e in self.runs.list_errors(document_id)],
            run_count=self.runs.count_runs(document_id),
            is_stale=bool(active and self._is_stale(run)),
            equation_is_stale=bool(
                run is not None
                and run.equation_status == EquationStatus.PROCESSING.value
                and self._equations_stale(run)
            ),
        )

    def count_blocks(self, document_id: uuid.UUID, page_number: int | None = None) -> int:
        return self.blocks.count(document_id, page_number)

    def list_blocks(
        self,
        document_id: uuid.UUID,
        *,
        offset: int = 0,
        limit: int = 50,
        page_number: int | None = None,
    ) -> list[ExtractedBlockRead]:
        rows = self.blocks.list(document_id, offset=offset, limit=limit, page_number=page_number)
        return [ExtractedBlockRead.model_validate(row) for row in rows]

    # -- commands ------------------------------------------------------------------

    def process(
        self, document_id: uuid.UUID, on_progress: RunProgressCallback | None = None
    ) -> ProcessingRunRead:
        """Start and run processing synchronously."""
        return self.execute(self.start(document_id), on_progress)

    def start(self, document_id: uuid.UUID) -> uuid.UUID:
        """Validate that the document can be processed and queue a run (status: pending)."""
        document = self.runs.lock_document(document_id)
        if document is None:
            self.session.rollback()
            raise DocumentNotFoundError("The document no longer exists.")
        try:
            self._check_startable(document)
        except ProcessingServiceError:
            self.session.rollback()
            raise

        run = self.runs.add_run(ProcessingRun(document_id=document.id))
        document.processing_status = ProcessingStatus.PENDING.value
        self.session.commit()
        logger.info("Queued processing run %s for document %s", run.id, document.id)
        return run.id

    def execute(
        self, run_id: uuid.UUID, on_progress: RunProgressCallback | None = None
    ) -> ProcessingRunRead:
        """Run a queued run to completion. Document problems end in a failed run, not an exception."""
        run = self.runs.get_run(run_id)
        if run is None:
            raise ProcessingServiceError("The processing run does not exist.")
        if run.status != ProcessingRunStatus.PENDING.value:
            raise ProcessingNotAllowedError("This processing run has already been started.")
        document = self.session.get(Document, run.document_id)
        if document is None:
            raise DocumentNotFoundError("The document no longer exists.")
        document_id = document.id

        self._begin(run, document)
        self._last_stage = ProcessingStage.PARSING.value
        text_saved = False
        try:
            with local_copy(
                self.storage, document.file_path, f".{document.file_format}", self._temp_root()
            ) as path:
                result = self._parse(document, path, run_id, on_progress)
                if result.status is ExtractionStatus.FAILED:
                    self._record_failure(run_id, document_id, result.errors)
                else:
                    block_ids = self._record_success(run_id, document_id, result)
                    text_saved = True
                    # Equations come after the text is safely stored and never affect its status.
                    self._extract_equations(run_id, document, path, result, block_ids)
        except Exception as exc:
            if text_saved:
                logger.exception("Unexpected error after the text of run %s was saved", run_id)
            else:
                self._record_exception(run_id, document_id, exc)
        except BaseException as exc:  # e.g. the UI session ended: don't leave the run "processing"
            if not text_saved:
                self._record_exception(run_id, document_id, exc)
            raise
        return self._read_run(self._load_run(run_id))

    # -- internals: workflow -------------------------------------------------------

    def _check_startable(self, document: Document) -> None:
        status = document.processing_status
        if status == ProcessingStatus.COMPLETED.value:
            raise ProcessingNotAllowedError("This document has already been processed.")
        if status in (ProcessingStatus.PENDING.value, ProcessingStatus.PROCESSING.value):
            active = self.runs.latest_active_run(document.id)
            if active is not None and not self._is_stale(active):
                raise ProcessingNotAllowedError("This document is already being processed.")
            if active is not None:
                self._abandon(active, document)

    def _is_stale(self, run: ProcessingRun) -> bool:
        reference = run.started_at or run.created_at
        return utcnow() - reference > timedelta(minutes=self.settings.processing_stale_minutes)

    def _abandon(self, run: ProcessingRun, document: Document) -> None:
        """Fail a run that never finished (e.g. the app was restarted mid-run) so a retry can start."""
        message = (
            f"Processing did not finish within {self.settings.processing_stale_minutes} minutes "
            "and was treated as interrupted."
        )
        logger.warning("Abandoning stale processing run %s for document %s", run.id, document.id)
        run.status = ProcessingRunStatus.FAILED.value
        run.completed_at = utcnow()
        run.error_message = message
        self.runs.add_error(
            document_id=document.id,
            run_id=run.id,
            error_type="interrupted",
            message=message,
            details={"stage": run.stage},
        )

    def _begin(self, run: ProcessingRun, document: Document) -> None:
        run.status = ProcessingRunStatus.PROCESSING.value
        run.started_at = utcnow()
        run.stage = ProcessingStage.PARSING.value
        run.progress_current = None
        run.progress_total = None
        document.processing_status = ProcessingStatus.PROCESSING.value
        self.session.commit()

    def _parse(
        self, document: Document, path: Path, run_id: uuid.UUID, on_progress: RunProgressCallback | None
    ) -> ParsingResult:
        """Parse and normalize the local copy of a document (nothing is saved here)."""
        parser = self.parser_factory.get_parser(document.file_format)
        result = parser.parse(
            path,
            document_id=document.id,
            original_filename=document.original_filename,
            on_progress=self._progress_callback(run_id, on_progress),
        )
        if result.status is ExtractionStatus.FAILED:
            return result

        self._write_progress(run_id, ProcessingStage.NORMALIZING.value, 0, 1, on_progress)
        blocks, warnings = normalize_blocks(result.blocks)
        result = result.model_copy(
            update={"blocks": blocks, "warnings": [*result.warnings, *warnings]}
        )
        self._write_progress(run_id, ProcessingStage.NORMALIZING.value, 1, 1, on_progress)
        return self._require_meaningful_content(result)

    def _require_meaningful_content(self, result: ParsingResult) -> ParsingResult:
        minimum = self.settings.min_content_characters
        if result.character_count >= minimum:
            return result
        issue = ParsingIssue(
            code="no_meaningful_content",
            message=(
                f"Only {result.character_count} characters of text were extracted (at least "
                f"{minimum} are required). The document may be empty, image-only or unreadable."
            ),
            details={"character_count": result.character_count, "minimum": minimum},
        )
        return result.model_copy(
            update={"status": ExtractionStatus.FAILED, "errors": [*result.errors, issue]}
        )

    def _temp_root(self) -> Path:
        storage_root = getattr(self.storage, "root", None)
        return Path(storage_root if storage_root is not None else self.settings.storage_root) / TEMPORARY_DIRNAME

    # -- internals: progress -------------------------------------------------------

    def _progress_callback(
        self, run_id: uuid.UUID, user_callback: RunProgressCallback | None
    ) -> Callable[[ParseStage, int, int], None]:
        def callback(stage: ParseStage, current: int, total: int) -> None:
            self._write_progress(run_id, stage.value, current, total, user_callback)

        return callback

    def _write_progress(
        self,
        run_id: uuid.UUID,
        stage: str,
        current: int,
        total: int,
        user_callback: RunProgressCallback | None,
    ) -> None:
        """Persist progress, throttled, so other sessions can show live status."""
        now = time.monotonic()
        if (
            stage != self._last_stage
            or current >= total
            or now - self._last_progress_write >= _PROGRESS_WRITE_INTERVAL_SECONDS
        ):
            self.runs.update_progress(run_id, stage=stage, current=current, total=total)
            self.session.commit()
            self._last_progress_write = now
            self._last_stage = stage
        if user_callback is not None:
            user_callback(stage, current, total)

    # -- internals: persistence ----------------------------------------------------

    def _record_success(
        self, run_id: uuid.UUID, document_id: uuid.UUID, result: ParsingResult
    ) -> dict[int, uuid.UUID]:
        """Replace the document's blocks and complete the run in a single transaction.

        Returns the new block ids by sequence number, so equations can link to their blocks.
        """
        block_ids = {block.sequence_number: uuid.uuid4() for block in result.blocks}
        self.blocks.replace_for_document(
            document_id, [self._block_row(b, block_ids[b.sequence_number]) for b in result.blocks]
        )
        summary = ProcessingSummary(
            extraction_method=result.extraction_method,
            status=result.status,
            extracted_at=result.extracted_at,
            page_count=result.page_count,
            character_count=result.character_count,
            block_count=len(result.blocks),
            ocr=result.ocr,
            warnings=result.warnings,
            metadata=result.metadata,
        )
        run = self._load_run(run_id)
        run.status = ProcessingRunStatus.COMPLETED.value
        run.completed_at = utcnow()
        run.error_message = None
        # A finished run has no active stage (failed runs keep theirs: it says where they failed).
        run.stage = None
        run.progress_current = None
        run.progress_total = None
        run.result_summary = summary.model_dump(mode="json")
        self._load_document(document_id).processing_status = ProcessingStatus.COMPLETED.value
        self.session.commit()
        logger.info(
            "Processing run %s completed: %d blocks, %d characters, status=%s",
            run_id, len(result.blocks), result.character_count, result.status.value,
        )
        return block_ids

    def _record_failure(
        self, run_id: uuid.UUID, document_id: uuid.UUID, issues: list[ParsingIssue]
    ) -> None:
        """Mark the run and document failed and keep the diagnostics. Nothing else is saved."""
        self.session.rollback()
        run = self._load_run(run_id)
        run.status = ProcessingRunStatus.FAILED.value
        run.completed_at = utcnow()
        run.error_message = "; ".join(issue.message for issue in issues)[:_MAX_ERROR_MESSAGE]
        for issue in issues:
            details = dict(issue.details)
            details["stage"] = run.stage
            if issue.page_number is not None:
                details["page_number"] = issue.page_number
            self.runs.add_error(
                document_id=document_id,
                run_id=run_id,
                error_type=issue.code,
                message=issue.message,
                details=details,
            )
        self._load_document(document_id).processing_status = ProcessingStatus.FAILED.value
        self.session.commit()
        logger.warning("Processing run %s failed: %s", run_id, run.error_message)

    def _record_exception(self, run_id: uuid.UUID, document_id: uuid.UUID, exc: BaseException) -> None:
        if isinstance(exc, ParserError):
            issues = [exc.to_issue()]
        else:
            logger.exception("Processing run %s failed unexpectedly", run_id)
            issues = [self._issue_for_exception(exc)]
        try:
            self._record_failure(run_id, document_id, issues)
        except Exception:
            logger.exception("Could not record the failure of processing run %s", run_id)
            raise

    @staticmethod
    def _issue_for_exception(exc: BaseException) -> ParsingIssue:
        details = {"exception": type(exc).__name__, "traceback": traceback.format_exc()}
        if isinstance(exc, StorageError):
            return ParsingIssue(
                code="storage_error",
                message=f"The stored file could not be read: {exc}",
                details=details,
            )
        return ParsingIssue(
            code="unexpected_error",
            message=f"Processing failed unexpectedly ({type(exc).__name__}). See the logs for details.",
            details=details,
        )

    @staticmethod
    def _block_row(block: ContentBlock, block_id: uuid.UUID) -> dict[str, Any]:
        return {
            "id": block_id,
            "sequence_number": block.sequence_number,
            "block_type": block.block_type.value,
            "text": block.text,
            "raw_text": block.raw_text,
            "page_number": block.page_number,
            "section_title": block.section_title,
            "source_location": block.source_location,
            "block_metadata": block.metadata,
        }

    # -- equations ------------------------------------------------------------------

    def retry_equations(
        self, document_id: uuid.UUID, on_progress: RunProgressCallback | None = None
    ) -> ProcessingRunRead:
        """(Re)run equation extraction for a processed document without touching its text.

        Safe to repeat: the stored equations are replaced in one transaction, never added to.
        """
        document = self.runs.lock_document(document_id)
        if document is None:
            self.session.rollback()
            raise DocumentNotFoundError("The document no longer exists.")
        run = self.runs.latest_run(document_id)
        if document.processing_status != ProcessingStatus.COMPLETED.value or run is None:
            self.session.rollback()
            raise ProcessingNotAllowedError("The document text must be processed before equations can be extracted.")
        if run.equation_status == EquationStatus.PROCESSING.value and not self._equations_stale(run):
            self.session.rollback()
            raise ProcessingNotAllowedError("Equations are already being extracted for this document.")
        run.equation_status = EquationStatus.PROCESSING.value
        run.equation_summary = {"progress": {"stage": "starting"}}
        self.session.commit()

        run_id = run.id
        try:
            with local_copy(self.storage, document.file_path, f".{document.file_format}", self._temp_root()) as path:
                parsing = self._parse(document, path, run_id, on_progress)
                if parsing.status is ExtractionStatus.FAILED:
                    raise RuntimeError("The document could not be parsed again: " + "; ".join(e.message for e in parsing.errors))
                block_ids = self.blocks.ids_by_sequence(document.id)
                if len(block_ids) != len(parsing.blocks):
                    logger.warning("Block count changed since the text was saved; equations will not be linked to blocks")
                    block_ids = {}
                self._extract_equations(run_id, document, path, parsing, block_ids)
        except Exception as exc:
            self._record_equation_failure(run_id, document.id, exc)
        return self._read_run(self._load_run(run_id))

    def _equations_stale(self, run: ProcessingRun) -> bool:
        reference = run.completed_at or run.created_at
        return utcnow() - reference > timedelta(minutes=self.settings.processing_stale_minutes)

    def _extract_equations(
        self,
        run_id: uuid.UUID,
        document: Document,
        path: Path,
        parsing: ParsingResult,
        block_ids: dict[int, uuid.UUID],
    ) -> None:
        """Run equation extraction and store its outcome. Problems are recorded, never raised."""
        if not self.settings.equation_extraction_enabled:
            return
        document_id = document.id
        self._write_equation_state(run_id, EquationStatus.PROCESSING, {"progress": {"stage": "starting"}})
        stored_keys: list[str] = []
        try:
            outcome = self.equation_service.extract(
                path=path,
                file_format=document.file_format,
                document_id=document_id,
                parsing=parsing,
                block_ids=block_ids,
                on_progress=self._equation_progress(run_id),
            )
            stored_keys = self._store_crops(document_id, outcome)
            old_keys = self.equations.crop_references(document_id)
            self._record_equations(run_id, document_id, outcome)
            self._delete_crops([key for key in old_keys if key not in stored_keys])
        except BaseException as exc:
            self._delete_crops(stored_keys)
            self._record_equation_failure(run_id, document_id, exc)
            if not isinstance(exc, Exception):
                raise

    def _equation_progress(self, run_id: uuid.UUID) -> Callable[[str, int, int], None]:
        state = {"last": 0.0}

        def callback(stage: str, current: int, total: int) -> None:
            now = time.monotonic()
            if current < total and now - state["last"] < _PROGRESS_WRITE_INTERVAL_SECONDS:
                return
            state["last"] = now
            self._write_equation_state(
                run_id, EquationStatus.PROCESSING, {"progress": {"stage": stage, "current": current, "total": total}}
            )

        return callback

    def _write_equation_state(self, run_id: uuid.UUID, status: EquationStatus, summary: dict[str, Any]) -> None:
        self.session.execute(
            update(ProcessingRun)
            .where(ProcessingRun.id == run_id)
            .values(equation_status=status.value, equation_summary=summary)
        )
        self.session.commit()

    def _store_crops(self, document_id: uuid.UUID, outcome: EquationOutcome) -> list[str]:
        """Save region images through the storage abstraction. Keys are unique per extraction."""
        token = uuid.uuid4().hex[:8]
        keys: list[str] = []
        try:
            for equation in outcome.result.equations:
                data = outcome.crops.get(equation.id)
                if data is None:
                    continue
                key = f"equations/{document_id}/{token}/{equation.sequence_number:05d}.png"
                self.storage.save(key, io.BytesIO(data))
                keys.append(key)
                equation.crop_reference = key
        except Exception:
            self._delete_crops(keys)
            raise
        return keys

    def _delete_crops(self, keys: list[str]) -> None:
        for key in keys:
            try:
                self.storage.delete(key)
            except StorageError:
                logger.exception("Could not delete equation crop %s", key)

    def _record_equations(self, run_id: uuid.UUID, document_id: uuid.UUID, outcome: EquationOutcome) -> None:
        """Replace the document's equations and record the outcome in one transaction."""
        result = outcome.result
        self.equations.replace_for_document(document_id, [self._equation_row(e) for e in result.equations])
        run = self._load_run(run_id)
        run.equation_status = result.status.value
        run.equation_summary = result.summary()
        if result.status in (EquationExtractionStatus.FAILED, EquationExtractionStatus.UNAVAILABLE):
            for issue in result.issues or []:
                self.runs.add_error(
                    document_id=document_id,
                    run_id=run_id,
                    error_type=f"equation_{issue.code}",
                    message=issue.message,
                    details={**issue.details, "stage": "equations"},
                )
        self.session.commit()

    def _record_equation_failure(self, run_id: uuid.UUID, document_id: uuid.UUID, exc: BaseException) -> None:
        """Record that equation extraction failed. The run and document status are untouched."""
        logger.exception("Equation extraction failed for run %s", run_id)
        try:
            self.session.rollback()
            run = self._load_run(run_id)
            message = f"Equation extraction failed ({type(exc).__name__}): {str(exc)[:300]}"
            run.equation_status = EquationStatus.FAILED.value
            run.equation_summary = {"status": "failed", "issues": [{"code": "equation_extraction_failed", "message": message}]}
            self.runs.add_error(
                document_id=document_id,
                run_id=run_id,
                error_type="equation_extraction_failed",
                message=message,
                details={"stage": "equations", "exception": type(exc).__name__, "traceback": traceback.format_exc()},
            )
            self.session.commit()
        except Exception:
            logger.exception("Could not record the equation failure of run %s", run_id)

    @staticmethod
    def _equation_row(equation: EquationBlock) -> dict[str, Any]:
        metadata = dict(equation.metadata)
        identity = metadata.pop("identity_hash", None) or hashlib.sha256(str(equation.id).encode()).hexdigest()
        return {
            "id": equation.id,
            "extracted_block_id": equation.extracted_block_id,
            "sequence_number": equation.sequence_number,
            "identity_hash": identity,
            "latex": equation.latex,
            "original_latex": equation.original_latex,
            "plain_text": equation.plain_text,
            "equation_type": equation.equation_type.value,
            "equation_category": equation.equation_category,
            "equation_number": equation.equation_number,
            "page_number": equation.page_number,
            "section_title": (equation.section_title or None) and equation.section_title[:1000],
            "surrounding_text": equation.surrounding_text,
            "source_location": equation.source_location,
            "bounding_box": equation.bounding_box,
            "crop_reference": equation.crop_reference,
            "extraction_method": equation.extraction_method.value,
            "confidence": equation.confidence,
            "validation_status": equation.validation_status.value,
            "validation_warnings": equation.validation_warnings or None,
            "equation_metadata": metadata or None,
        }

    def count_equations(
        self,
        document_id: uuid.UUID | None = None,
        *,
        page_number: int | None = None,
        equation_type: str | None = None,
        validation_status: str | None = None,
    ) -> int:
        return self.equations.count(
            document_id, page_number=page_number, equation_type=equation_type, validation_status=validation_status
        )

    def list_equations(
        self,
        document_id: uuid.UUID | None = None,
        *,
        page_number: int | None = None,
        equation_type: str | None = None,
        validation_status: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> list[EquationRead]:
        rows = self.equations.list(
            document_id,
            page_number=page_number,
            equation_type=equation_type,
            validation_status=validation_status,
            offset=offset,
            limit=limit,
        )
        return [EquationRead.model_validate(row) for row in rows]

    def read_crop(self, key: str) -> bytes | None:
        """The stored image of the region an equation was recognised from, if there is one."""
        try:
            with self.storage.open(key) as handle:
                return handle.read()
        except StorageError:
            return None

    # -- internals: loading --------------------------------------------------------

    def _load_document(self, document_id: uuid.UUID) -> Document:
        stmt = select(Document).where(Document.id == document_id).execution_options(populate_existing=True)
        document = self.session.execute(stmt).scalar_one_or_none()
        if document is None:
            raise DocumentNotFoundError("The document does not exist.")
        return document

    def _load_run(self, run_id: uuid.UUID) -> ProcessingRun:
        stmt = select(ProcessingRun).where(ProcessingRun.id == run_id).execution_options(populate_existing=True)
        return self.session.execute(stmt).scalar_one()

    @staticmethod
    def _read_run(run: ProcessingRun) -> ProcessingRunRead:
        read = ProcessingRunRead.model_validate(run)
        if run.result_summary:
            read.summary = ProcessingSummary.model_validate(run.result_summary)
        return read
