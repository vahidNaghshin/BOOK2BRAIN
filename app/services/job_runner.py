import logging
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.parsing.equation.equation_service import EquationService
from app.parsing.parser_factory import ParserFactory
from app.services.document_processing_service import DocumentProcessingService
from app.services.errors import ProcessingNotAllowedError
from app.services.pipeline_service import ProcessingPipeline
from app.services.storage_service import StorageBackend, get_storage
from database.connection import session_scope

logger = logging.getLogger(__name__)


class ProcessingJobRunner:
    """Runs the processing pipeline in a background worker thread.

    `submit` validates and queues the run in the caller's thread, so problems such as
    "already processing" surface immediately; the heavy work then happens on the worker.
    A single worker runs one document at a time, which keeps memory use predictable.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        session_factory: Callable[[], AbstractContextManager[Session]] = session_scope,
        storage: StorageBackend | None = None,
        parser_factory: ParserFactory | None = None,
        equation_service: EquationService | None = None,
        max_workers: int = 1,
    ) -> None:
        self.settings = settings or get_settings()
        self._session_factory = session_factory
        self._storage = storage or get_storage(self.settings)
        self._parser_factory = parser_factory
        self._equation_service = equation_service
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="processing")

    def _service(self, session: Session) -> DocumentProcessingService:
        return DocumentProcessingService(
            session, self._storage, self.settings, self._parser_factory, self._equation_service
        )

    def _pipeline(self, session: Session) -> ProcessingPipeline:
        return ProcessingPipeline(
            session, self._storage, self.settings, self._parser_factory, self._equation_service
        )

    def submit(self, document_id: uuid.UUID) -> uuid.UUID:
        """Queue the whole pipeline (or resume a failed run) for a document."""
        with self._session_factory() as session:
            run_id = self._pipeline(session).start(document_id)
        self._executor.submit(self._run, run_id)
        return run_id

    retry = submit  # a failed document is retried the same way it is started

    def submit_equations(self, document_id: uuid.UUID) -> None:
        """Queue equation extraction for an already processed document."""
        with self._session_factory() as session:
            service = self._service(session)
            info = service.get_info(document_id)
            run = info.latest_run
            if info.document_status != "completed" or run is None:
                raise ProcessingNotAllowedError(
                    "The document text must be processed before equations can be extracted."
                )
            if run.equation_status == "processing" and not info.equation_is_stale:
                raise ProcessingNotAllowedError("Equations are already being extracted for this document.")
        self._executor.submit(self._run_equations, document_id)

    def _run_equations(self, document_id: uuid.UUID) -> None:
        try:
            with self._session_factory() as session:
                self._service(session).retry_equations(document_id)
        except Exception:
            logger.exception("Background equation extraction for %s crashed", document_id)

    def _run(self, run_id: uuid.UUID) -> None:
        try:
            with self._session_factory() as session:
                self._pipeline(session).execute(run_id)
        except Exception:
            logger.exception("Background processing of run %s crashed", run_id)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
