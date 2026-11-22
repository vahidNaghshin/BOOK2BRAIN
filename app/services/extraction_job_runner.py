"""Runs knowledge extraction in a background worker thread, like the processing job runner.

`start` validates and records the run in the caller's thread, so a bad model, an empty scope or an
already running extraction is reported at once; the model calls then happen on the worker. A single
worker runs one extraction at a time.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager

from sqlalchemy.orm import Session

from app.knowledge.types import KnowledgeType
from app.repositories.passage_repository import ExtractionScope
from app.services.errors import ExtractionRunError
from app.services.knowledge_extraction_service import KnowledgeExtractionService, create_extraction_service
from database.connection import session_scope

logger = logging.getLogger(__name__)


class ExtractionJobRunner:
    def __init__(
        self,
        *,
        session_factory: Callable[[], AbstractContextManager[Session]] = session_scope,
        service_factory: Callable[[Session], KnowledgeExtractionService] = create_extraction_service,
        max_workers: int = 1,
    ) -> None:
        self._session_factory = session_factory
        self._service_factory = service_factory
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="extraction")

    def start(
        self,
        scope: ExtractionScope,
        categories: Iterable[KnowledgeType | str],
        *,
        model_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        force: bool = False,
        context_passages: int | None = None,
        extract_relationships: bool = False,
    ) -> uuid.UUID:
        with self._session_factory() as session:
            run_id = self._service_factory(session).create_run(
                scope, categories, model_id=model_id, limit=limit, offset=offset, force=force,
                context_passages=context_passages, extract_relationships=extract_relationships,
            )
        self._executor.submit(self._execute, run_id)
        return run_id

    def resume(self, run_id: uuid.UUID, *, takeover: bool = False) -> None:
        """Continue a failed, cancelled or interrupted run; finished passages are not repeated.

        Checked here, in the caller's thread, so "nothing to resume" or "still running" is reported at once.
        """
        with self._session_factory() as session:
            service = self._service_factory(session)
            info = service.get_resume_info(run_id)
            if not info.can_resume and not (takeover and info.status == "processing"):
                raise ExtractionRunError(info.reason or "The run cannot be resumed.")
        self._executor.submit(self._execute, run_id, takeover)

    def _execute(self, run_id: uuid.UUID, takeover: bool = False) -> None:
        try:
            with self._session_factory() as session:
                self._service_factory(session).execute_run(run_id, takeover=takeover)
        except Exception:
            logger.exception("Background extraction run %s crashed", run_id)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
