import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Protocol

from app.config import Settings, get_settings
from app.services.errors import StorageError

logger = logging.getLogger(__name__)

ORIGINAL_PREFIX = "original"
TEMPORARY_DIRNAME = "temporary"
_COPY_CHUNK = 1024 * 1024


def build_original_key(document_id: uuid.UUID, stored_filename: str) -> str:
    """Storage key for an original file: one directory per document UUID."""
    return f"{ORIGINAL_PREFIX}/{document_id}/{stored_filename}"


class StorageBackend(Protocol):
    """Where original files live. Keys are relative, '/'-separated, backend-agnostic."""

    def save(self, key: str, source: BinaryIO) -> int:
        """Store the stream under key without overwriting. Returns bytes written."""

    def open(self, key: str) -> BinaryIO: ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None:
        """Remove the object. Missing objects are not an error."""


class LocalStorage:
    """Filesystem storage rooted at a configurable directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def _resolve(self, key: str) -> Path:
        if not key or key.startswith("/") or "\\" in key or "\x00" in key:
            raise StorageError(f"Invalid storage key: {key!r}")
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise StorageError(f"Storage key escapes the storage root: {key!r}")
        return path

    def save(self, key: str, source: BinaryIO) -> int:
        destination = self._resolve(key)
        temp_dir = self.root / TEMPORARY_DIRNAME
        temp_path = temp_dir / f"{uuid.uuid4().hex}.part"
        created_dir = False
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise StorageError(f"Refusing to overwrite existing file: {key}")

            written = 0
            with temp_path.open("xb") as target:
                while chunk := source.read(_COPY_CHUNK):
                    target.write(chunk)
                    written += len(chunk)
                target.flush()
                os.fsync(target.fileno())

            created_dir = not destination.parent.exists()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp_path.rename(destination)
        except StorageError:
            self._discard(temp_path, destination.parent if created_dir else None)
            raise
        except OSError as exc:
            logger.error("Failed to store %s: %s", key, exc)
            self._discard(temp_path, destination.parent if created_dir else None)
            raise StorageError("The file could not be written to storage.") from exc
        except Exception:
            self._discard(temp_path, destination.parent if created_dir else None)
            raise
        logger.info("Stored %s (%d bytes)", key, written)
        return written

    def open(self, key: str) -> BinaryIO:
        path = self._resolve(key)
        try:
            return path.open("rb")
        except OSError as exc:
            raise StorageError(f"Could not open stored file: {key}") from exc

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        try:
            path.unlink(missing_ok=True)
            self._remove_empty_parents(path.parent)
        except OSError as exc:
            raise StorageError(f"Could not delete stored file: {key}") from exc
        logger.info("Deleted %s", key)

    def _remove_empty_parents(self, directory: Path) -> None:
        # Never remove the root or the top-level 'original' directory.
        while directory != self.root and directory.parent != self.root:
            try:
                directory.rmdir()
            except OSError:
                return
            directory = directory.parent

    def _discard(self, temp_path: Path, created_dir: Path | None) -> None:
        try:
            temp_path.unlink(missing_ok=True)
            if created_dir is not None:
                shutil.rmtree(created_dir, ignore_errors=True)
        except OSError as exc:
            logger.warning("Could not clean up after failed save: %s", exc)


def get_storage(settings: Settings | None = None) -> LocalStorage:
    return LocalStorage((settings or get_settings()).storage_root)
