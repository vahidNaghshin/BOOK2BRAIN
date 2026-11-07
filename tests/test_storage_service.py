import io
from pathlib import Path

import pytest

from app.services.errors import StorageError
from app.services.storage_service import LocalStorage, build_original_key


@pytest.fixture
def local_storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "root")


def files_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_save_writes_file_and_returns_size(local_storage: LocalStorage) -> None:
    written = local_storage.save("original/abc/file.pdf", io.BytesIO(b"content"))

    assert written == 7
    assert local_storage.exists("original/abc/file.pdf")
    with local_storage.open("original/abc/file.pdf") as handle:
        assert handle.read() == b"content"


def test_save_leaves_no_temporary_files(local_storage: LocalStorage) -> None:
    local_storage.save("original/abc/file.pdf", io.BytesIO(b"content"))

    names = [p.relative_to(local_storage.root).as_posix() for p in files_under(local_storage.root)]
    assert names == ["original/abc/file.pdf"]


def test_save_never_overwrites(local_storage: LocalStorage) -> None:
    local_storage.save("original/abc/file.pdf", io.BytesIO(b"first"))

    with pytest.raises(StorageError):
        local_storage.save("original/abc/file.pdf", io.BytesIO(b"second"))

    with local_storage.open("original/abc/file.pdf") as handle:
        assert handle.read() == b"first"


def test_failed_save_leaves_nothing_behind(local_storage: LocalStorage) -> None:
    class ExplodingStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            raise OSError("disk on fire")

    with pytest.raises(StorageError):
        local_storage.save("original/abc/file.pdf", ExplodingStream(b"x"))

    assert files_under(local_storage.root) == []
    assert not (local_storage.root / "original" / "abc").exists()


def test_failed_save_from_unexpected_error_cleans_up(local_storage: LocalStorage) -> None:
    class BrokenStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        local_storage.save("original/abc/file.pdf", BrokenStream(b"x"))

    assert files_under(local_storage.root) == []


def test_delete_removes_file_and_empty_document_directory(local_storage: LocalStorage) -> None:
    local_storage.save("original/abc/file.pdf", io.BytesIO(b"content"))

    local_storage.delete("original/abc/file.pdf")

    assert not local_storage.exists("original/abc/file.pdf")
    assert not (local_storage.root / "original" / "abc").exists()
    assert (local_storage.root / "original").exists()


def test_delete_missing_file_is_not_an_error(local_storage: LocalStorage) -> None:
    local_storage.delete("original/nothing/here.pdf")


@pytest.mark.parametrize(
    "key",
    ["", "../escape.pdf", "original/../../escape.pdf", "/etc/passwd", "a\\b", "original/\x00.pdf", "."],
)
def test_unsafe_keys_are_rejected(local_storage: LocalStorage, key: str) -> None:
    with pytest.raises(StorageError):
        local_storage.save(key, io.BytesIO(b"x"))
    with pytest.raises(StorageError):
        local_storage.exists(key)


def test_build_original_key_uses_document_uuid() -> None:
    import uuid

    document_id = uuid.uuid4()
    assert build_original_key(document_id, f"{document_id}.pdf") == (
        f"original/{document_id}/{document_id}.pdf"
    )
