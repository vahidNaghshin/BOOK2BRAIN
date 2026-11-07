"""Shared fixtures.

Database tests run against a dedicated PostgreSQL database (default: the configured
database name plus '_test'). It is created on demand and built with the real Alembic
migrations. The production database is never touched.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT, Settings, get_settings
from app.schemas import BookMetadata
from app.services.document_service import DocumentService
from app.services.storage_service import LocalStorage
from database.connection import build_engine

TABLES = "processing_errors, processing_runs, documents, books"


def derive_test_url(suffix: str = "_test") -> str:
    """Test database URL: TEST_DATABASE_URL if set, else the app database + suffix."""
    explicit = os.environ.get("TEST_DATABASE_URL")
    url = make_url(explicit or get_settings().database_url.get_secret_value())
    if not explicit:
        url = url.set(database=f"{url.database}{suffix}")
    if "_test" not in (url.database or ""):
        raise RuntimeError(
            f"Refusing to run tests against database '{url.database}': "
            "the test database name must contain '_test'."
        )
    return url.render_as_string(hide_password=False)


def ensure_database(url: str) -> None:
    target = make_url(url)
    admin = build_engine(target.set(database="postgres").render_as_string(hide_password=False))
    try:
        with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": target.database}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{target.database}"'))
    except OperationalError as exc:
        pytest.fail(
            "PostgreSQL is not reachable for database tests. "
            f"Start it with 'docker compose up -d'. ({exc.orig})",
            pytrace=False,
        )
    finally:
        admin.dispose()


def alembic_config(url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["sqlalchemy_url"] = url
    return config


def migrate(url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = derive_test_url()
    ensure_database(url)
    migrate(url)
    return url


@pytest.fixture(scope="session")
def engine(test_database_url: str) -> Iterator[Engine]:
    test_engine = build_engine(test_database_url)
    yield test_engine
    test_engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    db_session = Session(bind=engine, autoflush=False, expire_on_commit=False)
    yield db_session
    db_session.rollback()
    db_session.close()
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {TABLES} CASCADE"))


@pytest.fixture
def settings(test_database_url: str, tmp_path: Path) -> Settings:
    return Settings(
        database_url=test_database_url,
        storage_root=tmp_path / "storage",
        max_upload_size_mb=1,
        _env_file=None,
    )


@pytest.fixture
def storage(settings: Settings) -> LocalStorage:
    return LocalStorage(settings.storage_root)


@pytest.fixture
def document_service(session: Session, storage: LocalStorage, settings: Settings) -> DocumentService:
    return DocumentService(session, storage, settings)


@pytest.fixture
def metadata() -> BookMetadata:
    return BookMetadata(title="Dune", author="Frank Herbert")


def stored_files(root: Path) -> list[Path]:
    """All regular files under the storage root, excluding temporary uploads in progress."""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())
