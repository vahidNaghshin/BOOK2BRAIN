from collections.abc import Iterator

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from database.connection import build_engine
from database.models import Base
from tests.conftest import alembic_config, derive_test_url, ensure_database, migrate

EXPECTED_TABLES = {"books", "documents", "processing_runs", "processing_errors"}


@pytest.fixture
def migration_engine() -> Iterator[Engine]:
    """A throwaway database so downgrade tests never disturb the shared test database."""
    url = derive_test_url("_test_migrations")
    ensure_database(url)
    migration_engine = build_engine(url)
    command.downgrade(alembic_config(url), "base")
    yield migration_engine
    migration_engine.dispose()


def test_upgrade_creates_schema_and_vector_extension(migration_engine: Engine) -> None:
    migrate(str(migration_engine.url.render_as_string(hide_password=False)))

    assert EXPECTED_TABLES <= set(inspect(migration_engine).get_table_names())
    with migration_engine.connect() as conn:
        extension = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
    assert extension == 1


def test_migrations_match_models(migration_engine: Engine) -> None:
    migrate(migration_engine.url.render_as_string(hide_password=False))

    with migration_engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)

    assert diff == []


def test_downgrade_removes_tables_and_upgrade_is_repeatable(migration_engine: Engine) -> None:
    url = migration_engine.url.render_as_string(hide_password=False)
    migrate(url)
    command.downgrade(alembic_config(url), "base")
    assert not (EXPECTED_TABLES & set(inspect(migration_engine).get_table_names()))

    migrate(url)
    assert EXPECTED_TABLES <= set(inspect(migration_engine).get_table_names())
