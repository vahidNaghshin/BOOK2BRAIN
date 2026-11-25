"""Check that the migrations work on a copy of your development database without losing its data.

    python -m tools.check_migrations_on_copy                  # copy the configured database, test, drop the copy
    python -m tools.check_migrations_on_copy --keep           # keep the copy for inspection
    python -m tools.check_migrations_on_copy --from-revision 3ff86bf68465

Your database is only read (it is copied with CREATE DATABASE ... TEMPLATE, which needs no other
connection to it: stop the app first if PostgreSQL refuses). On the copy it
  1. fingerprints every table (row count, and a checksum of the ids and of the Phase 1 rows),
  2. runs `alembic upgrade head` (a no-op when the database is current),
  3. downgrades to --from-revision (default: the last revision before Day 4), which gives a real database
     at the older schema, and upgrades it to head again,
  4. checks that Phase 1 rows are byte-for-byte unchanged by the downgrade and the upgrade, that the upgrade
     lost no row and no id of any table the older schema had, and that the models and migrations agree
     (`alembic check`).
Rolling back is destructive by design: it drops the later revisions' tables and columns, so their data
(flags, equation sources, relationship executions) is gone from the copy; the upgrade only backfills what
it can derive. The report lists both.
"""

from __future__ import annotations

import argparse
import sys

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine, make_url

from app.config import PROJECT_ROOT, get_settings
from database.connection import build_engine

PHASE_1_TABLES = (
    "books", "documents", "processing_runs", "processing_errors", "extracted_blocks", "equations", "structural_nodes",
    "paragraphs", "passages", "passage_paragraphs", "structure_analyses",
)
DEFAULT_FROM = "b9221d0781db"  # the last revision before Day 4


def alembic_config(url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["sqlalchemy_url"] = url
    return config


def fingerprint(engine: Engine) -> dict[str, tuple[int, str, str]]:
    """table -> (rows, checksum of ids, checksum of whole rows). Whole-row checksums are only compared for Phase 1."""
    result: dict[str, tuple[int, str, str]] = {}
    with engine.connect() as conn:
        for table in sorted(inspect(conn).get_table_names()):
            if table == "alembic_version":
                continue
            has_id = any(c["name"] == "id" for c in inspect(conn).get_columns(table))
            rows = conn.execute(text(f'SELECT count(*) FROM "{table}"')).scalar()
            ids = conn.execute(text(f"SELECT md5(coalesce(string_agg(id::text, ',' ORDER BY id::text), '')) FROM \"{table}\"")).scalar() if has_id else "-"
            whole = conn.execute(text(f'SELECT md5(coalesce(string_agg(t::text, \'|\' ORDER BY t::text), \'\')) FROM "{table}" t')).scalar()
            result[table] = (rows, ids, whole)
    return result


def create_copy(url: str, name: str) -> str:
    source = make_url(url)
    admin = build_engine(source.set(database="postgres").render_as_string(hide_password=False))
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        conn.execute(text(f'CREATE DATABASE "{name}" TEMPLATE "{source.database}"'))
    admin.dispose()
    return source.set(database=name).render_as_string(hide_password=False)


def drop_copy(url: str, name: str) -> None:
    admin = build_engine(make_url(url).set(database="postgres").render_as_string(hide_password=False))
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    admin.dispose()


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the migrations on a copy of the development database.")
    parser.add_argument("--from-revision", default=DEFAULT_FROM, help="downgrade the copy to this revision, then upgrade to head")
    parser.add_argument("--keep", action="store_true", help="keep the copy")
    args = parser.parse_args(argv)

    url = get_settings().database_url.get_secret_value()
    name = f"{make_url(url).database}_migration_check"
    problems: list[str] = []
    try:
        copy_url = create_copy(url, name)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not copy the database: {getattr(exc, 'orig', exc)}\nStop anything connected to it (the app, a notebook) and try again.", file=sys.stderr)
        return 2
    engine = build_engine(copy_url)
    try:
        start = current_revision(engine)
        before = fingerprint(engine)
        print(f"Copy '{name}' at revision {start}: {sum(r for r, _, _ in before.values())} rows in {len(before)} tables.")

        command.upgrade(alembic_config(copy_url), "head")
        after_noop = fingerprint(engine)
        if after_noop != before:
            problems.append("`upgrade head` changed a database that was already current.")
        print("1. upgrade head on the current database: " + ("no change." if after_noop == before else "CHANGED"))

        command.downgrade(alembic_config(copy_url), args.from_revision)
        middle = fingerprint(engine)
        print(f"2. downgrade to {args.from_revision} (this is the 'older database' the upgrade is tested on): {len(middle)} tables remain.")
        for table in PHASE_1_TABLES:
            if middle.get(table) != before.get(table):
                problems.append(f"Downgrading changed Phase 1 table '{table}'.")
        removed = {t: before[t][0] - middle[t][0] for t in middle if middle[t][0] != before[t][0]}
        for table, lost in sorted(removed.items()):
            print(f"   downgrade removed {lost} row(s) of {table} that only exist in the newer schema (expected, rolling back is destructive)")

        command.upgrade(alembic_config(copy_url), "head")
        final = fingerprint(engine)
        print(f"3. upgrade head from the older database: revision {current_revision(engine)}.")
        for table, (rows, ids, whole) in middle.items():
            if table not in final:
                problems.append(f"Table '{table}' is missing after the upgrade.")
            elif table in PHASE_1_TABLES and final[table] != middle[table]:
                problems.append(f"Phase 1 table '{table}' changed by the upgrade.")
            elif final[table][0] < rows:
                problems.append(f"Table '{table}' lost rows in the upgrade ({rows} -> {final[table][0]}).")
            elif final[table][1] != ids:
                problems.append(f"Table '{table}' has different ids after the upgrade.")
        for table in sorted(t for t in final if t not in middle):
            print(f"   created by the upgrade: {table}: {final[table][0]} row(s) backfilled")

        try:
            command.check(alembic_config(copy_url))
            print("4. alembic check: models and migrations agree.")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"alembic check: {exc}")
    finally:
        engine.dispose()
        if not args.keep:
            drop_copy(url, name)

    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print("\nOK: Phase 1 data is unchanged, the upgrade lost no row, and the schema matches the models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
