import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)


def build_engine(database_url: str) -> Engine:
    return create_engine(
        database_url,
        pool_pre_ping=True,
        # Sessions always run in UTC so timestamps are consistent across machines.
        connect_args={"connect_timeout": 3, "options": "-c timezone=UTC"},
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return build_engine(get_settings().database_url.get_secret_value())


def get_session() -> Session:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a session and always close it. Callers commit explicitly."""
    session = get_session()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> tuple[bool, str]:
    """Return (connected, detail) without raising."""
    try:
        get_settings()
    except ValidationError:
        return False, "DATABASE_URL is not set"
    try:
        with get_engine().connect() as conn:
            version = conn.execute(text("select version()")).scalar() or ""
        return True, version.split(" on ")[0]
    except Exception as exc:
        # psycopg messages identify host, port and user but never the password.
        reason = str(getattr(exc, "orig", None) or exc).strip().splitlines()
        logger.warning("Database connection check failed: %s", reason[0] if reason else exc)
        return False, reason[0] if reason else type(exc).__name__


if __name__ == "__main__":
    ok, info = check_connection()
    print(("Connected: " if ok else "Not connected: ") + info)
    raise SystemExit(0 if ok else 1)
