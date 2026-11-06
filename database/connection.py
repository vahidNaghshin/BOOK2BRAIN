import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(
        os.environ["DATABASE_URL"],
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3},
    )


def get_session() -> Session:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)()


def check_connection() -> tuple[bool, str]:
    """Return (connected, detail) without raising."""
    if not os.getenv("DATABASE_URL"):
        return False, "DATABASE_URL is not set"
    try:
        with get_engine().connect() as conn:
            version = conn.execute(text("select version()")).scalar() or ""
        return True, version.split(" on ")[0]
    except Exception as exc:
        # psycopg messages identify host, port and user but never the password.
        reason = str(getattr(exc, "orig", None) or exc).strip().splitlines()
        return False, reason[0] if reason else type(exc).__name__


if __name__ == "__main__":
    ok, info = check_connection()
    print(("Connected: " if ok else "Not connected: ") + info)
    raise SystemExit(0 if ok else 1)
