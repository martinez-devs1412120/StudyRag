"""SQLAlchemy models (reserved for Phase 2). The API boots without them.

A real database connection is intentionally NOT opened here — we only declare
the engine and sessionmaker conditionally on DATABASE_URL being set, so the
health endpoint works on a fresh dev machine with no Postgres available.
"""
import logging
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Project-wide declarative base for all ORM models."""


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def _ensure_engine():
    """Lazy-init the engine only when a DATABASE_URL is actually set."""
    global _engine, _SessionLocal
    if _engine is not None:
        return
    url = get_settings().database_url
    if not url:
        return  # DB-less mode: API still works, just no persistence
    _engine = create_engine(url, future=True, pool_pre_ping=True)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    logger.info("Database engine initialized (url=%s...)", url[:20])


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency. Yields a session per request, or does nothing if
    the database isn't configured (so endpoints that don't need a DB still work)."""
    _ensure_engine()
    if _SessionLocal is None:
        yield None
        return
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
