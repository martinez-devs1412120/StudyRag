"""SQLAlchemy engine + session factory + FastAPI dependency.

Pattern follows the SQLAlchemy 2.0 "future" style: one Engine, one
SessionLocal, sessions opened per-request. The engine is created lazily
so importing this module never opens a network connection (important
for unit tests and for the /health endpoint that runs without a DB).

Callers inject a Session via `Depends(get_db)`. The dependency yields
None when no DATABASE_URL is configured so endpoints that don't need
persistence still work; endpoints that DO need it must handle the None
case explicitly (so we never accidentally write to "nowhere").
"""
import logging
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from api.config import get_settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _ensure_engine() -> Engine | None:
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine
    url = get_settings().database_url
    if not url:
        return None
    _engine = create_engine(url, future=True, pool_pre_ping=True)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    logger.info("DB engine initialized (url=%s...)", url.split("@", 1)[-1])
    return _engine


def get_db() -> Generator[Session | None, None, None]:
    """FastAPI dependency. Yields a Session per request, or None if
    DATABASE_URL is not configured (so endpoints can still respond)."""
    _ensure_engine()
    if _SessionLocal is None:
        yield None
        return
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def is_db_available() -> bool:
    """True iff a DATABASE_URL was configured and the engine is initialized."""
    return _ensure_engine() is not None
