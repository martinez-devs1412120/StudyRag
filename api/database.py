"""Database connection helper (reserved for Phase 2).

Kept separate from `models.py` so callers can introspect DB availability
without importing the entire SQLAlchemy stack. Today this is a thin
wrapper that reports whether a DATABASE_URL was configured.
"""
from api.config import get_settings


def is_database_configured() -> bool:
    """True iff DATABASE_URL is set. The API boots and serves requests
    regardless; persistence is opt-in."""
    return bool(get_settings().database_url)
