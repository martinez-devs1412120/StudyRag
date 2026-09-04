"""Database availability helper.

Canonical engine + session logic lives in `api.db.session`. This module
exists for backwards compatibility with the Task 1 scaffold and exposes
just the question "is the database configured?".
"""
from api.db.session import is_db_available


def is_database_configured() -> bool:
    """True iff DATABASE_URL is set. The API boots and serves requests
    regardless; persistence is opt-in."""
    return is_db_available()
