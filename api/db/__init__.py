"""Database package: engine, session, and FastAPI dependencies.

Lives parallel to `api/database.py` (which is a small DB-availability
helper). When the database is configured, sessions are created here and
injected into route handlers via `get_db`.
"""
