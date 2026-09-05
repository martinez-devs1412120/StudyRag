"""Alembic environment: wires DATABASE_URL from api.config and target_metadata
from api.models so `alembic revision --autogenerate` picks up the schema.

Run from the repo root:
  alembic -c api/alembic.ini upgrade head
  alembic -c api/alembic.ini revision --autogenerate -m "..."
"""
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make sure the repo root is importable so 'api.*' resolves when alembic
# is invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.config import get_settings  # noqa: E402
from api.models import Base  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Pull DATABASE_URL from the same place the rest of the API uses.
# env var wins over alembic.ini (lets CI override per-environment).
db_url = os.getenv("DATABASE_URL") or get_settings().database_url
if not db_url:
    raise RuntimeError(
        "DATABASE_URL is not set. Export it (e.g. postgresql+psycopg://studyrag:studyrag@localhost:5432/studyrag) "
        "or set it in .env before running alembic."
    )
config.set_main_option("sqlalchemy.url", db_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (emits SQL to stdout)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section) or {},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
