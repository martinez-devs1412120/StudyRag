"""API configuration: Pydantic Settings for env-driven config.

The PostgreSQL URL is declared but NOT connected to — wiring it into a real
database will happen in a later phase. The application boots and serves
health checks without a DB.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Env vars are read automatically; .env file is honored when present
    # but the /api layer doesn't share the repo-root .env by default — pass
    # DATABASE_URL, etc., explicitly when running uvicorn.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_title: str = "StudyRAG API"
    api_version: str = "0.1.0"

    # Reserved for Phase 2: Postgres/Supabase connection. Optional — the
    # health endpoint works without it.
    database_url: str | None = None

    # Reuses the existing retrieval config; the API and Streamlit share
    # the same on-disk vector store under VECTOR_DB_PATH.
    vector_db_path: str = "./data/chroma_db"

    # RAG / LLM settings
    top_k: int = 5
    relevance_threshold: float = 0.35  # top score below this -> no LLM call

    # LLM provider (Task 13 will add Groq streaming)
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"

    # Observability (Task 15 will add Langfuse)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    # Rate limiting (Task 16)
    rate_limit_per_minute: int = 10


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached settings accessor.

    Re-parsing env on every call is wasteful and creates surprising test
    behavior (changing env mid-process wouldn't take effect). Cached; tests
    that need different env can call `reset_settings_cache()`.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Drop the cached Settings so the next get_settings() re-reads env.

    Used by tests and by any code that needs to pick up an env change.
    """
    global _settings
    _settings = None
