"""Health endpoints: liveness vs readiness.

GET /health  -> always cheap {"status": "ok"} for load balancers
GET /readyz  -> detailed readiness (DB configured, vector store chunks,
                API version) for ops dashboards and Kubernetes probes

The two are split so LBs can hit /health thousands of times/sec without
touching the DB, while /readyz gives ops a real signal.
"""
import logging
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])

logger = logging.getLogger(__name__)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


class ReadyResponse(BaseModel):
    status: str  # "ready" | "degraded"
    api_version: str
    database_configured: bool
    database_reachable: bool
    vector_store_chunks: int | None = None


@router.get("/readyz", response_model=ReadyResponse)
def readyz() -> ReadyResponse:
    """Detailed readiness. Never raises; reports partial state honestly."""
    from api.config import get_settings
    from api.db.session import is_db_available

    settings = get_settings()
    db_configured = bool(settings.database_url)
    db_reachable = is_db_available()

    # Best-effort vector store probe — the SQLite file lives in the API
    # process's working dir (or wherever VECTOR_DB_PATH points).
    chunk_count: int | None = None
    try:
        from src.rag.config import get_config as get_rag_config
        db_path = Path(get_rag_config()["VECTOR_DB_PATH"]) / "chunks.db"
        if db_path.exists():
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    except Exception:
        logger.exception("readyz: vector store probe failed")

    status = "ready" if (not db_configured or db_reachable) else "degraded"
    return ReadyResponse(
        status=status,
        api_version=settings.api_version,
        database_configured=db_configured,
        database_reachable=db_reachable,
        vector_store_chunks=chunk_count,
    )
