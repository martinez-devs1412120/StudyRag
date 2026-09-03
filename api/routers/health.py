"""GET /health — liveness + readiness in one endpoint.

Reports the API version, whether a database URL is configured, and (best
effort) how many chunks are in the vector store. The store call is wrapped:
a corrupt store must not make the whole API look down.
"""
import logging
from pathlib import Path

from fastapi import APIRouter, Depends

from api.config import Settings, get_settings
from api.database import is_database_configured
from api.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    chunk_count: int | None = None
    try:
        # Reuse the existing pipeline's store when available — no new import
        # path. Falls through gracefully if the index isn't initialized.
        from src.rag.embeddings import EmbeddingStore
        from src.rag.config import get_config as get_rag_config

        rag_cfg = get_rag_config()
        db_path = Path(rag_cfg["VECTOR_DB_PATH"]) / "chunks.db"
        if db_path.exists():
            # Cheap COUNT(*) without booting the full pipeline.
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    except Exception:
        logger.exception("health: vector store probe failed")

    return HealthResponse(
        status="ok",
        api_version=settings.api_version,
        database_configured=is_database_configured(),
        vector_store_chunks=chunk_count,
    )
