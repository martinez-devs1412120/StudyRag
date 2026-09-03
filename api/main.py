"""StudyRAG API — FastAPI entry point.

Separate from the existing Streamlit app and CLI. The Streamlit UI keeps
working unchanged; this service exposes a JSON API for future frontends,
mobile apps, or programmatic clients.

Run locally:
    pip install -r requirements-api.txt
    uvicorn api.main:app --reload

No database is required to start — DATABASE_URL is optional. When unset,
endpoints that need persistence simply skip it.
"""
import logging

from fastapi import FastAPI

from api.config import get_settings
from api.routers import health

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("studyrag.api")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        # The existing Streamlit UI is the current "frontend"; the API is
        # only for programmatic clients, so no CORS allow-all here.
    )
    app.include_router(health.router)
    logger.info("StudyRAG API %s ready (db configured: %s)", settings.api_version, bool(settings.database_url))
    return app


app = create_app()
