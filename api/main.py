"""StudyRAG API — FastAPI entry point.

Separate from the existing Streamlit app and CLI. The Streamlit UI keeps
working unchanged; this service exposes a JSON API for future frontends,
mobile apps, or programmatic clients.

Run locally:
    pip install -r requirements-api.txt
    uvicorn api.main:app --reload
"""
import logging
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.routers import health

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("studyrag.api")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.api_title, version=settings.api_version)

    # CORS: open by default (the API is read-only and public-info today).
    # Lock down origins/credentials when real auth lands.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # JSON error handlers — never leak HTML stack traces from middleware.
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"error": "internal_server_error"})

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": str(exc)})

    app.include_router(health.router)
    logger.info("StudyRAG API %s ready (db configured: %s)", settings.api_version, bool(settings.database_url))
    return app


app = create_app()
