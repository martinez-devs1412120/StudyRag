"""Pydantic request/response schemas for the API surface.

Kept thin and forward-compatible: future endpoints (query, ingest, history)
will live here too.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when the service is responsive")
    api_version: str
    database_configured: bool
    vector_store_chunks: Optional[int] = Field(
        None, description="Number of chunks in the on-disk vector store, if reachable"
    )


class Source(BaseModel):
    source: str
    chunk_id: int
    score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=400)
    top_k: int = Field(5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    relevant: bool = True
