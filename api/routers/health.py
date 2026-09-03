"""GET /health — liveness probe. Returns {"status": "ok"}.

Detailed readiness (DB configured, vector-store chunk count, API version)
will live in a separate /readyz endpoint added in a later task, so this
one stays cheap and dependency-free for load balancers.
"""
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}
