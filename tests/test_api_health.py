"""Health and readiness endpoints: liveness stays cheap, readiness reports state."""
from fastapi.testclient import TestClient

from api.main import app


def test_health_returns_literal_ok():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_reports_db_unconfigured_when_no_url():
    client = TestClient(app)
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ready", "degraded")
    assert body["database_configured"] is False
    assert body["database_reachable"] is False
    assert body["api_version"]


def test_health_does_not_touch_vector_store(monkeypatch):
    """Liveness must be cheap — no FS probe. The store is read only by /readyz."""
    called = {"count": 0}

    def boom(*a, **kw):
        called["count"] += 1
        raise RuntimeError("should not be called from /health")

    import src.rag.config as rag_cfg
    monkeypatch.setattr(rag_cfg, "get_config", boom)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert called["count"] == 0
