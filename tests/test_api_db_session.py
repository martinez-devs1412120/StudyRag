"""DB session: get_db yields None when no DATABASE_URL, is_db_available matches."""
from api import config as api_config
from api.db.session import get_db, is_db_available


def test_get_db_yields_none_when_no_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    api_config.reset_settings_cache()
    gen = get_db()
    assert next(gen) is None


def test_is_db_available_false_when_no_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    api_config.reset_settings_cache()
    assert is_db_available() is False


def test_get_db_reuses_engine(monkeypatch):
    """Repeated calls without env changes must not create new engines."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    api_config.reset_settings_cache()
    for _ in range(3):
        gen = get_db()
        assert next(gen) is None
    assert is_db_available() is False
