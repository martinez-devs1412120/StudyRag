"""Settings: cached, env-driven, reset clears the cache."""
from api import config as api_config


def test_settings_default_values():
    s = api_config.get_settings()
    assert s.api_title == "StudyRAG API"
    assert s.api_version == "0.1.0"
    assert s.top_k == 5
    assert s.rate_limit_per_minute == 10


def test_settings_cached():
    s1 = api_config.get_settings()
    s2 = api_config.get_settings()
    assert s1 is s2


def test_reset_clears_cache():
    api_config.get_settings()
    api_config.reset_settings_cache()
    s_after = api_config.get_settings()
    assert s_after is not None


def test_database_url_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:5432/d")
    api_config.reset_settings_cache()
    assert api_config.get_settings().database_url == "postgresql+psycopg://u:p@h:5432/d"
