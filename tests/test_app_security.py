"""End-to-end security regressions: the real app must keep failing closed.

Runs app.py via Streamlit's AppTest (no network calls: no chat query is
submitted, so Groq is never hit; Firebase/Firestore stay unconfigured).
"""
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.usefixtures("admin_emails")

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def admin_emails(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "boss@gmail.com")
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)


def sget(at, key, default=None):
    try:
        return at.session_state[key]
    except Exception:
        return default


def run_app(**kwargs):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    for k, v in kwargs.items():
        if k == "query_params":
            for qk, qv in v.items():
                at.query_params[qk] = qv
        else:
            at.session_state[k] = v
    at.run()
    return at


def go_database(at):
    at.button(key="nav_DATABASE").click().run()
    return at


def db_buttons(at):
    d = next((b for b in at.button if (b.key or "").startswith("del_")), None)
    c = next((b for b in at.button if "Clear store" in (b.label or "")), None)
    i = next((b for b in at.button if "Ingest" in (b.label or "")), None)
    return d, c, i


def test_app_runs_clean():
    at = run_app()
    assert not at.exception


def test_xss_payload_rendered_escaped():
    payload = "<img src=x onerror=alert(document.cookie)>"
    at = run_app(messages=[
        {"role": "user", "content": payload},
        {"role": "assistant", "content": payload,
         "sources": [{"source": payload + ".pdf", "score": 0.9, "chunk_id": 0}]},
    ])
    joined = "\n".join(m.value for m in at.markdown)
    assert "&lt;img src=x" in joined
    assert "<img src=x" not in joined


def test_verified_email_without_token_fails_closed():
    at = run_app(query_params={"verified_email": "victim@gmail.com"})
    assert not at.exception
    assert sget(at, "user") is None
    assert "verified_email" not in dict(at.query_params)


def test_mock_signin_still_works():
    at = run_app()
    at.text_input(key="gmail_demo").set_value("attacker@gmail.com")
    at.button(key="btn_signin").click().run()
    user = sget(at, "user") or {}
    assert user.get("email") == "attacker@gmail.com"
    assert user.get("provider") == "mock"


def test_mock_user_cannot_manage_even_with_admin_email():
    # mock sign-in AS the admin's email must stay locked out
    at = run_app()
    at.text_input(key="gmail_demo").set_value("boss@gmail.com")
    at.button(key="btn_signin").click().run()
    go_database(at)
    d, c, _ = db_buttons(at)
    if d is not None:
        assert d.disabled
    if c is not None:
        assert c.disabled


def test_verified_admin_can_manage():
    at = run_app(user={"email": "boss@gmail.com", "name": "boss", "provider": "google-verified"})
    go_database(at)
    d, c, _ = db_buttons(at)
    if d is not None:
        assert not d.disabled
    if c is not None:
        assert not c.disabled
    assert not any("Admins only" in (x.value or "") for x in at.caption)


def test_verified_non_admin_cannot_manage():
    at = run_app(user={"email": "intruder@gmail.com", "name": "n", "provider": "google-verified"})
    go_database(at)
    d, c, _ = db_buttons(at)
    if d is not None:
        assert d.disabled
    if c is not None:
        assert c.disabled
    assert any("ADMIN_EMAILS" in (x.value or "") for x in at.caption)


def test_empty_allowlist_locks_everyone(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "")
    at = run_app(user={"email": "boss@gmail.com", "name": "boss", "provider": "google-verified"})
    go_database(at)
    d, c, _ = db_buttons(at)
    if d is not None:
        assert d.disabled


def test_oauth_callback_with_bad_state_fails_closed(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "fake-id.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "fake-secret")
    at = run_app(query_params={"code": "4/0-fakecode", "state": "12345.deadbeef"})
    assert not at.exception
    assert sget(at, "user") is None
    assert "code" not in dict(at.query_params)


def test_ingest_button_absent_without_admin():
    for user in (None,
                 {"email": "boss@gmail.com", "provider": "mock"},
                 {"email": "intruder@gmail.com", "provider": "google-verified"}):
        at = run_app(**({"user": user} if user else {}))
        go_database(at)
        ingest = next((b for b in at.button if "Ingest" in (b.label or "")), None)
        assert ingest is None, f"ingest reachable for {user}"
