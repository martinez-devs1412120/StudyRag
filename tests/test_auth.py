"""OAuth state signing and id_token verification fail-closed behavior."""
import hmac
import hashlib
import time

import pytest

from src.rag import auth as A


@pytest.fixture
def oauth_secret(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "unit-test-secret")
    monkeypatch.delenv("STUDYRAG_SECRET", raising=False)
    return b"unit-test-secret"


def test_state_roundtrip(oauth_secret):
    state = A.make_oauth_state()
    assert state and A.verify_oauth_state(state)


def test_state_tamper_rejected(oauth_secret):
    state = A.make_oauth_state()
    bad = state[:-1] + ("0" if state[-1] != "0" else "1")
    assert not A.verify_oauth_state(bad)


def test_state_garbage_rejected(oauth_secret):
    assert not A.verify_oauth_state(None)
    assert not A.verify_oauth_state("abc.def")
    assert not A.verify_oauth_state("not-a-state")


def test_state_expired_rejected(oauth_secret, monkeypatch):
    old_ts = str(int(time.time()) - 3600)
    sig = hmac.new(oauth_secret, old_ts.encode(), hashlib.sha256).hexdigest()[:32]
    assert not A.verify_oauth_state(f"{old_ts}.{sig}")


def test_state_without_secret_skips(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("STUDYRAG_SECRET", raising=False)
    assert A.make_oauth_state() is None
    assert A.verify_oauth_state(None) is True  # flow can't work without client secret anyway


def test_id_token_garbage_rejected(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "fake-id.apps.googleusercontent.com")
    assert A.verify_google_id_token("aaa.bbb.ccc") is None
    assert A.verify_google_id_token(None) is None
    assert A.verify_google_id_token("") is None


def test_id_token_without_client_id_rejected(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    assert A.verify_google_id_token("aaa.bbb.ccc") is None
