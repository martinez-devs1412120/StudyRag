"""Local history: path sanitization and demo-account isolation."""
from pathlib import Path

from src.rag import history


def test_hostile_email_cannot_escape_history_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path)
    p = history._local_path("../../evil@gmail.com")
    assert p.parent == tmp_path
    assert "/" not in p.name and "\\" not in p.name


def test_filename_stays_legacy_compatible(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path)
    assert history._local_path("user@gmail.com").name == "user_at_gmail_com.json"


def test_local_only_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path)
    email = "sectest@gmail.com"
    history.save_record(email, "what is <script>x</script>", "answer", [], local_only=True)
    rows = history.load_history(email, local_only=True)
    assert len(rows) == 1 and rows[0]["answer"] == "answer"
    history.clear_history(email, local_only=True)
    assert history.load_history(email, local_only=True) == []
