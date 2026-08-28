"""Per-user history — Supabase table `history` or local JSON fallback."""
import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict

HISTORY_DIR = Path("data/history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

def _local_path(email: str) -> Path:
    safe = email.replace("@","_at_").replace(".","_")
    return HISTORY_DIR / f"{safe}.json"

def save_record(email: str, question: str, answer: str, sources: List[Dict]):
    rec = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "ts": datetime.now(timezone.utc).isoformat(),
        "email": email,
    }
    # try Supabase first
    try:
        from src.rag.auth import get_supabase
        sb = get_supabase()
        if sb:
            sb.table("history").insert({
                "user_email": email,
                "question": question,
                "answer": answer,
                "sources": json.dumps(sources),
            }).execute()
            return True
    except Exception:
        pass
    # fallback local
    p = _local_path(email)
    data = []
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = []
    data.append(rec)
    # keep last 100
    data = data[-100:]
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return True

def load_history(email: str, limit: int = 20) -> List[Dict]:
    try:
        from src.rag.auth import get_supabase
        sb = get_supabase()
        if sb:
            res = sb.table("history").select("*").eq("user_email", email).order("created_at", desc=True).limit(limit).execute()
            rows = getattr(res, "data", []) or []
            # map to local shape
            out = []
            for r in rows:
                out.append({
                    "question": r.get("question"),
                    "answer": r.get("answer"),
                    "sources": json.loads(r.get("sources","[]")) if isinstance(r.get("sources"), str) else r.get("sources", []),
                    "ts": r.get("created_at"),
                    "email": r.get("user_email"),
                })
            if out:
                return list(reversed(out))
    except Exception:
        pass
    p = _local_path(email)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data[-limit:]
    except Exception:
        return []

def clear_history(email: str):
    p = _local_path(email)
    if p.exists():
        p.unlink()
    try:
        from src.rag.auth import get_supabase
        sb = get_supabase()
        if sb:
            sb.table("history").delete().eq("user_email", email).execute()
    except Exception:
        pass
