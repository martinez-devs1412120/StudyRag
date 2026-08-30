"""Per-user history — Firebase Firestore (primary) -> Supabase -> local JSON fallback."""
import os
import re
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict

HISTORY_DIR = Path("data/history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

def _local_path(email: str) -> Path:
    # Keep the legacy mapping (@ -> _at_, everything else non-alphanumeric -> _)
    # so files written by older versions stay readable, while hostile email
    # strings can never introduce path separators or escape HISTORY_DIR.
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", email.replace("@", "_at_")) or "unknown"
    return HISTORY_DIR / f"{safe}.json"

def save_record(email: str, question: str, answer: str, sources: List[Dict], local_only: bool = False):
    rec = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "ts": datetime.now(timezone.utc).isoformat(),
        "email": email,
    }
    # try Firebase Firestore first (persistent on Render)
    if not local_only:
        try:
            from src.rag.auth_firebase import init_firebase
            db = init_firebase()
            if db is not None:
                doc = {
                    "user_email": email,
                    "question": question,
                    "answer": answer,
                    "sources": json.dumps(sources),
                    "created_at": datetime.now(timezone.utc),
                }
                db.collection("history").add(doc)
                return True
        except Exception:
            pass
        # try Supabase
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

def load_history(email: str, limit: int = 20, local_only: bool = False) -> List[Dict]:
    # Firebase first
    if not local_only:
        try:
            from src.rag.auth_firebase import init_firebase
            db = init_firebase()
            if db is not None:
                q = db.collection("history").where("user_email","==",email).order_by("created_at", direction="DESCENDING").limit(limit).stream()
                rows = list(q)
                if rows:
                    out = []
                    for d in rows:
                        r = d.to_dict()
                        out.append({
                            "question": r.get("question"),
                            "answer": r.get("answer"),
                            "sources": json.loads(r.get("sources","[]")) if isinstance(r.get("sources"), str) else r.get("sources", []),
                            "ts": r.get("created_at").isoformat() if hasattr(r.get("created_at"), "isoformat") else str(r.get("created_at")),
                            "email": r.get("user_email"),
                        })
                    return list(reversed(out))
        except Exception:
            pass
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

def clear_history(email: str, local_only: bool = False):
    p = _local_path(email)
    if p.exists():
        p.unlink()
    if local_only:
        return
    # Firebase
    try:
        from src.rag.auth_firebase import init_firebase
        db = init_firebase()
        if db is not None:
            docs = db.collection("history").where("user_email","==",email).stream()
            for d in docs:
                d.reference.delete()
    except Exception:
        pass
    try:
        from src.rag.auth import get_supabase
        sb = get_supabase()
        if sb:
            sb.table("history").delete().eq("user_email", email).execute()
    except Exception:
        pass
