"""Durable chunk storage in Firestore — the source of truth for ingested documents.

Render's filesystem is ephemeral: every deploy wipes data/, so the SQLite
vector index alone would lose all documents on each release. Chunks persisted
here let the app rebuild its index on boot (stateless app + durable store +
rebuildable index). When Firebase is not configured every function degrades
gracefully to a no-op and the app runs local-only (index survives until the
next deploy).
"""
from typing import Dict, List
import logging

from src.rag.chunking import chunk_hash

logger = logging.getLogger(__name__)

COLLECTION = "documents"
_BATCH = 400  # Firestore caps a single batched write at 500 operations


def _db():
    try:
        from src.rag.auth_firebase import init_firebase
        return init_firebase()
    except Exception:
        logger.exception("Firestore init failed")
        return None


def push_chunks(chunks: List[Dict]) -> bool:
    """Upsert chunks (content-addressed IDs -> re-pushing is idempotent)."""
    db = _db()
    if db is None:
        return False
    try:
        batch = db.batch()
        written = 0
        for ch in chunks:
            doc = {
                "source": ch["source"],
                "chunk_id": int(ch["chunk_id"]),
                "text": ch["text"],
            }
            batch.set(db.collection(COLLECTION).document(chunk_hash(ch["source"], ch["text"])), doc)
            written += 1
            if written % _BATCH == 0:
                batch.commit()
                batch = db.batch()
        if written % _BATCH:
            batch.commit()
        return True
    except Exception:
        logger.exception("Firestore push_chunks failed")
        return False


def remove_source(source: str) -> bool:
    db = _db()
    if db is None:
        return False
    try:
        batch = db.batch()
        docs = db.collection(COLLECTION).where("source", "==", source).stream()
        n = 0
        for d in docs:
            batch.delete(d.reference)
            n += 1
            if n % _BATCH == 0:
                batch.commit()
                batch = db.batch()
        if n % _BATCH:
            batch.commit()
        return True
    except Exception:
        logger.exception("Firestore remove_source failed")
        return False


def all_chunks() -> List[Dict]:
    """Every stored chunk, ordered so the index rebuild is deterministic."""
    db = _db()
    if db is None:
        return []
    try:
        docs = db.collection(COLLECTION).stream()
        chunks = [
            {"source": d.to_dict().get("source"), "chunk_id": d.to_dict().get("chunk_id"), "text": d.to_dict().get("text")}
            for d in docs
        ]
        chunks = [c for c in chunks if c["text"]]
        chunks.sort(key=lambda c: (c["source"], c["chunk_id"]))
        return chunks
    except Exception:
        logger.exception("Firestore all_chunks failed")
        return []


def clear_all() -> bool:
    db = _db()
    if db is None:
        return False
    try:
        batch = db.batch()
        docs = db.collection(COLLECTION).stream()
        n = 0
        for d in docs:
            batch.delete(d.reference)
            n += 1
            if n % _BATCH == 0:
                batch.commit()
                batch = db.batch()
        if n % _BATCH:
            batch.commit()
        return True
    except Exception:
        logger.exception("Firestore clear_all failed")
        return False
