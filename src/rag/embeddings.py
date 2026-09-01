"""Dense embeddings (all-MiniLM-L6-v2 via fastembed/ONNX) + SQLite vector store.

Why not TF-IDF: keyword matching can't bridge vocabulary ("sunlight" never
matched "photosynthesis"). Why SQLite: durable, transactional, and has no
native vector-DB dependency — chromadb's Rust core segfaults on some Windows
installs and hnswlib needs a compiler, while sqlite3 is in the stdlib on both
Windows dev machines and Render's Linux image. At study-material scale
(thousands of chunks) brute-force cosine over 384-dim vectors is milliseconds.

Layout under VECTOR_DB_PATH (default ./data/chroma_db):
  chunks.db   SQLite:
              chunks(id TEXT PK, source TEXT, chunk_id INT, text TEXT, embedding BLOB)
              meta(key TEXT PK, value TEXT)   -- e.g. embedding model name
Legacy TF-IDF files found there are re-embedded once and then removed.
"""
from typing import List, Dict, Any
import json
import logging
import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np

from src.rag.config import get_config
from src.rag.chunking import chunk_hash

logger = logging.getLogger(__name__)

_MODEL = None  # process-wide singleton; model load takes seconds


def _get_model():
    global _MODEL
    if _MODEL is None:
        from fastembed import TextEmbedding
        cfg = get_config()
        name = cfg.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        cache_dir = os.getenv("FASTEMBED_CACHE_DIR") or None
        if cache_dir:
            _MODEL = TextEmbedding(model_name=name, cache_dir=cache_dir)
        else:
            _MODEL = TextEmbedding(model_name=name)
    return _MODEL


class EmbeddingStore:
    """Vector store: dense embeddings + cosine similarity, persisted in SQLite."""

    def __init__(self):
        cfg = get_config()
        self.db_path = Path(cfg["VECTOR_DB_PATH"])
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.db_path / "chunks.db"
        self.model_name = cfg.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._matrix_cache = None  # (sources, chunk_ids, texts, matrix) — invalidated on write
        self._init_schema()
        self._check_model_change()
        self._migrate_legacy()

    # ---------- sqlite plumbing ----------
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS chunks(
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
            conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")

    def _check_model_change(self) -> None:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='model'").fetchone()
        if row and row[0] != self.model_name:
            logger.warning(
                "Embedding model changed (%s -> %s): old vectors are from a different "
                "model. Clear and re-ingest for best retrieval quality.",
                row[0], self.model_name,
            )

    # ---------- embeddings ----------
    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = np.array(list(_get_model().embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-10)  # stored normalized -> cosine is a dot product

    # ---------- write path ----------
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Upsert chunks. Content-addressed IDs make this idempotent: ingesting
        the same source twice never duplicates chunks."""
        if not documents:
            return 0
        rows = [
            (chunk_hash(doc["source"], doc["text"]), doc["source"], int(doc["chunk_id"]), doc["text"])
            for doc in documents
        ]
        embeddings = self._embed([r[3] for r in rows])
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks(id, source, chunk_id, text, embedding) VALUES (?,?,?,?,?)",
                [(r[0], r[1], r[2], r[3], e.tobytes()) for r, e in zip(rows, embeddings)],
            )
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('model', ?)", (self.model_name,))
        self._matrix_cache = None
        return len(rows)

    def replace_source(self, source: str, documents: List[Dict[str, Any]]) -> int:
        """Re-ingest a document: drop its old chunks, upsert the new ones."""
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self._matrix_cache = None
        return self.add_documents(documents)

    def delete_source(self, source: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self._matrix_cache = None

    def clear(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks")
        self._matrix_cache = None

    # ---------- read path ----------
    def count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def get_collection_info(self) -> Dict[str, Any]:
        return {"document_count": self.count()}

    def get_all(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT source, chunk_id, text FROM chunks ORDER BY source, chunk_id").fetchall()
        return [{"source": r[0], "chunk_id": r[1], "text": r[2]} for r in rows]

    def _load_matrix(self):
        if self._matrix_cache is not None:
            return self._matrix_cache
        with self._conn() as conn:
            rows = conn.execute("SELECT source, chunk_id, text, embedding FROM chunks").fetchall()
        if not rows:
            self._matrix_cache = ([], [], [], np.empty((0, 384), dtype=np.float32))
        else:
            matrix = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
            self._matrix_cache = ([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], matrix)
        return self._matrix_cache

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Top-k chunks by cosine similarity (embeddings stored normalized)."""
        if self.count() == 0:
            return []
        query_vec = self._embed([query_text])[0]
        sources, chunk_ids, texts, matrix = self._load_matrix()
        scores = matrix @ query_vec
        k = min(max(int(top_k), 1), len(texts))
        top = np.argsort(scores)[-k:][::-1]
        return [
            {
                "text": texts[i],
                "source": sources[i],
                "chunk_id": chunk_ids[i],
                "score": round(float(max(scores[i], 0.0)), 3),
            }
            for i in top
        ]

    # ---------- legacy migration ----------
    def _migrate_legacy(self) -> None:
        """Re-embed the pre-Phase-1 TF-IDF store (metadata.json/pkl) into SQLite."""
        legacy_json = self.db_path / "metadata.json"
        legacy_pkl = self.db_path / "metadata.pkl"
        if self.count() > 0 or not (legacy_json.exists() or legacy_pkl.exists()):
            return
        try:
            if legacy_json.exists():
                data = json.loads(legacy_json.read_text(encoding="utf-8"))
            else:
                with open(legacy_pkl, "rb") as f:
                    data = pickle.load(f)
            chunks = [d for d in data if isinstance(d, dict) and d.get("text")]
            if chunks:
                n = self.add_documents(chunks)
                logger.info("Migrated legacy TF-IDF store: re-embedded %d chunks", n)
            for name in ("metadata.json", "metadata.pkl", "vectorizer.json", "vectorizer.pkl", "tfidf_vectors.npy"):
                (self.db_path / name).unlink(missing_ok=True)
        except Exception:
            logger.exception("Legacy migration failed; legacy files left in place")
