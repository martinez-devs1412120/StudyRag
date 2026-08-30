"""Embeddings and vector store using TF-IDF (pure Python, no native deps).

Persistence layout (data/chroma_db/):
  tfidf_vectors.npy   float32 matrix of document vectors
  metadata.json       list of {source, chunk_id, text}
  vectorizer.json     fitted TfidfVectorizer state (vocab + idf + params)

No pickle files are written or (except for a one-time migration of the
store's own legacy files) read from this directory: pickle.load() on data
that an attacker could plant would be remote code execution.
"""
from typing import List, Dict, Any
import json
import logging
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from src.rag.config import get_config

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """TF-IDF based vector store with cosine similarity."""

    VEC_PARAMS = {"max_features": 5000, "stop_words": "english", "ngram_range": (1, 2)}

    def __init__(self):
        self.vectorizer = TfidfVectorizer(**self.VEC_PARAMS)
        self.fitted = False

        cfg = get_config()
        self.db_path = Path(cfg["VECTOR_DB_PATH"])
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.vectors_path = self.db_path / "tfidf_vectors.npy"
        self.meta_path = self.db_path / "metadata.json"
        self.meta_legacy_path = self.db_path / "metadata.pkl"
        self.vec_path = self.db_path / "vectorizer.json"
        self.vec_legacy_path = self.db_path / "vectorizer.pkl"

        self.vectors = self._load_vectors()
        self.metadata = self._load_metadata()
        self._load_vectorizer()

    def _load_vectors(self) -> np.ndarray:
        """Load existing vectors or create empty array."""
        if self.vectors_path.exists():
            return np.load(self.vectors_path)
        return np.empty((0, 5000), dtype=np.float32)

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load metadata from disk (JSON; migrate legacy pickle once)."""
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception("Corrupt metadata.json; starting with empty metadata")
                return []
        if self.meta_legacy_path.exists():
            try:
                with open(self.meta_legacy_path, "rb") as f:
                    data = pickle.load(f)
                self.meta_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                self.meta_legacy_path.unlink()
                logger.info("Migrated metadata.pkl -> metadata.json")
                return data
            except Exception:
                logger.exception("Failed to migrate legacy metadata.pkl; ignoring it")
                return []
        return []

    def _load_vectorizer(self) -> None:
        """Load fitted vectorizer (JSON rebuild; migrate legacy pickle once)."""
        if self.vec_path.exists():
            try:
                state = json.loads(self.vec_path.read_text(encoding="utf-8"))
                if self._rebuild_vectorizer(state):
                    return
            except Exception:
                logger.exception("Failed to load vectorizer.json")
        if self.vec_legacy_path.exists():
            try:
                with open(self.vec_legacy_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                self.fitted = True
                self._save_vectorizer_state()
                self.vec_legacy_path.unlink()
                logger.info("Migrated vectorizer.pkl -> vectorizer.json")
            except Exception:
                logger.exception("Failed to migrate legacy vectorizer.pkl")
                self.vectorizer = TfidfVectorizer(**self.VEC_PARAMS)
                self.fitted = False

    def _rebuild_vectorizer(self, state: Dict[str, Any]) -> bool:
        """Reconstruct a fitted TfidfVectorizer from its persisted state."""
        try:
            params = dict(state.get("params") or self.VEC_PARAMS)
            params["ngram_range"] = tuple(params["ngram_range"])
            vec = TfidfVectorizer(**params)
            vec.vocabulary_ = {str(k): int(v) for k, v in state["vocab"].items()}
            vec.idf_ = np.asarray(state["idf"], dtype=np.float64)
            self.vectorizer = vec
            self.fitted = True
            return True
        except Exception:
            logger.exception("Rebuilding TF-IDF vectorizer failed; a re-ingest will be needed")
            self.vectorizer = TfidfVectorizer(**self.VEC_PARAMS)
            self.fitted = False
            return False

    def _save_vectorizer_state(self) -> None:
        state = {
            "params": {"max_features": 5000, "stop_words": "english", "ngram_range": [1, 2]},
            "vocab": {str(k): int(v) for k, v in self.vectorizer.vocabulary_.items()},
            "idf": np.asarray(self.vectorizer.idf_).tolist(),
        }
        self.vec_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    def _save(self) -> None:
        """Save vectors, vectorizer, and metadata to disk."""
        np.save(self.vectors_path, self.vectors)
        self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")
        self.meta_legacy_path.unlink(missing_ok=True)
        if self.fitted:
            self._save_vectorizer_state()
        else:
            # Never persist an unfitted vectorizer: __init__ treats the file's
            # existence as proof of training, which bricks the next ingest
            # with NotFittedError.
            self.vec_path.unlink(missing_ok=True)
        self.vec_legacy_path.unlink(missing_ok=True)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate TF-IDF vectors for texts."""
        if not self.fitted:
            # Fit on first batch
            vectors = self.vectorizer.fit_transform(texts)
            self.fitted = True
        else:
            vectors = self.vectorizer.transform(texts)
        return vectors.toarray().tolist()

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to vector store."""
        texts = [doc["text"] for doc in documents]
        metadatas = [
            {"source": doc["source"], "chunk_id": doc["chunk_id"], "text": doc["text"]}
            for doc in documents
        ]

        embeddings = np.array(self.embed(texts), dtype=np.float32)
        if self.vectors.size == 0:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])
        self.metadata.extend(metadatas)
        self._save()

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query similar documents using cosine similarity."""
        if self.vectors.size == 0 or not self.fitted:
            return []

        query_embedding = np.array(self.embed([query_text]), dtype=np.float32)
        # Normalize for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8)
        doc_norms = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)

        scores = doc_norms @ query_norm.T
        scores = scores.flatten()

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append({
                    "text": meta["text"],
                    "source": meta["source"],
                    "chunk_id": meta["chunk_id"],
                    "score": float(scores[idx])
                })
        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {"document_count": len(self.vectors)}

    def clear(self) -> None:
        """Clear all documents from index."""
        self.vectors = np.empty((0, 5000), dtype=np.float32)
        self.metadata = []
        self.fitted = False
        self.vectorizer = TfidfVectorizer(**self.VEC_PARAMS)
        self._save()
