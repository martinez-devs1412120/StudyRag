"""Embeddings and vector store using TF-IDF (pure Python, no native deps)."""
from typing import List, Dict, Any
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from src.rag.config import get_config


class EmbeddingStore:
    """TF-IDF based vector store with cosine similarity."""

    def __init__(self):
        cfg = get_config()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.fitted = False

        self.db_path = Path(cfg["VECTOR_DB_PATH"])
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.vectors_path = self.db_path / "tfidf_vectors.npy"
        self.vectorizer_path = self.db_path / "vectorizer.pkl"
        self.meta_path = self.db_path / "metadata.pkl"

        self.vectors = self._load_vectors()
        self.metadata = self._load_metadata()
        self._load_vectorizer()

    def _load_vectors(self) -> np.ndarray:
        """Load existing vectors or create empty array."""
        if self.vectors_path.exists():
            return np.load(self.vectors_path)
        return np.empty((0, 5000), dtype=np.float32)

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load metadata from disk."""
        if self.meta_path.exists():
            with open(self.meta_path, "rb") as f:
                return pickle.load(f)
        return []

    def _load_vectorizer(self) -> None:
        """Load fitted vectorizer."""
        if self.vectorizer_path.exists():
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            self.fitted = True

    def _save(self) -> None:
        """Save vectors, vectorizer, and metadata to disk."""
        np.save(self.vectors_path, self.vectors)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

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
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self._save()