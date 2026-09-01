import pytest
from unittest.mock import patch

from src.rag import embeddings as E

BIO = {"source": "bio.pdf", "chunk_id": 0,
       "text": "photosynthesis converts light energy into chemical energy inside plant cells"}
MATH = {"source": "math.pdf", "chunk_id": 0,
        "text": "the quadratic formula gives the roots of any quadratic equation"}


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    """EmbeddingStore pointed at a throwaway SQLite file."""
    def make():
        with patch.object(E, "get_config",
                          lambda: {"VECTOR_DB_PATH": str(tmp_path),
                                   "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"}):
            return E.EmbeddingStore()
    return make
