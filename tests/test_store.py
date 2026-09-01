"""Vector store: semantic retrieval, idempotency, persistence, legacy migration."""
import json
import pickle

from src.rag import embeddings as E
from tests.conftest import BIO, MATH


def test_semantic_search_beats_keywords(store_factory):
    store = store_factory()
    store.add_documents([BIO, MATH])
    # "sunlight" shares zero keywords with the bio chunk — only embeddings find it
    hits = store.query("how do plants use sunlight", top_k=2)
    assert hits[0]["source"] == "bio.pdf"
    assert 0 < hits[0]["score"] <= 1.0


def test_add_documents_is_idempotent(store_factory):
    store = store_factory()
    store.add_documents([BIO, MATH])
    store.add_documents([BIO, MATH])  # re-ingest same content
    assert store.count() == 2


def test_replace_source_swaps_content(store_factory):
    store = store_factory()
    store.add_documents([BIO])
    new_bio = dict(BIO, text="mitochondria produce ATP in eukaryotic cells")
    store.replace_source("bio.pdf", [new_bio])
    texts = [c["text"] for c in store.get_all()]
    assert texts == [new_bio["text"]]


def test_delete_source_and_clear(store_factory):
    store = store_factory()
    store.add_documents([BIO, MATH])
    store.delete_source("bio.pdf")
    assert store.count() == 1
    store.clear()
    assert store.count() == 0
    assert store.query("anything") == []


def test_survives_restart(store_factory, tmp_path):
    store = store_factory()
    store.add_documents([BIO, MATH])
    # fresh instance = app restart: same SQLite file, cache invalidated
    store2 = store_factory()
    assert store2.count() == 2
    assert store2.query("solving quadratic equations", top_k=1)[0]["source"] == "math.pdf"


def test_legacy_tfidf_json_migrates(store_factory, tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps([BIO, MATH]), encoding="utf-8")
    (tmp_path / "tfidf_vectors.npy").write_bytes(b"junk")
    store = store_factory()
    assert store.count() == 2
    assert not (tmp_path / "metadata.json").exists()
    assert not (tmp_path / "tfidf_vectors.npy").exists()


def test_legacy_tfidf_pkl_migrates(store_factory, tmp_path):
    (tmp_path / "metadata.pkl").write_bytes(pickle.dumps([BIO, MATH]))
    store = store_factory()
    assert store.count() == 2
    assert not (tmp_path / "metadata.pkl").exists()


def test_corrupt_legacy_files_do_not_break_boot(store_factory, tmp_path):
    (tmp_path / "metadata.json").write_text("{not json", encoding="utf-8")
    store = store_factory()
    assert store.count() == 0
