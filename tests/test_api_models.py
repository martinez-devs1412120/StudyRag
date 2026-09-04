"""Model layer: schema registered, pgvector Vector(384), cascade deletes declared."""
import re

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine
from sqlalchemy.schema import CreateTable

from api.models import Base, Chunk, Document, User, Workspace, WorkspaceUser


def test_all_tables_registered():
    assert sorted(Base.metadata.tables.keys()) == [
        "chunks", "documents", "users", "workspace_users", "workspaces",
    ]


def test_chunk_embedding_is_pgvector_384():
    col = Chunk.__table__.c.embedding
    assert isinstance(col.type, Vector)
    assert col.type.dim == 384
    assert col.nullable is False


def test_pks_have_uuid_default():
    for cls in (User, Workspace, Document, Chunk):
        pk = list(cls.__table__.primary_key.columns)[0]
        assert pk.default is not None


def test_cascade_deletes_compile_to_on_delete_cascade():
    engine = create_engine("sqlite:///:memory:")
    for tbl_name in ("workspace_users", "documents", "chunks"):
        ddl = str(CreateTable(Base.metadata.tables[tbl_name]).compile(engine))
        count = len(re.findall(r"ON DELETE CASCADE", ddl, re.IGNORECASE))
        assert count >= 2, f"{tbl_name} missing cascade deletes:\n{ddl}"


def test_workspace_relationships_wired():
    # Workspace has: members (WorkspaceUser m2m), documents (Document 1:N)
    assert "members" in Workspace.__mapper__.relationships
    assert "documents" in Workspace.__mapper__.relationships
    # Document has: workspace (back-ref), chunks
    assert "workspace" in Document.__mapper__.relationships
    assert "chunks" in Document.__mapper__.relationships
    # Chunk has: document (back-ref)
    assert "document" in Chunk.__mapper__.relationships


def test_documents_status_default_is_pending():
    col = Document.__table__.c.status
    assert col.default.arg == "pending"
