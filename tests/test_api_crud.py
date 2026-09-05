"""CRUD: exercises create_workspace, add_user_to_workspace, create_document_record
against an in-memory SQLite session. Vector columns are skipped (pgvector
is Postgres-only) but the rest of the schema is exercised end-to-end.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import api.crud as crud
import api.models as models


@pytest.fixture
def db():
    # SQLite for tests. The chunks table uses pgvector's VECTOR(384) which
    # SQLite can't represent, so we create a stub table with a plain BLOB
    # column — the CRUD tests never insert into it, but a stub is required
    # because the Document model's `chunks` relationship has lazy="selectin"
    # and refreshes trigger an auto-SELECT.
    from sqlalchemy import Column, String, Text, event

    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _fk_pragma_on_connect(dbapi_con, _):
        cur = dbapi_con.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    models.Base.metadata.create_all(engine)

    # Replace the chunks table with a stub. SQLAlchemy creates tables from
    # metadata; we drop and recreate with a BLOB instead of VECTOR(384).
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS chunks")
        conn.exec_driver_sql(
            "CREATE TABLE chunks ("
            " id TEXT PRIMARY KEY,"
            " document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,"
            " workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,"
            " chunk_index INTEGER NOT NULL,"
            " text TEXT NOT NULL,"
            " embedding BLOB NOT NULL,"
            " created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL"
            ")"
        )

    SessionLocal = sessionmaker(bind=engine)
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def test_get_or_create_user_creates_then_returns_same(db):
    u1 = crud.get_or_create_user(db, email="Alice@Example.com")
    u2 = crud.get_or_create_user(db, email="alice@example.com")
    assert u1.id == u2.id
    assert u1.email == "alice@example.com"  # stored lowercased


def test_create_workspace_adds_owner_as_admin(db):
    owner = crud.get_or_create_user(db, email="owner@example.com")
    ws = crud.create_workspace(db, owner=owner, name="My Class")
    assert ws.owner_id == owner.id
    # Owner is the first admin
    members = ws.members
    assert len(members) == 1
    assert members[0].user_id == owner.id
    assert members[0].role == "admin"


def test_create_workspace_rejects_invalid_name(db):
    owner = crud.get_or_create_user(db, email="o@example.com")
    with pytest.raises(crud.ValidationError):
        crud.create_workspace(db, owner=owner, name="   ")
    with pytest.raises(crud.ValidationError):
        crud.create_workspace(db, owner=owner, name="a/b")


def test_add_user_to_workspace_idempotent(db):
    owner = crud.get_or_create_user(db, email="o@example.com")
    ws = crud.create_workspace(db, owner=owner, name="W")
    m1 = crud.add_user_to_workspace(db, workspace_id=ws.id, email="bob@example.com")
    m2 = crud.add_user_to_workspace(db, workspace_id=ws.id, email="bob@example.com")
    assert m1.user_id == m2.user_id
    assert m1.workspace_id == m2.workspace_id
    assert len(ws.members) == 2  # owner + bob


def test_add_user_to_workspace_provisions_user(db):
    owner = crud.get_or_create_user(db, email="o@example.com")
    ws = crud.create_workspace(db, owner=owner, name="W")
    m = crud.add_user_to_workspace(db, workspace_id=ws.id, email="new@example.com", role="admin")
    assert m.role == "admin"
    # New user now exists in the users table
    assert crud.get_user(db, m.user_id) is not None


def test_add_user_to_workspace_rejects_bad_role(db):
    owner = crud.get_or_create_user(db, email="o@example.com")
    ws = crud.create_workspace(db, owner=owner, name="W")
    with pytest.raises(crud.ValidationError):
        crud.add_user_to_workspace(db, workspace_id=ws.id, email="x@y.com", role="superuser")


def test_add_user_to_workspace_404_on_missing_ws(db):
    with pytest.raises(crud.NotFoundError):
        crud.add_user_to_workspace(db, workspace_id="nope", email="x@y.com")


def test_create_document_record_pending(db):
    owner = crud.get_or_create_user(db, email="o@example.com")
    ws = crud.create_workspace(db, owner=owner, name="W")
    doc = crud.create_document_record(
        db,
        workspace_id=ws.id,
        uploaded_by=owner.id,
        filename="lecture.pdf",
        storage_path=f"{ws.id}/lecture.pdf",
        sha256="a" * 64,
        file_size=42,
    )
    assert doc.status == "pending"
    assert doc.workspace_id == ws.id
    assert doc.uploaded_by == owner.id
    assert doc.sha256 == "a" * 64


def test_create_document_rejects_path_traversal(db):
    owner = crud.get_or_create_user(db, email="o@example.com")
    ws = crud.create_workspace(db, owner=owner, name="W")
    with pytest.raises(crud.ValidationError):
        crud.create_document_record(
            db,
            workspace_id=ws.id,
            uploaded_by=owner.id,
            filename="../etc/passwd",
            storage_path="x",
            sha256="a" * 64,
        )


def test_create_document_rejects_nonhex_sha256(db):
    owner = crud.get_or_create_user(db, email="o@example.com")
    ws = crud.create_workspace(db, owner=owner, name="W")
    with pytest.raises(crud.ValidationError):
        crud.create_document_record(
            db,
            workspace_id=ws.id,
            uploaded_by=owner.id,
            filename="x.pdf",
            storage_path="x",
            sha256="z" * 64,  # not hex
        )


def test_create_document_404_on_missing_workspace(db):
    with pytest.raises(crud.NotFoundError):
        crud.create_document_record(
            db,
            workspace_id="nope",
            uploaded_by="nope",
            filename="x.pdf",
            storage_path="x",
            sha256="a" * 64,
        )


def test_list_user_workspaces_returns_only_member_of(db):
    owner = crud.get_or_create_user(db, email="o@example.com")
    other = crud.get_or_create_user(db, email="p@example.com")
    ws_a = crud.create_workspace(db, owner=owner, name="A")
    crud.create_workspace(db, owner=other, name="B")  # owner is 'other', not 'owner'

    visible = crud.list_user_workspaces(db, user_id=owner.id)
    assert [w.id for w in visible] == [ws_a.id]


def test_list_workspace_documents_orders_newest_first(db):
    owner = crud.get_or_create_user(db, email="o@example.com")
    ws = crud.create_workspace(db, owner=owner, name="W")
    d1 = crud.create_document_record(db, workspace_id=ws.id, uploaded_by=owner.id,
                                     filename="a.pdf", storage_path="a", sha256="a" * 64)
    d2 = crud.create_document_record(db, workspace_id=ws.id, uploaded_by=owner.id,
                                     filename="b.pdf", storage_path="b", sha256="b" * 64)
    docs = crud.list_workspace_documents(db, workspace_id=ws.id)
    assert [d.filename for d in docs] == ["b.pdf", "a.pdf"]
