"""Pydantic V2 schemas: validation rules, UUID/datetime round-trip, ORM mode."""
import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError as PydValidationError

from api import schemas


# ---------- UserCreate / UserRead ----------

def test_user_create_requires_email():
    with pytest.raises(PydValidationError):
        schemas.UserCreate(email="not-an-email")


def test_user_create_strips_invalid_display_name():
    with pytest.raises(PydValidationError):
        schemas.UserCreate(email="a@b.com", display_name="x/y")


def test_user_read_from_orm_attrs():
    class FakeUser:
        id = "abc-123"
        email = "a@b.com"
        display_name = "Alice"
        created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    u = schemas.UserRead.model_validate(FakeUser())
    assert u.id == "abc-123" and u.email == "a@b.com" and u.display_name == "Alice"


# ---------- WorkspaceCreate / WorkspaceRead / WorkspaceDetail ----------

def test_workspace_create_rejects_path_separators():
    with pytest.raises(PydValidationError):
        schemas.WorkspaceCreate(name="../etc")
    with pytest.raises(PydValidationError):
        schemas.WorkspaceCreate(name="a\\b")


def test_workspace_create_rejects_empty():
    with pytest.raises(PydValidationError):
        schemas.WorkspaceCreate(name="")
    with pytest.raises(PydValidationError):
        schemas.WorkspaceCreate(name="   ")


def test_workspace_read_round_trip():
    class FakeWS:
        id = "ws-1"
        name = "My Class"
        owner_id = "u-1"
        created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        members = []

    w = schemas.WorkspaceRead.model_validate(FakeWS())
    assert w.id == "ws-1" and w.name == "My Class"


# ---------- WorkspaceUserAdd ----------

def test_workspace_user_add_role_must_be_admin_or_member():
    schemas.WorkspaceUserAdd(email="a@b.com", role="admin")  # ok
    schemas.WorkspaceUserAdd(email="a@b.com", role="member")  # ok
    with pytest.raises(PydValidationError):
        schemas.WorkspaceUserAdd(email="a@b.com", role="superuser")


# ---------- DocumentCreate / DocumentRead ----------

def test_document_create_rejects_bad_sha256():
    with pytest.raises(PydValidationError):
        schemas.DocumentCreate(
            workspace_id="ws-1",
            filename="x.pdf",
            storage_path="ws-1/x.pdf",
            sha256="not-a-hash",
        )
    with pytest.raises(PydValidationError):
        schemas.DocumentCreate(
            workspace_id="ws-1",
            filename="x.pdf",
            storage_path="ws-1/x.pdf",
            sha256="A" * 64,  # right length but uppercase not allowed
        )


def test_document_create_rejects_path_traversal_in_filename():
    with pytest.raises(PydValidationError):
        schemas.DocumentCreate(
            workspace_id="ws-1",
            filename="../etc/passwd",
            storage_path="ws-1/x.pdf",
            sha256="a" * 64,
        )
    with pytest.raises(PydValidationError):
        schemas.DocumentCreate(
            workspace_id="ws-1",
            filename="x.pdf",
            storage_path="../escape.pdf",  # traversal blocked
            sha256="a" * 64,
        )


def test_document_create_accepts_good_input():
    d = schemas.DocumentCreate(
        workspace_id=str(uuid.uuid4()),
        filename="lecture.pdf",
        storage_path="ws-1/lecture.pdf",
        sha256="a" * 64,
        file_size=1234,
    )
    assert d.mime_type == "application/octet-stream"
    assert d.file_size == 1234


def test_document_read_from_orm():
    class FakeDoc:
        id = "d-1"
        workspace_id = "ws-1"
        uploaded_by = "u-1"
        filename = "x.pdf"
        storage_path = "ws-1/x.pdf"
        sha256 = "a" * 64
        file_size = 100
        mime_type = "application/pdf"
        status = "pending"
        error = None
        created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    d = schemas.DocumentRead.model_validate(FakeDoc())
    assert d.status == "pending" and d.error is None


# ---------- helpers ----------

def test_is_valid_uuid_accepts_and_rejects():
    assert schemas.is_valid_uuid(str(uuid.uuid4()))
    assert schemas.is_valid_uuid("00000000-0000-0000-0000-000000000000")
    assert not schemas.is_valid_uuid("not-a-uuid")
    assert not schemas.is_valid_uuid("")
    assert not schemas.is_valid_uuid(None)
