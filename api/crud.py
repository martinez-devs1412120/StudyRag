"""CRUD utilities for the API. Thin: one function per action, no implicit
transactions, raises specific exceptions so route handlers can map them
to HTTP codes (Task 7 will wire them up).

All functions take a SQLAlchemy Session explicitly — never import the
session module here. That keeps CRUD testable with an in-memory SQLite
session and free of engine-management code.
"""
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api import models, schemas

logger = logging.getLogger(__name__)


# ---------- exceptions ----------

class NotFoundError(Exception):
    """Raised when a referenced row doesn't exist."""


class ConflictError(Exception):
    """Raised on a unique-constraint or idempotency violation."""


class ValidationError(Exception):
    """Raised when input fails business validation (e.g. wrong role)."""


# ---------- users ----------

def get_or_create_user(db: Session, *, email: str, display_name: Optional[str] = None) -> models.User:
    """Find an existing user by email, or create one. Case-insensitive email
    match (emails are stored lowercased here for deterministic lookups)."""
    email = email.strip().lower()
    existing = db.execute(
        select(models.User).where(models.User.email == email)
    ).scalar_one_or_none()
    if existing:
        return existing
    user = models.User(email=email, display_name=display_name)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # Another request created the same user between SELECT and INSERT.
        db.rollback()
        existing = db.execute(
            select(models.User).where(models.User.email == email)
        ).scalar_one()
        return existing
    db.refresh(user)
    return user


def get_user(db: Session, user_id: str) -> Optional[models.User]:
    return db.get(models.User, user_id)


# ---------- workspaces ----------

def create_workspace(db: Session, *, owner: models.User, name: str) -> models.Workspace:
    """Create a workspace and add the owner as the first 'admin' member.

    Returns the persisted Workspace with members loaded.
    """
    name = _check_name(name)
    ws = models.Workspace(name=name, owner_id=owner.id)
    db.add(ws)
    db.flush()  # populate ws.id before creating the membership row
    db.add(models.WorkspaceUser(workspace_id=ws.id, user_id=owner.id, role="admin"))
    db.commit()
    db.refresh(ws)
    return ws


def get_workspace(db: Session, workspace_id: str) -> Optional[models.Workspace]:
    return db.get(models.Workspace, workspace_id)


def list_user_workspaces(db: Session, *, user_id: str) -> List[models.Workspace]:
    """All workspaces the user is a member of, by way of the WorkspaceUser link."""
    stmt = (
        select(models.Workspace)
        .join(models.WorkspaceUser, models.WorkspaceUser.workspace_id == models.Workspace.id)
        .where(models.WorkspaceUser.user_id == user_id)
        .order_by(models.Workspace.created_at.desc())
    )
    return list(db.execute(stmt).scalars())


def add_user_to_workspace(
    db: Session,
    *,
    workspace_id: str,
    email: str,
    role: str = "member",
) -> models.WorkspaceUser:
    """Add an existing (or newly-provisioned) user to a workspace.

    Idempotent: re-adding the same user returns the existing membership
    row instead of failing on the composite-PK constraint.
    """
    if role not in ("admin", "member"):
        raise ValidationError(f"role must be 'admin' or 'member', got {role!r}")
    if not get_workspace(db, workspace_id):
        raise NotFoundError(f"workspace {workspace_id} not found")
    user = get_or_create_user(db, email=email)
    existing = db.execute(
        select(models.WorkspaceUser).where(
            models.WorkspaceUser.workspace_id == workspace_id,
            models.WorkspaceUser.user_id == user.id,
        )
    ).scalar_one_or_none()
    if existing:
        return existing
    membership = models.WorkspaceUser(workspace_id=workspace_id, user_id=user.id, role=role)
    db.add(membership)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.execute(
            select(models.WorkspaceUser).where(
                models.WorkspaceUser.workspace_id == workspace_id,
                models.WorkspaceUser.user_id == user.id,
            )
        ).scalar_one()
        return existing
    db.refresh(membership)
    return membership


# ---------- documents ----------

def create_document_record(
    db: Session,
    *,
    workspace_id: str,
    uploaded_by: str,
    filename: str,
    storage_path: str,
    sha256: str,
    file_size: int = 0,
    mime_type: str = "application/octet-stream",
) -> models.Document:
    """Insert a 'pending' document. Caller must verify workspace membership
    and file hash before calling this."""
    if not get_workspace(db, workspace_id):
        raise NotFoundError(f"workspace {workspace_id} not found")
    if not get_user(db, uploaded_by):
        raise NotFoundError(f"user {uploaded_by} not found")
    sha256 = sha256.lower()
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValidationError("sha256 must be a 64-char lowercase hex string")
    if "/" in filename or "\\" in filename or "\x00" in filename:
        raise ValidationError("filename must not contain path separators or NUL bytes")
    doc = models.Document(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        uploaded_by=uploaded_by,
        filename=filename,
        storage_path=storage_path,
        sha256=sha256,
        file_size=file_size,
        mime_type=mime_type,
        status="pending",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def list_workspace_documents(
    db: Session, *, workspace_id: str, limit: int = 100
) -> List[models.Document]:
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_id == workspace_id)
        # id is a deterministic tiebreaker — without it, two rows with the
        # same created_at (common on SQLite which stores seconds) are ordered
        # non-deterministically, breaking pagination.
        .order_by(models.Document.created_at.desc(), models.Document.id.desc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars())


# ---------- helpers ----------

def _check_name(name: str) -> str:
    name = (name or "").strip()
    if not name or len(name) > 255 or "/" in name or "\\" in name or "\x00" in name:
        raise ValidationError("name must be 1-255 chars and must not contain path separators")
    return name
