"""SQLAlchemy 2.0 ORM models for StudyRAG.

Schema:
  workspaces <-- workspace_users --> users
       |
       v
   documents
       |
       v
     chunks  (pgvector embeddings, 384 dims)

Notes:
- UUIDs as PKs (text columns, portable, no native-UUID dependency).
- pgvector's Vector type stores dense floats; dim 384 matches the
  all-MiniLM-L6-v2 model used by the existing src/rag/embeddings.py.
- Cascade deletes: removing a workspace removes its documents and
  chunks; removing a document removes its chunks. WorkspaceUser rows
  are removed when either side is deleted.
- Relationships are bidirectional via selectinload-friendly lazy='selectin'
  so multi-tenant queries don't N+1.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _new_id() -> str:
    """String UUID4 — stored as TEXT for portability across SQLite/Postgres."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Python-side UTC now (microsecond precision, portable across dialects)."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Project-wide declarative base. All ORM models inherit from this."""


# ---------- core multi-tenancy ----------

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    workspaces: Mapped[List["WorkspaceUser"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    members: Mapped[List["WorkspaceUser"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan", lazy="selectin"
    )
    documents: Mapped[List["Document"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan", lazy="selectin"
    )


class WorkspaceUser(Base):
    """Many-to-many membership with a role column. Composite PK = (workspace_id, user_id)."""

    __tablename__ = "workspace_users"
    __table_args__ = (UniqueConstraint("workspace_id", "user_id", name="uq_workspace_user"),)

    workspace_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    # "admin" can manage members + documents; "member" can only query.
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="member")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    workspace: Mapped["Workspace"] = relationship(back_populates="members")
    user: Mapped["User"] = relationship(back_populates="workspaces")


# ---------- documents & embeddings ----------

# Document lifecycle: pending -> parsing -> chunking -> embedding -> completed
#                    \-> failed
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    workspace_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    uploaded_by: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False, default="application/octet-stream")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    workspace: Mapped["Workspace"] = relationship(back_populates="documents")
    chunks: Mapped[List["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", lazy="selectin"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Denormalized workspace_id on every chunk makes the per-workspace
    # vector search a single indexed filter (no join required).
    workspace_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")
