"""Pydantic V2 schemas for the API surface.

Conventions:
- "Create" schemas: fields the client supplies (no id, no timestamps).
- "Read" schemas: server-owned fields too (id, created_at, updated_at).
- All ids are UUID4 strings; all timestamps are timezone-aware datetimes.
- ORM-friendly: every Read schema has `model_config = ConfigDict(from_attributes=True)`
  so route handlers can return ORM objects directly.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# Used by the chat/retrieval work (Tasks 12-14). Kept here so all API
# shapes live in one place.
class Source(BaseModel):
    source: str
    chunk_id: int
    score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=400)
    top_k: int = Field(5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    relevant: bool = True


# ---------- shared ----------

# Names: non-empty, trimmed, capped. Path-like characters are rejected so
# a hostile name can't break storage_path later.
_NAME_RE = re.compile(r"^[^/\\\x00]{1,255}$")


def _check_name(name: str, field: str) -> str:
    if not isinstance(name, str):
        raise ValueError(f"{field} must be a string")
    name = name.strip()
    if not _NAME_RE.match(name):
        raise ValueError(f"{field} must be 1-255 chars and must not contain '/', '\\', or NUL")
    return name


# ---------- User ----------

class UserCreate(BaseModel):
    email: EmailStr
    display_name: Optional[str] = Field(None, max_length=255)

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return _check_name(v, "display_name")


class UserRead(BaseModel):
    id: str
    email: EmailStr
    display_name: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------- Workspace ----------

class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        return _check_name(v, "name")


class WorkspaceRead(BaseModel):
    id: str
    name: str
    owner_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WorkspaceMemberRead(BaseModel):
    user_id: str
    role: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WorkspaceDetail(WorkspaceRead):
    """Workspace with its members included — used by GET /workspaces/{id}."""
    members: List[WorkspaceMemberRead] = []


# ---------- WorkspaceUser (membership) ----------

class WorkspaceUserAdd(BaseModel):
    email: EmailStr
    role: str = Field("member", pattern="^(admin|member)$")


# ---------- Document ----------

class DocumentCreate(BaseModel):
    """The server-computed fields (id, sha256, status, timestamps) are NOT
    accepted from clients — they're set by the upload endpoint / worker."""
    workspace_id: str
    filename: str
    storage_path: str
    sha256: str = Field(..., min_length=64, max_length=64, pattern=r"^[a-f0-9]+$")
    file_size: int = Field(0, ge=0)
    mime_type: str = Field("application/octet-stream", max_length=128)

    @field_validator("filename")
    @classmethod
    def _filename_no_traversal(cls, v: str) -> str:
        # filename is the user-facing display name: no separators at all.
        if "/" in v or "\\" in v or "\x00" in v or ".." in v:
            raise ValueError("filename must not contain path separators, NUL bytes, or '..'")
        return v

    @field_validator("storage_path")
    @classmethod
    def _storage_path_no_traversal(cls, v: str) -> str:
        # storage_path is server-built; forward slashes ARE expected. Only
        # block path traversal ('..') and NUL bytes.
        if ".." in v.split("/") or "\x00" in v:
            raise ValueError("storage_path must not contain '..' or NUL bytes")
        return v


class DocumentRead(BaseModel):
    id: str
    workspace_id: str
    uploaded_by: str
    filename: str
    storage_path: str
    sha256: str
    file_size: int
    mime_type: str
    status: str
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------- Chunk ----------

class ChunkRead(BaseModel):
    id: str
    document_id: str
    workspace_id: str
    chunk_index: int
    text: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------- helpers ----------

def is_valid_uuid(value: str) -> bool:
    """Cheap UUID4 shape check used by path-param validators."""
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False
