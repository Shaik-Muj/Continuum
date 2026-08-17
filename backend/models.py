"""Database models and API schemas for Continuum."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    """Database model representing a Continuum user."""

    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    username: str = Field(
        index=True,
        unique=True,
    )

    email: str = Field(
        unique=True,
    )

    hashed_password: str


class UserRegister(SQLModel):
    """Request model used when registering a new user."""

    username: str
    email: str
    password: str


class UserLogin(SQLModel):
    """Request model used when logging in."""

    email: str
    password: str


class Memory(SQLModel, table=True):
    """Database model representing a user's stored memory."""

    id: Optional[int] = Field(
        default=None,
        primary_key=True,
    )

    user_id: int = Field(
        foreign_key="user.id",
    )

    content: str
    source: str

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
    )


class MemoryCreate(SQLModel):
    """Request model used when creating a new memory."""

    content: str
    source: str


class Token(SQLModel):
    """Authentication token response."""

    access_token: str
    token_type: str


class SearchResult(BaseModel):
    """Represents a semantic search result."""

    memory_id: int
    score: float


class ContextPackage(BaseModel):
    """
    Portable representation of a user's working context.

    This model is independent of any specific LLM provider.
    """

    project: str | None = None
    objective: str | None = None
    current_state: str | None = None

    decisions: list[str] = Field(
        default_factory=list,
    )

    constraints: list[str] = Field(
        default_factory=list,
    )

    relevant_memories: list[str] = Field(
        default_factory=list,
    )

    open_tasks: list[str] = Field(
        default_factory=list,
    )