from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    username: str = Field(index=True, unique=True)
    email: str = Field(unique=True)
    hashed_password: str

class UserRegister(SQLModel):
    username: str
    email: str
    password: str

class UserLogin(SQLModel):
    email: str
    password: str


class Memory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    user_id: int = Field(foreign_key="user.id")
    content: str
    source: str

    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Token(SQLModel):
    access_token: str
    token_type: str