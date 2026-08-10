"""Memory router handling CRUD operations for user memories."""

from fastapi import APIRouter, Query, Depends
from sqlmodel import Session

from database import engine
from models import Memory, MemoryCreate, User
from auth import get_current_user
from services.memory_service import (
    create_memory,
    get_user_memories,
    search_memories,
    get_memory,
    delete_memory,
)

router = APIRouter(tags=["memory"])


def get_session():
    """Dependency to get database session."""
    with Session(engine) as session:
        yield session


@router.post("/memory", response_model=Memory)
def create_memory_endpoint(
    memory: MemoryCreate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Create a new memory for the current user.
    
    - **content**: Memory content/text
    - **source**: Source of the memory
    """
    return create_memory(memory, current_user.id, session)


@router.get("/memory", response_model=list[Memory])
def get_memories(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Retrieve all memories for the current user.
    """
    return get_user_memories(current_user.id, session)


@router.get("/search", response_model=list[Memory])
def search_memory_endpoint(
    q: str = Query(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Search memories by content.
    
    - **q**: Search query string
    """
    return search_memories(
    q,
    current_user.id,
    session
)


@router.get("/memory/{memory_id}", response_model=Memory)
def get_memory_endpoint(
    memory_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Retrieve a specific memory by ID (only if owned by current user).
    """
    return get_memory(memory_id, current_user.id, session)


@router.delete("/memory/{memory_id}")
def delete_memory_endpoint(
    memory_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Delete a memory by ID (only if owned by current user).
    """
    return delete_memory(memory_id, current_user.id, session)