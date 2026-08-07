"""Memory service containing business logic for memory CRUD operations."""

from fastapi import HTTPException
from sqlmodel import Session, select

from models import Memory, MemoryCreate
from services.embedding_service import (
    store_embedding,
    find_similar_memories,
)


def create_memory(
    memory: MemoryCreate,
    user_id: int,
    session: Session
) -> Memory:
    """
    Create a new memory for a user.

    Args:
        memory: MemoryCreate model with content and source.
        user_id: ID of the user creating the memory.
        session: SQLModel session.

    Returns:
        Created Memory object.
    """

    db_memory = Memory(
        content=memory.content,
        source=memory.source,
        user_id=user_id
    )

    session.add(db_memory)
    session.commit()
    session.refresh(db_memory)

    # Store the embedding in ChromaDB
    store_embedding(
        memory_id=db_memory.id,
        text=db_memory.content
    )

    return db_memory


def get_user_memories(
    user_id: int,
    session: Session
) -> list[Memory]:
    """
    Retrieve all memories for a specific user.

    Args:
        user_id: ID of the user.
        session: SQLModel session.

    Returns:
        List of Memory objects belonging to the user.
    """

    memories = session.exec(
        select(Memory).where(
            Memory.user_id == user_id
        )
    ).all()

    return memories


def search_memories(
    q: str,
    session: Session
) -> list[Memory]:
    """
    Search memories using semantic similarity.

    Args:
        q: User search query.
        session: SQLModel session.

    Returns:
        List of semantically similar memories.
    """

    search_results = find_similar_memories(q)

    memories = []

    for result in search_results:

        memory = session.get(
            Memory,
            result.memory_id
        )

        if memory is not None:
            memories.append(memory)

    return memories


def get_memory(
    memory_id: int,
    user_id: int,
    session: Session
) -> Memory:
    """
    Retrieve a specific memory, with ownership check.

    Args:
        memory_id: ID of the memory.
        user_id: ID of the user requesting the memory.
        session: SQLModel session.

    Returns:
        Memory object if found and user is owner.

    Raises:
        HTTPException: If memory not found or user doesn't own it.
    """

    memory = session.get(Memory, memory_id)

    if memory is None:
        raise HTTPException(
            status_code=404,
            detail="Memory not found"
        )

    if memory.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )

    return memory


def delete_memory(
    memory_id: int,
    user_id: int,
    session: Session
) -> dict:
    """
    Delete a memory, with ownership check.

    Args:
        memory_id: ID of the memory to delete.
        user_id: ID of the user requesting deletion.
        session: SQLModel session.

    Returns:
        Dictionary with success message.

    Raises:
        HTTPException: If memory not found or user doesn't own it.
    """

    memory = session.get(Memory, memory_id)

    if memory is None:
        raise HTTPException(
            status_code=404,
            detail="Memory not found"
        )

    if memory.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )

    session.delete(memory)
    session.commit()

    return {
        "message": f"Memory {memory_id} deleted successfully"
    }