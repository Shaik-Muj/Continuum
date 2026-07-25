"""Memory service containing business logic for memory CRUD operations."""

from fastapi import HTTPException
from sqlmodel import Session, select

from models import Memory, MemoryCreate


def create_memory(memory: MemoryCreate, user_id: int, session: Session) -> Memory:
    """
    Create a new memory for a user.
    
    Args:
        memory: MemoryCreate model with content and source
        user_id: ID of the user creating the memory
        session: SQLModel session
        
    Returns:
        Created Memory object
    """
    db_memory = Memory(
        content=memory.content,
        source=memory.source,
        user_id=user_id
    )
    
    session.add(db_memory)
    session.commit()
    session.refresh(db_memory)
    
    return db_memory


def get_user_memories(user_id: int, session: Session) -> list[Memory]:
    """
    Retrieve all memories for a specific user.
    
    Args:
        user_id: ID of the user
        session: SQLModel session
        
    Returns:
        List of Memory objects belonging to the user
    """
    memories = session.exec(
        select(Memory).where(Memory.user_id == user_id)
    ).all()
    
    return memories


def search_memories(q: str, session: Session) -> list[Memory]:
    """
    Search memories by content.
    
    Args:
        q: Search query string
        session: SQLModel session
        
    Returns:
        List of Memory objects matching the search query
    """
    memories = session.exec(
        select(Memory).where(Memory.content.contains(q))
    ).all()
    
    return memories


def get_memory(memory_id: int, user_id: int, session: Session) -> Memory:
    """
    Retrieve a specific memory, with ownership check.
    
    Args:
        memory_id: ID of the memory
        user_id: ID of the user requesting the memory
        session: SQLModel session
        
    Returns:
        Memory object if found and user is owner
        
    Raises:
        HTTPException: If memory not found or user doesn't own it
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


def delete_memory(memory_id: int, user_id: int, session: Session) -> dict:
    """
    Delete a memory, with ownership check.
    
    Args:
        memory_id: ID of the memory to delete
        user_id: ID of the user requesting deletion
        session: SQLModel session
        
    Returns:
        Dictionary with success message
        
    Raises:
        HTTPException: If memory not found or user doesn't own it
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
    
    return {"message": f"Memory {memory_id} deleted successfully"}
