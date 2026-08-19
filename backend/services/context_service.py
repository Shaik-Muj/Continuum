"""Service for building portable context packages."""

from sqlmodel import Session, select

from models import ContextPackage, Memory
from services.embedding_service import find_similar_memories


def build_context(
    user_id: int,
    query: str,
    session: Session,
    top_k: int = 5,
) -> ContextPackage:
    """
    Build a portable context package from relevant user memories.

    Args:
        user_id: ID of the user requesting context.
        query: Current task, conversation, or question used to
            determine which memories are relevant.
        session: Database session.
        top_k: Maximum number of relevant memories to retrieve.

    Returns:
        ContextPackage containing the relevant memories.
    """

    search_results = find_similar_memories(
        query=query,
        user_id=user_id,
        top_k=top_k,
    )

    if not search_results:
        return ContextPackage()

    memory_ids = [
        result.memory_id
        for result in search_results
    ]

    memories = session.exec(
        select(Memory).where(
            Memory.id.in_(memory_ids),
            Memory.user_id == user_id,
        )
    ).all()

    memory_lookup = {
        memory.id: memory
        for memory in memories
    }

    relevant_memories = [
        memory_lookup[result.memory_id].content
        for result in search_results
        if result.memory_id in memory_lookup
    ]

    return ContextPackage(
        relevant_memories=relevant_memories,
    )