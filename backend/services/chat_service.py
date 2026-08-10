"""Service responsible for memory-aware conversations."""

from sqlmodel import Session, select

from core.huggingface_llm import HuggingFaceLLM
from models import Memory
from services.embedding_service import find_similar_memories


llm = HuggingFaceLLM()


def chat(
    user_message: str,
    user_id: int,
    session: Session,
) -> str:
    """
    Generate a response using the user's relevant memories.

    Args:
        user_message: Message sent by the user.
        user_id: ID of the current user.
        session: Database session.

    Returns:
        Generated response from the LLM.
    """

    search_results = find_similar_memories(
        query=user_message,
        user_id=user_id,
        top_k=5,
    )

    if not search_results:
        memory_context = "No relevant memories were found."

    else:
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

        memory_context = "\n".join(
            f"- {memory_lookup[result.memory_id].content}"
            for result in search_results
            if result.memory_id in memory_lookup
        )

    prompt = f"""
You are Continuum, a personal AI assistant with access to the user's
stored memories.

Use the stored memories when they are relevant to the user's message.
Do not invent personal information that is not present in the memories.

Stored memories:
{memory_context}

User message:
{user_message}

Answer naturally and concisely.
"""

    return llm.generate(prompt)