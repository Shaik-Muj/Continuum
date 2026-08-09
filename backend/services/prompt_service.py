"""Service for building prompts using retrieved memories."""

from models import Memory


def build_prompt(
    user_message: str,
    memories: list[Memory]
) -> str:
    """
    Build an LLM prompt using the user's message
    and relevant memories.

    Args:
        user_message: Current user message.
        memories: Relevant memories retrieved from Continuum.

    Returns:
        Prompt string to send to an LLM.
    """

    memory_context = ""

    for index, memory in enumerate(memories, start=1):
        memory_context += (
            f"[Memory {index}]\n"
            f"{memory.content}\n\n"
        )

    prompt = f"""You are Continuum, an AI assistant with access to
the user's long-term memory.

Use the memories below when they are relevant to the user's
current request.

Do not assume that a memory is relevant just because it exists.
If the memories do not contain useful information for answering
the request, answer based on the user's message alone.

RELEVANT MEMORIES:

{memory_context}

USER MESSAGE:

{user_message}
"""

    return prompt