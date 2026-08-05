from core.embedding import embedding_model
from core.chromadb import collection


def store_embedding(
    memory_id: int,
    text: str
) -> None:
    """
    Generate an embedding for a memory and store it in ChromaDB.

    Args:
        memory_id: PostgreSQL memory ID.
        text: Memory content.
    """

    vector = embedding_model.embed([text])[0]

    collection.add(
        ids=[str(memory_id)],
        documents=[text],
        embeddings=[vector],
    )