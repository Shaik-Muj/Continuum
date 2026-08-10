"""Services for generating, storing, and retrieving memory embeddings."""

from core.embedding import embedding_model
from core.chromadb import collection

from models import Memory, SearchResult


def store_embedding(
    memory_id: int,
    user_id: int,
    text: str
) -> None:
    """
    Generate an embedding for a memory and store it in ChromaDB.

    Args:
        memory_id: ID of the memory in PostgreSQL.
        user_id: ID of the memory owner.
        text: Memory content.
    """

    vector = embedding_model.embed([text])[0]

    collection.add(
        ids=[str(memory_id)],
        documents=[text],
        embeddings=[vector],
        metadatas=[
            {
                "memory_id": memory_id,
                "user_id": user_id,
            }
        ],
    )


def find_similar_memories(
    query: str,
    user_id: int,
    top_k: int = 5
) -> list[SearchResult]:
    """
    Find the most semantically similar memories for a user.

    Args:
        query: User search query.
        user_id: ID of the user performing the search.
        top_k: Maximum number of results.

    Returns:
        Ranked semantic search results.
    """

    query_embedding = embedding_model.embed(
        [query]
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"user_id": user_id},
    )

    search_results = []

    ids = results["ids"][0]
    distances = results["distances"][0]

    for memory_id, distance in zip(ids, distances):
        score = 1 - distance

        search_results.append(
            SearchResult(
                memory_id=int(memory_id),
                score=score,
            )
        )

    return search_results


def reindex_memories(
    memories: list[Memory]
) -> None:
    """
    Rebuild ChromaDB embeddings from PostgreSQL memories.

    This is useful when ChromaDB has been cleared or needs
    to be rebuilt from the PostgreSQL source of truth.

    Args:
        memories: List of Memory objects from PostgreSQL.
    """

    for memory in memories:
        store_embedding(
            memory_id=memory.id,
            user_id=memory.user_id,
            text=memory.content,
        )