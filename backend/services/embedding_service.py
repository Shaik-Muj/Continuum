"""Services for generating embeddings and performing semantic retrieval."""

from core.embedding import embedding_model
from core.chromadb import collection
from models import Memory, SearchResult


DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.4


def store_embedding(
    memory_id: int,
    user_id: int,
    text: str,
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
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[SearchResult]:
    """
    Find semantically similar memories belonging to a specific user.

    Results below the similarity threshold are discarded.

    Args:
        query: Search query.
        user_id: ID of the user.
        top_k: Maximum number of results to return.
        threshold: Minimum similarity score required.

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

    if not results["ids"]:
        return []

    ids = results["ids"][0]
    distances = results["distances"][0]

    search_results: list[SearchResult] = []

    for memory_id, distance in zip(ids, distances):
        score = 1 - distance

        if score < threshold:
            continue

        search_results.append(
            SearchResult(
                memory_id=int(memory_id),
                score=score,
            )
        )

    return search_results


def reindex_memories(memories: list[Memory]) -> None:
    """
    Rebuild ChromaDB embeddings from PostgreSQL memories.

    Args:
        memories: Memories retrieved from PostgreSQL.
    """

    if not memories:
        return

    for memory in memories:
        store_embedding(
            memory_id=memory.id,
            user_id=memory.user_id,
            text=memory.content,
        )