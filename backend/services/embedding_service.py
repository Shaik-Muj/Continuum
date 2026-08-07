from core.embedding import embedding_model
from core.chromadb import collection

from models import SearchResult


def store_embedding(
    memory_id: int,
    text: str
) -> None:
    """
    Generate an embedding for a memory and store it in ChromaDB.
    """

    vector = embedding_model.embed([text])[0]

    collection.add(
        ids=[str(memory_id)],
        documents=[text],
        embeddings=[vector],
        metadatas=[
            {
                "memory_id": memory_id
            }
        ]
    )


def find_similar_memories(
    query: str,
    top_k: int = 5
) -> list[SearchResult]:
    """
    Find the most semantically similar memories.

    Args:
        query: User search query.
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
    )

    search_results = []

    ids = results["ids"][0]
    distances = results["distances"][0]

    for memory_id, distance in zip(ids, distances):

        score = 1 - distance

        search_results.append(
            SearchResult(
                memory_id=int(memory_id),
                score=score
            )
        )

    return search_results