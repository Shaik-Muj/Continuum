"""Cross-encoder reranker for semantic memory retrieval."""

from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-base"


reranker = CrossEncoder(
    MODEL_NAME,
    local_files_only=True,
)


def rerank(
    query: str,
    documents: list[str],
) -> list[float]:
    """
    Score documents for relevance to a query.

    Args:
        query: User's search query.
        documents: Candidate memories retrieved by semantic search.

    Returns:
        Relevance score for each document, in the same order.
    """

    if not documents:
        return []

    pairs = [
        [query, document]
        for document in documents
    ]

    scores = reranker.predict(pairs)

    return scores.tolist()