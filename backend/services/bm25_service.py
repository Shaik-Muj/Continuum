"""BM25 lexical retrieval service for Continuum memories."""

import re

from rank_bm25 import BM25Okapi

from models import Memory, SearchResult


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 retrieval.

    Args:
        text: Input text.

    Returns:
        Lowercase word tokens.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def search_bm25(
    query: str,
    memories: list[Memory],
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Search memories using BM25 lexical matching.

    Args:
        query: User's search query.
        memories: User's memories from PostgreSQL.
        top_k: Maximum number of results.

    Returns:
        Ranked BM25 results.
    """

    if not memories:
        return []

    documents = [
        memory.content
        for memory in memories
    ]

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        zip(memories, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    results = []

    for memory, score in ranked[:top_k]:
        results.append(
            SearchResult(
                memory_id=memory.id,
                score=float(score),
            )
        )

    return results