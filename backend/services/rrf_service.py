"""Reciprocal Rank Fusion for combining retrieval rankings."""

from models import SearchResult


DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]],
    k: int = DEFAULT_RRF_K,
) -> list[SearchResult]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        rankings: Multiple ranked SearchResult lists.
        k: RRF constant used to reduce the impact of high ranks.

    Returns:
        Unified ranking ordered by RRF score.
    """

    scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.memory_id] = (
                scores.get(result.memory_id, 0.0)
                + 1 / (k + rank)
            )

    fused_results = [
        SearchResult(
            memory_id=memory_id,
            score=score,
        )
        for memory_id, score in scores.items()
    ]

    fused_results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return fused_results