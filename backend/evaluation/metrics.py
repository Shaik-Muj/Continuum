"""Ranking metrics for retrieval evaluation."""


def first_relevant_rank(
    ranked_ids: list[int],
    relevant_ids: set[int],
) -> int | None:
    """
    Find the 1-based rank of the first relevant memory.

    Args:
        ranked_ids: Memory IDs in ranked order.
        relevant_ids: Ground-truth relevant memory IDs.

    Returns:
        Rank of the first relevant memory, or None if none was retrieved.
    """

    for rank, memory_id in enumerate(ranked_ids, start=1):
        if memory_id in relevant_ids:
            return rank

    return None


def reciprocal_rank(
    ranked_ids: list[int],
    relevant_ids: set[int],
) -> float:
    """
    Compute the reciprocal rank of the first relevant memory.

    Averaging this over queries gives MRR (Mean Reciprocal Rank).

    Args:
        ranked_ids: Memory IDs in ranked order.
        relevant_ids: Ground-truth relevant memory IDs.

    Returns:
        1 / rank of the first relevant memory, or 0.0 if none was retrieved.
    """

    rank = first_relevant_rank(ranked_ids, relevant_ids)

    return 0.0 if rank is None else 1 / rank


def recall_at_k(
    ranked_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    """
    Compute capped Recall@k.

    The denominator is min(k, number of relevant memories), so a query
    with more relevant memories than k can still reach 1.0. This makes
    Recall@1 equal to "is the top result relevant?" for every query.

    Args:
        ranked_ids: Memory IDs in ranked order.
        relevant_ids: Ground-truth relevant memory IDs.
        k: Cutoff rank.

    Returns:
        Fraction of the achievable relevant memories found in the top k.
    """

    if not relevant_ids:
        raise ValueError("Recall is undefined for a query with no relevant memories.")

    hits = len(set(ranked_ids[:k]) & relevant_ids)

    return hits / min(k, len(relevant_ids))
