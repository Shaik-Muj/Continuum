"""
Offline retrieval evaluation for Continuum.

Compares BM25, BGE, Reciprocal Rank Fusion and cross-encoder reranking
on a controlled memory corpus with known relevant memories per query.

Nothing touches PostgreSQL or ChromaDB. The corpus is loaded from JSON
and every retriever runs in memory, reusing the production code
(services.bm25_service, services.rrf_service, core.embedding,
core.reranker). BGE scores are exact cosine similarities between
normalized embeddings: the ranking Chroma produces with
hnsw:space=cosine (HNSW is approximate, but exact at this corpus size).

Usage, from backend/:
    .continuum\\Scripts\\python.exe evaluation\\evaluate_retrieval.py
    .continuum\\Scripts\\python.exe evaluation\\evaluate_retrieval.py --per-query --output results.json
"""

import os

# Evaluation must only use models that are already in the local HF cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.embedding import embedding_model
from core.reranker import MODEL_NAME as RERANKER_MODEL_NAME, rerank
from evaluation.metrics import first_relevant_rank, recall_at_k, reciprocal_rank
from models import Memory, SearchResult
from services.bm25_service import search_bm25, tokenize
from services.rrf_service import DEFAULT_RRF_K, reciprocal_rank_fusion


DEFAULT_DATASET = Path(__file__).with_name("retrieval_dataset.json")

# Query prefix recommended by the BGE v1.5 model card for short queries
# against passages. Memories are embedded without it.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

RECALL_CUTOFFS = (1, 3, 5)
BASELINE_PIPELINE = "bge"

PIPELINES = {
    "bm25": "BM25 over all memories",
    "bge": "BGE cosine, as used in production today",
    "bge_instr": "BGE cosine with the BGE query instruction",
    "rrf": "RRF over BGE and BM25 candidates",
    "rrf_instr": "RRF over instructed-BGE and BM25 candidates",
    "rrf_rerank": "RRF candidates reordered by the cross-encoder",
    "rerank_all": "Cross-encoder over every memory (diagnostic)",
}

# Scored retrievers whose raw scores are inspected for threshold calibration.
SCORED_PIPELINES = ("bge", "bge_instr", "rerank_all")

# Function words ignored when deciding whether a query shares vocabulary
# with its relevant memories. Only used to split the report into
# lexical and paraphrase queries; retrieval itself is unaffected.
STOPWORDS = {
    "a", "am", "an", "and", "any", "are", "at", "be", "by", "can", "d",
    "did", "do", "does", "don", "for", "from", "get", "have", "how", "i",
    "in", "is", "it", "m", "me", "my", "of", "on", "or", "s", "should",
    "t", "the", "to", "what", "when", "where", "which", "who", "why",
    "will", "with", "yet", "you", "your",
}


def load_dataset(path: Path) -> dict:
    """
    Load the evaluation dataset and check that its labels are consistent.

    Args:
        path: Path to the dataset JSON file.

    Returns:
        Parsed dataset.

    Raises:
        ValueError: If IDs are duplicated or labels reference unknown memories.
    """

    dataset = json.loads(path.read_text(encoding="utf-8"))

    memory_ids = [memory["id"] for memory in dataset["memories"]]
    if len(memory_ids) != len(set(memory_ids)):
        raise ValueError("Duplicate memory IDs in dataset.")

    all_queries = dataset["queries"] + dataset["no_answer_queries"]
    query_ids = [query["id"] for query in all_queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query IDs in dataset.")

    known_ids = set(memory_ids)

    for query in dataset["queries"]:
        if not query["relevant"]:
            raise ValueError(
                f"{query['id']} has no relevant memories; "
                "move it to no_answer_queries."
            )

        unknown_ids = set(query["relevant"]) - known_ids
        if unknown_ids:
            raise ValueError(
                f"{query['id']} references unknown memories {sorted(unknown_ids)}."
            )

    return dataset


def rank_by_score(
    memory_ids: list[int],
    scores: list[float],
) -> list[SearchResult]:
    """
    Sort memories by score, highest first.

    Args:
        memory_ids: Memory IDs.
        scores: Score for each memory, in the same order.

    Returns:
        Ranked search results.
    """

    ranked = sorted(
        zip(memory_ids, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        SearchResult(memory_id=memory_id, score=float(score))
        for memory_id, score in ranked
    ]


def retrieve(
    query: str,
    memories: list[Memory],
    memory_vectors: np.ndarray,
    candidates: int,
    rerank_pool: int,
) -> dict[str, list[SearchResult]]:
    """
    Run every pipeline for one query.

    Args:
        query: Query text.
        memories: Corpus memories.
        memory_vectors: Normalized BGE embeddings, one row per memory.
        candidates: Results taken from each retriever before fusion.
        rerank_pool: Fused results passed to the reranker.

    Returns:
        Ranking produced by each pipeline, best first.
    """

    memory_ids = [memory.id for memory in memories]

    bm25 = search_bm25(
        query=query,
        memories=memories,
        top_k=len(memories),
    )

    plain_vector, instructed_vector = embedding_model.embed(
        [query, BGE_QUERY_INSTRUCTION + query]
    )

    bge = rank_by_score(memory_ids, memory_vectors @ np.asarray(plain_vector))
    bge_instr = rank_by_score(memory_ids, memory_vectors @ np.asarray(instructed_vector))

    rrf = reciprocal_rank_fusion(
        rankings=[bge[:candidates], bm25[:candidates]],
    )
    rrf_instr = reciprocal_rank_fusion(
        rankings=[bge_instr[:candidates], bm25[:candidates]],
    )

    # Cross-encoder scores are independent per (query, memory) pair, so
    # scoring every memory once is equivalent to reranking any subset.
    rerank_scores = dict(
        zip(
            memory_ids,
            rerank(
                query=query,
                documents=[memory.content for memory in memories],
            ),
        )
    )

    pool_ids = [result.memory_id for result in rrf[:rerank_pool]]

    rrf_rerank = rank_by_score(
        pool_ids,
        [rerank_scores[memory_id] for memory_id in pool_ids],
    )
    rerank_all = rank_by_score(
        memory_ids,
        [rerank_scores[memory_id] for memory_id in memory_ids],
    )

    return {
        "bm25": bm25,
        "bge": bge,
        "bge_instr": bge_instr,
        "rrf": rrf,
        "rrf_instr": rrf_instr,
        "rrf_rerank": rrf_rerank,
        "rerank_all": rerank_all,
    }


def query_type(
    query: str,
    relevant_contents: list[str],
) -> str:
    """
    Classify a query as lexical or paraphrase.

    Args:
        query: Query text.
        relevant_contents: Content of the query's relevant memories.

    Returns:
        "lexical" if any relevant memory shares a non-stopword token
        with the query, otherwise "paraphrase".
    """

    query_terms = set(tokenize(query)) - STOPWORDS

    for content in relevant_contents:
        if query_terms & set(tokenize(content)):
            return "lexical"

    return "paraphrase"


def evaluate(
    dataset: dict,
    candidates: int,
    rerank_pool: int,
) -> tuple[list[dict], list[dict]]:
    """
    Run every pipeline over every query in the dataset.

    Args:
        dataset: Loaded evaluation dataset.
        candidates: Results taken from each retriever before fusion.
        rerank_pool: Fused results passed to the reranker.

    Returns:
        Results for answerable queries and for no-answer queries.
    """

    memories = [
        Memory(
            id=memory["id"],
            user_id=1,
            content=memory["content"],
            source="evaluation",
        )
        for memory in dataset["memories"]
    ]
    content_by_id = {memory.id: memory.content for memory in memories}

    memory_vectors = np.asarray(
        embedding_model.embed([memory.content for memory in memories])
    )

    answerable = []

    for query in dataset["queries"]:
        answerable.append(
            {
                "id": query["id"],
                "query": query["query"],
                "relevant": set(query["relevant"]),
                "type": query_type(
                    query["query"],
                    [content_by_id[memory_id] for memory_id in query["relevant"]],
                ),
                "rankings": retrieve(
                    query["query"], memories, memory_vectors, candidates, rerank_pool
                ),
            }
        )

    no_answer = [
        {
            "id": query["id"],
            "query": query["query"],
            "rankings": retrieve(
                query["query"], memories, memory_vectors, candidates, rerank_pool
            ),
        }
        for query in dataset["no_answer_queries"]
    ]

    return answerable, no_answer


def ranked_ids(results: list[SearchResult]) -> list[int]:
    """Extract memory IDs from ranked search results."""

    return [result.memory_id for result in results]


def summarize(results: list[dict]) -> dict[str, dict[str, float]]:
    """
    Average the ranking metrics for each pipeline.

    Args:
        results: Results for answerable queries.

    Returns:
        Recall@k and MRR per pipeline.
    """

    summary = {}

    for pipeline in PIPELINES:
        metrics = {}

        for k in RECALL_CUTOFFS:
            metrics[f"recall@{k}"] = statistics.mean(
                recall_at_k(ranked_ids(result["rankings"][pipeline]), result["relevant"], k)
                for result in results
            )

        metrics["mrr"] = statistics.mean(
            reciprocal_rank(ranked_ids(result["rankings"][pipeline]), result["relevant"])
            for result in results
        )

        summary[pipeline] = metrics

    return summary


def compare_to_baseline(results: list[dict]) -> dict[str, tuple[int, int]]:
    """
    Count per-query wins and losses against the baseline pipeline.

    A pipeline wins a query when its first relevant memory ranks higher
    than the baseline's, and loses when it ranks lower.

    Args:
        results: Results for answerable queries.

    Returns:
        (wins, losses) per pipeline.
    """

    comparison = {}

    for pipeline in PIPELINES:
        wins = losses = 0

        for result in results:
            baseline_rr = reciprocal_rank(
                ranked_ids(result["rankings"][BASELINE_PIPELINE]), result["relevant"]
            )
            pipeline_rr = reciprocal_rank(
                ranked_ids(result["rankings"][pipeline]), result["relevant"]
            )

            if pipeline_rr > baseline_rr:
                wins += 1
            elif pipeline_rr < baseline_rr:
                losses += 1

        comparison[pipeline] = (wins, losses)

    return comparison


def format_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    """Render rows as a plain-text table with the first column left-aligned."""

    widths = [
        max(len(row[column]) for row in [headers, *rows])
        for column in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        cells = [row[0].ljust(widths[0])]
        cells += [cell.rjust(width) for cell, width in zip(row[1:], widths[1:])]
        return "  ".join(cells)

    lines = [render(headers), "  ".join("-" * width for width in widths)]
    lines += [render(row) for row in rows]

    return "\n".join(lines)


def describe_scores(scores: list[float]) -> str:
    """Summarize scores as 'median [min, max]'."""

    return f"{statistics.median(scores):.3f} [{min(scores):.3f}, {max(scores):.3f}]"


def print_report(
    dataset: dict,
    answerable: list[dict],
    no_answer: list[dict],
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    """
    Print the evaluation report.

    Args:
        dataset: Loaded evaluation dataset.
        answerable: Results for answerable queries.
        no_answer: Results for no-answer queries.
        args: Command-line arguments.

    Returns:
        Overall metrics per pipeline.
    """

    by_type = {
        name: [result for result in answerable if result["type"] == name]
        for name in ("lexical", "paraphrase")
    }

    print()
    print(
        f"Dataset: {dataset['name']} | {len(dataset['memories'])} memories | "
        f"{len(answerable)} queries ({len(by_type['lexical'])} lexical, "
        f"{len(by_type['paraphrase'])} paraphrase) | {len(no_answer)} no-answer"
    )
    print(
        f"Models: {embedding_model.model_name}, {RERANKER_MODEL_NAME} | "
        f"RRF k={DEFAULT_RRF_K}, candidates per retriever={args.candidates}, "
        f"rerank pool={args.rerank_pool}"
    )

    summary = summarize(answerable)
    comparison = compare_to_baseline(answerable)

    print("\nOverall")
    print(
        format_table(
            ["pipeline", *(f"R@{k}" for k in RECALL_CUTOFFS), "MRR", f"vs {BASELINE_PIPELINE}"],
            [
                [
                    pipeline,
                    *(f"{metrics[f'recall@{k}']:.3f}" for k in RECALL_CUTOFFS),
                    f"{metrics['mrr']:.3f}",
                    "baseline"
                    if pipeline == BASELINE_PIPELINE
                    else f"+{comparison[pipeline][0]} / -{comparison[pipeline][1]}",
                ]
                for pipeline, metrics in summary.items()
            ],
        )
    )
    print(
        "R@k = relevant memories in the top k / min(k, relevant memories), so "
        "R@3 can be lower than R@1 on multi-memory queries."
    )
    print(
        f"'vs {BASELINE_PIPELINE}' counts queries where the first relevant memory "
        "ranked higher (+) or lower (-) than in the baseline."
    )

    type_summaries = {
        name: summarize(results)
        for name, results in by_type.items()
        if results
    }

    print("\nBy query type (R@1 / MRR)")
    print(
        format_table(
            [
                "pipeline",
                *(f"{name} (n={len(by_type[name])})" for name in type_summaries),
            ],
            [
                [
                    pipeline,
                    *(
                        f"{type_summary[pipeline]['recall@1']:.3f} / "
                        f"{type_summary[pipeline]['mrr']:.3f}"
                        for type_summary in type_summaries.values()
                    ),
                ]
                for pipeline in PIPELINES
            ],
        )
    )

    print("\nScore calibration: median [min, max]")
    calibration_rows = []

    for pipeline in SCORED_PIPELINES:
        best_relevant_scores = [
            max(
                result.score
                for result in query["rankings"][pipeline]
                if result.memory_id in query["relevant"]
            )
            for query in answerable
        ]
        top_scores_without_answer = [
            query["rankings"][pipeline][0].score
            for query in no_answer
        ]

        calibration_rows.append(
            [
                pipeline,
                describe_scores(best_relevant_scores),
                describe_scores(top_scores_without_answer),
            ]
        )

    print(
        format_table(
            ["scores", "best relevant (answerable)", "top-1 (no-answer)"],
            calibration_rows,
        )
    )
    print(
        "BGE scores are cosine similarities. Reranker scores are the "
        "CrossEncoder's sigmoid outputs: monotonic in the logits, not "
        "calibrated probabilities."
    )

    relevant_bge_scores = [
        result.score
        for query in answerable
        for result in query["rankings"]["bge"]
        if result.memory_id in query["relevant"]
    ]
    kept = sum(score >= args.threshold for score in relevant_bge_scores)
    leaked = sum(
        query["rankings"]["bge"][0].score >= args.threshold
        for query in no_answer
    )

    print(
        f"BGE threshold {args.threshold}: keeps {kept}/{len(relevant_bge_scores)} "
        f"relevant memories; {leaked}/{len(no_answer)} no-answer queries still "
        "return a memory."
    )

    if args.per_query:
        print("\nRank of first relevant memory ('-' = not retrieved)")
        print(
            format_table(
                ["query", "type", *PIPELINES],
                [
                    [
                        f"{result['id']} {result['query'][:44]}",
                        result["type"],
                        *(
                            str(
                                first_relevant_rank(
                                    ranked_ids(result["rankings"][pipeline]),
                                    result["relevant"],
                                )
                                or "-"
                            )
                            for pipeline in PIPELINES
                        ),
                    ]
                    for result in answerable
                ],
            )
        )

    return summary


def write_results(
    path: Path,
    dataset: dict,
    answerable: list[dict],
    summary: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> None:
    """
    Write metrics and per-query top-5 rankings to JSON for later comparison.

    Args:
        path: Output file path.
        dataset: Loaded evaluation dataset.
        answerable: Results for answerable queries.
        summary: Overall metrics per pipeline.
        args: Command-line arguments.
    """

    output = {
        "dataset": dataset["name"],
        "config": {
            "embedding_model": embedding_model.model_name,
            "reranker_model": RERANKER_MODEL_NAME,
            "rrf_k": DEFAULT_RRF_K,
            "candidates": args.candidates,
            "rerank_pool": args.rerank_pool,
        },
        "pipelines": PIPELINES,
        "summary": summary,
        "queries": [
            {
                "id": result["id"],
                "query": result["query"],
                "type": result["type"],
                "relevant": sorted(result["relevant"]),
                "first_relevant_rank": {
                    pipeline: first_relevant_rank(
                        ranked_ids(rankings), result["relevant"]
                    )
                    for pipeline, rankings in result["rankings"].items()
                },
                "top5": {
                    pipeline: ranked_ids(rankings)[:5]
                    for pipeline, rankings in result["rankings"].items()
                },
            }
            for result in answerable
        ],
    }

    path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Continuum retrieval pipelines on a labelled dataset.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--candidates",
        type=int,
        default=20,
        help="Results taken from BGE and from BM25 before RRF fusion.",
    )
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=20,
        help="Top fused results passed to the reranker.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="BGE similarity threshold to check (production default is 0.4).",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="Print the rank of the first relevant memory for every query.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write metrics and per-query rankings to this JSON file.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    query_count = len(dataset["queries"]) + len(dataset["no_answer_queries"])
    print(f"Evaluating {len(PIPELINES)} pipelines on {query_count} queries...")

    answerable, no_answer = evaluate(dataset, args.candidates, args.rerank_pool)
    summary = print_report(dataset, answerable, no_answer, args)

    if args.output:
        write_results(args.output, dataset, answerable, summary, args)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
