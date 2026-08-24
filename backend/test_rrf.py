from services.rrf_service import reciprocal_rank_fusion
from models import SearchResult


bge_results = [
    SearchResult(memory_id=2, score=0.5235),
    SearchResult(memory_id=1, score=0.4785),
    SearchResult(memory_id=3, score=0.4435),
]

bm25_results = [
    SearchResult(memory_id=3, score=0.1009),
    SearchResult(memory_id=2, score=0.0898),
    SearchResult(memory_id=1, score=0.0771),
]


results = reciprocal_rank_fusion(
    rankings=[
        bge_results,
        bm25_results,
    ]
)


for result in results:
    print(
        f"memory_id={result.memory_id}, "
        f"rrf_score={result.score:.6f}"
    )