from services.embedding_service import find_similar_memories


results = find_similar_memories(
    query="Where do I work and what am I doing professionally?",
    user_id=1,
    top_k=5,
    threshold=0.4,
)

for result in results:
    print(
        f"memory_id={result.memory_id}, "
        f"score={result.score}"
    )