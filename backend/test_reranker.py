from core.reranker import rerank


tests = [
    (
        "Where do I work?",
        [
            "I want to pursue my Masters in Germany.",
            "I joined Infor as an AI Developer Intern in FEBRUARY 2026.",
            "I enjoy playing football on weekends.",
        ],
    ),
    (
        "What are my plans for higher education?",
        [
            "I want to pursue my Masters in Germany.",
            "I joined Infor as an AI Developer Intern in FEBRUARY 2026.",
            "I enjoy playing football on weekends.",
        ],
    ),
    (
        "What do I do professionally?",
        [
            "I want to pursue my Masters in Germany.",
            "I joined Infor as an AI Developer Intern in FEBRUARY 2026.",
            "I enjoy playing football on weekends.",
        ],
    ),
]


for query, documents in tests:
    print("=" * 60)
    print(f"Query: {query}")
    print()

    scores = rerank(
        query=query,
        documents=documents,
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    for document, score in ranked:
        print(f"{score:.4f}  {document}")

    print()