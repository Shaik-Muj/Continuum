from sqlmodel import Session, select

from database import engine
from models import Memory
from services.bm25_service import search_bm25


queries = [
    "Where do I work?",
    "What are my plans for higher education?",
    "What do I do professionally?",
]


with Session(engine) as session:

    memories = session.exec(
        select(Memory).where(
            Memory.user_id == 1
        )
    ).all()

    for query in queries:

        print("=" * 60)
        print(f"Query: {query}")
        print()

        results = search_bm25(
            query=query,
            memories=memories,
            top_k=5,
        )

        for result in results:
            memory = next(
                memory
                for memory in memories
                if memory.id == result.memory_id
            )

            print(
                f"{result.score:.4f}  "
                f"{memory.content}"
            )

        print()