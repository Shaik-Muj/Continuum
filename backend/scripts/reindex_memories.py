from pathlib import Path
import sys

from sqlmodel import Session, select


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from database import engine
from models import Memory
from services.embedding_service import reindex_memories


def main() -> None:
    with Session(engine) as session:
        memories = session.exec(
            select(Memory)
        ).all()

        print(f"Found {len(memories)} memories.")

        reindex_memories(memories)

        print("ChromaDB reindex completed.")


if __name__ == "__main__":
    main()