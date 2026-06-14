from fastapi import APIRouter, Query
from sqlmodel import Session, select

from database import engine
from models import Memory

router = APIRouter()


@router.post("/memory")
def create_memory(memory: Memory):
    with Session(engine) as session:
        session.add(memory)
        session.commit()
        session.refresh(memory)

        return memory


@router.get("/memory")
def get_memories():
    with Session(engine) as session:
        memories = session.exec(
            select(Memory)
        ).all()

        return memories


@router.get("/search")
def search_memories(q: str = Query(...)):
    with Session(engine) as session:
        memories = session.exec(
            select(Memory).where(
                Memory.content.contains(q)
            )
        ).all()

        return memories