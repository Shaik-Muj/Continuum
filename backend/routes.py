from fastapi import APIRouter, Query, HTTPException
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
    
@router.get("/memory/{memory_id}")
def get_memory(memory_id: int):
    with Session(engine) as session:
        memory = session.get(Memory, memory_id)

        if not memory:
            raise HTTPException(
                status_code=404,
                detail="Memory not found"
            )

        return memory
    
@router.delete("/memory/{memory_id}")
def delete_memory(memory_id: int):
    with Session(engine) as session:
        memory = session.get(Memory, memory_id)

        if memory is None:
            raise HTTPException(
                status_code=404,
                detail="Memory not found"
            )

        session.delete(memory)
        session.commit()

        return {
            "message": f"Memory {memory_id} deleted successfully"
        }