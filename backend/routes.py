from fastapi import APIRouter
from sqlmodel import Session

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