from fastapi import APIRouter, Query, HTTPException
from sqlmodel import Session, select

from database import engine
from models import Memory, UserLogin, Token

from security import hash_password, verify_password
from models import User, UserRegister
from auth import create_access_token

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


@router.post("/register")
def register(user: UserRegister):

    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hash_password(user.password)
    )

    with Session(engine) as session:

        session.add(db_user)
        session.commit()
        session.refresh(db_user)

        return db_user

@router.post("/login", response_model=Token)
def login(user: UserLogin):

    with Session(engine) as session:

        db_user = session.exec(
            select(User).where(User.email == user.email)
        ).first()

        if not db_user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        if not verify_password(
            user.password,
            db_user.hashed_password
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        access_token = create_access_token(
            {"sub": db_user.email}
        )

        return {
            "access_token": access_token,
            "token_type": "bearer"
        }