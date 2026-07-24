from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm

from sqlmodel import Session, select

from database import engine
from models import (
    Memory,
    User,
    UserRegister,
    Token,
    MemoryCreate,
)

from security import hash_password, verify_password
from auth import create_access_token, get_current_user

router = APIRouter()


# ----------------------------
# Memory Endpoints
# ----------------------------

@router.post("/memory")
def create_memory(
    memory: MemoryCreate,
    current_user: User = Depends(get_current_user)
):
    db_memory = Memory(
        content=memory.content,
        source=memory.source,
        user_id=current_user.id
    )

    with Session(engine) as session:
        session.add(db_memory)
        session.commit()
        session.refresh(db_memory)

        return db_memory


@router.get("/memory")
def get_memories(
    current_user: User = Depends(get_current_user)
):
    with Session(engine) as session:
        memories = session.exec(
            select(Memory).where(
                Memory.user_id == current_user.id
            )
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
def get_memory(
    memory_id: int,
    current_user: User = Depends(get_current_user)
):
    with Session(engine) as session:

        memory = session.get(Memory, memory_id)

        if memory is None:
            raise HTTPException(
                status_code=404,
                detail="Memory not found"
            )

        if memory.user_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="Access denied"
            )

        return memory


@router.delete("/memory/{memory_id}")
def delete_memory(
    memory_id: int,
    current_user: User = Depends(get_current_user)
):
    with Session(engine) as session:

        memory = session.get(Memory, memory_id)

        if memory is None:
            raise HTTPException(
                status_code=404,
                detail="Memory not found"
            )

        if memory.user_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="Access denied"
            )

        session.delete(memory)
        session.commit()

        return {
            "message": f"Memory {memory_id} deleted successfully"
        }


# ----------------------------
# Authentication
# ----------------------------

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
def login(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    with Session(engine) as session:

        print("Received username:", repr(form_data.username))
        print("Received password:", repr(form_data.password))

        email = form_data.username.strip()

        db_user = session.exec(
        select(User).where(User.email == email)
        ).first()

        print("User found:", db_user)

        if db_user is None:
            raise HTTPException(
                status_code=401,
                detail="User not found"
            )

        password_ok = verify_password(
            form_data.password,
            db_user.hashed_password
        )

        print("Password OK:", password_ok)

        if not password_ok:
            raise HTTPException(
                status_code=401,
                detail="Password incorrect"
            )

        access_token = create_access_token(
            {"sub": db_user.email}
        )

        return Token(
            access_token=access_token,
            token_type="bearer"
        )