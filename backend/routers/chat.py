"""Chat router for memory-aware conversations."""

from fastapi import APIRouter, Depends
from sqlmodel import Session

from auth import get_current_user
from models import User
from routers.memory import get_session
from schemas.chat import ChatRequest, ChatResponse
from services.chat_service import chat


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post("/", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> ChatResponse:
    """
    Generate a response using the current user's memories.
    """

    response = chat(
        user_message=request.message,
        user_id=current_user.id,
        session=session,
    )

    return ChatResponse(
        response=response
    )