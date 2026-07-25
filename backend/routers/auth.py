"""Authentication router handling user registration and login endpoints."""

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session

from database import engine
from models import User, UserRegister, Token
from services.auth_service import register_user, login_user

router = APIRouter(tags=["auth"])


def get_session():
    """Dependency to get database session."""
    with Session(engine) as session:
        yield session


@router.post("/register", response_model=User)
def register(
    user: UserRegister,
    session: Session = Depends(get_session)
):
    """
    Register a new user.
    
    - **username**: Unique username
    - **email**: Unique email address
    - **password**: Password (will be hashed)
    """
    return register_user(user, session)


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session)
):
    """
    Login with email and password.
    
    Returns JWT access token for authenticated requests.
    """
    email = form_data.username.strip()
    return login_user(email, form_data.password, session)
