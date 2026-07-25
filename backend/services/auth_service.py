"""Authentication service containing business logic for user registration and login."""

from fastapi import HTTPException
from sqlmodel import Session, select

from models import User, UserRegister, Token
from security import hash_password, verify_password
from auth import create_access_token


def register_user(user: UserRegister, session: Session) -> User:
    """
    Register a new user.
    
    Args:
        user: UserRegister model with username, email, and password
        session: SQLModel session
        
    Returns:
        Created User object
        
    Raises:
        HTTPException: If username or email already exists
    """
    # Check if user already exists
    existing_user = session.exec(
        select(User).where(User.email == user.email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    existing_username = session.exec(
        select(User).where(User.username == user.username)
    ).first()
    
    if existing_username:
        raise HTTPException(
            status_code=400,
            detail="Username already taken"
        )
    
    # Create new user
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hash_password(user.password)
    )
    
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    
    return db_user


def login_user(email: str, password: str, session: Session) -> Token:
    """
    Authenticate user and return access token.
    
    Args:
        email: User email
        password: User password (plain text)
        session: SQLModel session
        
    Returns:
        Token object with access_token and token_type
        
    Raises:
        HTTPException: If user not found or password incorrect
    """
    # Find user by email
    db_user = session.exec(
        select(User).where(User.email == email)
    ).first()
    
    if db_user is None:
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )
    
    # Verify password
    password_ok = verify_password(password, db_user.hashed_password)
    
    if not password_ok:
        raise HTTPException(
            status_code=401,
            detail="Password incorrect"
        )
    
    # Create and return access token
    access_token = create_access_token({"sub": db_user.email})
    
    return Token(
        access_token=access_token,
        token_type="bearer"
    )
