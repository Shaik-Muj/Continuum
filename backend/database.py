from sqlmodel import SQLModel, create_engine

# Import all table models BEFORE create_all()
from models import Memory
from config import settings

engine = create_engine(settings.DATABASE_URL)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)