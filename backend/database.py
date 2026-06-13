from sqlmodel import SQLModel, create_engine

# Import all table models BEFORE create_all()
from models import Memory

DATABASE_URL = "postgresql://postgres:123456@localhost/continuum"

engine = create_engine(DATABASE_URL)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)