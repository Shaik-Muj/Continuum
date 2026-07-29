# main.py

from fastapi import FastAPI

from database import create_db_and_tables
from routers.auth import router as auth_router
from routers.memory import router as memory_router

app = FastAPI(
    title="Continuum API",
    description="Memory management API",
    version="1.0.0"
)


# @app.on_event("startup")
# def on_startup():
#     create_db_and_tables()


app.include_router(auth_router)
app.include_router(memory_router)


@app.get("/")
def root():
    return {"message": "Continuum API Running"}