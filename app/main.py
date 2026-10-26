import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError

from app.api.routes import router
from app.config import get_settings
from app.db.database import close_database, create_database_tables


async def initialize_database() -> None:
    for attempt in range(5):
        try:
            await create_database_tables()
            return
        except SQLAlchemyError:
            if attempt == 4:
                raise
            await asyncio.sleep(2**attempt)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if len(get_settings().api_key) < 32:
        raise RuntimeError("Set API_KEY in .env to a random value of at least 32 characters")
    try:
        await initialize_database()
        yield
    finally:
        await close_database()


app = FastAPI(
    title="Autonomous Research Agent",
    description="Local RAG assistant for research papers",
    version="2.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health", tags=["operations"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
