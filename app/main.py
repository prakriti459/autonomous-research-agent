import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from time import perf_counter

from fastapi import FastAPI, Request, Response
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.base import RequestResponseEndpoint

from app.api.audit import record_request
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


@app.middleware("http")
async def audit_requests(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    started_at = perf_counter()
    error = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        if status_code >= 400:
            error = f"HTTP {status_code}"
        return response
    except Exception as exception:
        error = type(exception).__name__
        raise
    finally:
        await record_request(
            session_id=getattr(request.state, "session_id", None),
            tool_used=getattr(request.state, "tool_used", None),
            latency_ms=(perf_counter() - started_at) * 1000,
            grounded=getattr(request.state, "grounded", None),
            error=error,
            status_code=status_code,
            method=request.method,
            path=request.url.path,
        )


@app.get("/health", tags=["operations"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
