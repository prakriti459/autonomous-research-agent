import json

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.audit import record_request
from app.db.models import Base, RequestLog


@pytest.mark.asyncio
async def test_request_is_logged_as_json_and_persisted(caplog: pytest.LogCaptureFixture) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    with caplog.at_level("INFO", logger="research_agent.requests"):
        await record_request(
            session_factory=session_factory,
            session_id=None,
            tool_used="search_papers",
            latency_ms=124.5,
            grounded=True,
            error=None,
            status_code=200,
            method="POST",
            path="/ask",
        )

    payload = json.loads(caplog.records[-1].message)
    assert payload["tool_used"] == "search_papers"
    assert payload["grounded"] is True
    assert payload["latency_ms"] == 124.5
    async with session_factory() as session:
        logged = await session.scalar(select(RequestLog))
    assert logged is not None
    assert logged.status_code == 200
    assert logged.error is None
    await engine.dispose()
