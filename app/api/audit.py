import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db.database import SessionFactory
from app.db.models import RequestLog


logger = logging.getLogger("research_agent.requests")


async def record_request(
    *,
    session_factory: async_sessionmaker[Any] | None = None,
    session_id: str | None,
    tool_used: str | None,
    latency_ms: float,
    grounded: bool | None,
    error: str | None,
    status_code: int,
    method: str,
    path: str,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "time": timestamp,
        "session_id": session_id,
        "tool_used": tool_used,
        "latency_ms": round(latency_ms, 2),
        "grounded": grounded,
        "error": error,
        "status_code": status_code,
        "method": method,
        "path": path,
    }
    logger.info(json.dumps(payload, sort_keys=True))

    async with (session_factory or SessionFactory)() as db:
        db.add(
            RequestLog(
                session_id=session_id,
                tool_used=tool_used,
                latency_ms=latency_ms,
                grounded=grounded,
                error=error,
                status_code=status_code,
            )
        )
        await db.commit()
