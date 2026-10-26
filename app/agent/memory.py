from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.repositories import get_recent_messages


def trim_history(
    messages: list[dict[str, str]], *, max_tokens: int
) -> list[dict[str, str]]:
    budget = max_tokens
    selected: list[dict[str, str]] = []
    for message in reversed(messages):
        token_estimate = max(1, (len(message["content"]) + 3) // 4) + 4
        if token_estimate > budget:
            break
        selected.append(message)
        budget -= token_estimate
    return list(reversed(selected))


async def load_history(
    db: AsyncSession, session_id: str
) -> list[dict[str, str]]:
    settings = get_settings()
    rows = await get_recent_messages(
        db,
        session_id,
        settings.conversation_history_limit,
    )
    messages = [
        {"role": row.role, "content": row.content}
        for row in rows
        if row.role in {"user", "assistant"}
    ]
    return trim_history(messages, max_tokens=settings.history_token_budget)
