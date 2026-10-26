import re
from uuid import uuid4

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Message, Paper, SessionRecord


async def create_session(db: AsyncSession) -> str:
    session = SessionRecord(id=str(uuid4()))
    db.add(session)
    await db.flush()
    return session.id


async def get_recent_messages(
    db: AsyncSession, session_id: str, limit: int
) -> list[Message]:
    result = await db.scalars(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
    )
    return list(reversed(result.all()))


async def save_message(
    db: AsyncSession,
    *,
    session_id: str,
    role: str,
    content: str,
    citations: list[dict[str, object]] | None = None,
    grounded: bool | None = None,
) -> None:
    db.add(
        Message(
            session_id=session_id,
            role=role,
            content=content,
            citations=citations,
            grounded=grounded,
        )
    )
    await db.flush()


async def lookup_paper_metadata(db: AsyncSession, query: str) -> list[Paper]:
    year_match = re.search(r"\b(?:19|20)\d{2}\b", query)
    if year_match:
        statement = select(Paper).where(Paper.year == int(year_match.group()))
    else:
        terms = [
            term
            for term in re.findall(r"[A-Za-z0-9-]{2,}", query)
            if term.lower()
            not in {"which", "what", "papers", "paper", "from", "published", "about"}
        ]
        if not terms:
            return []
        statement = select(Paper).where(
            and_(
                *[
                    or_(
                        Paper.title.contains(term, autoescape=True),
                        Paper.file_name.contains(term, autoescape=True),
                    )
                    for term in terms
                ]
            )
        )

    result = await db.scalars(statement.order_by(Paper.year.desc(), Paper.title).limit(20))
    return list(result.all())
