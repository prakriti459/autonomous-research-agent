import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import Base, Paper
from app.db.repositories import (
    get_or_create_session,
    get_recent_messages,
    lookup_paper_metadata,
    save_message,
)


@pytest.mark.asyncio
async def test_metadata_lookup_uses_year_and_title_filters() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        session.add_all(
            [
                Paper(
                    id="paper-2023",
                    title="Efficient Retrieval Systems",
                    authors=["A. Researcher"],
                    year=2023,
                    file_name="retrieval.pdf",
                ),
                Paper(
                    id="paper-2022",
                    title="Other Methods",
                    authors=["B. Scientist"],
                    year=2022,
                    file_name="methods.pdf",
                ),
            ]
        )
        await session.commit()

        by_year = await lookup_paper_metadata(session, "Which papers are from 2023?")
        by_title = await lookup_paper_metadata(session, "Efficient Retrieval Systems")

    assert [paper.id for paper in by_year] == ["paper-2023"]
    assert [paper.id for paper in by_title] == ["paper-2023"]
    await engine.dispose()


@pytest.mark.asyncio
async def test_messages_are_persisted_and_loaded_in_chronological_order() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        session_id = await get_or_create_session(session)
        await save_message(
            session, session_id=session_id, role="user", content="First question"
        )
        await save_message(
            session, session_id=session_id, role="assistant", content="First answer"
        )
        await save_message(
            session, session_id=session_id, role="user", content="Follow-up"
        )
        await session.commit()

        recent = await get_recent_messages(session, session_id, limit=2)

    assert [message.content for message in recent] == ["First answer", "Follow-up"]
    await engine.dispose()
