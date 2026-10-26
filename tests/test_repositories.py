import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import Base, Paper
from app.db.repositories import lookup_paper_metadata


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
