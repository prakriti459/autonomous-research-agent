import asyncio
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import lookup_paper_metadata
from app.tools.retrieval import get_retrieval_pipeline


ALLOWED_TOOLS = frozenset({"search_papers", "lookup_paper_metadata"})


async def search_papers(query: str) -> list[dict[str, Any]]:
    pipeline = await asyncio.to_thread(get_retrieval_pipeline)
    return await asyncio.to_thread(pipeline.search, query)


async def lookup_paper_metadata_tool(
    db: AsyncSession, query: str
) -> list[dict[str, Any]]:
    papers = await lookup_paper_metadata(db, query)
    return [
        {
            "paper_id": paper.id,
            "paper": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "file_name": paper.file_name,
            "text": (
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(paper.authors)}\n"
                f"Year: {paper.year}\n"
                f"File: {paper.file_name}"
            ),
        }
        for paper in papers
    ]


async def execute_tool(
    tool_name: str,
    query: str,
    db: AsyncSession,
) -> list[dict[str, Any]]:
    if tool_name not in ALLOWED_TOOLS:
        raise ValueError(f"Tool is not allow-listed: {tool_name}")
    if tool_name == "search_papers":
        return await search_papers(query)
    return await lookup_paper_metadata_tool(db, query)
