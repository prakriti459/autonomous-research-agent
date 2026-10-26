import asyncio
import re
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from ollama import ResponseError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.memory import load_history
from app.agent.structured import StructuredOutputError
from app.agent.workflow import CitationValidationError, answer_question
from app.config import get_settings
from app.db.database import get_db
from app.db.models import Paper, SessionRecord
from app.db.repositories import (
    get_or_create_session,
    get_recent_messages,
    save_message,
)
from app.api.auth import require_api_key
from app.api.schemas import (
    AskRequest,
    AskResponse,
    HistoryMessage,
    HistoryResponse,
    IngestResponse,
)
from app.ingestion.pdf import parse_pdf
from app.tools.retrieval import get_retrieval_pipeline


router = APIRouter()
MAX_HISTORY_MESSAGES = 100


def _parse_authors(value: str) -> list[str]:
    authors = [author.strip() for author in value.split(",") if author.strip()]
    if not authors:
        raise ValueError("At least one paper author is required")
    return authors


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
async def ingest(
    file: UploadFile = File(...),
    title: str = Form(..., min_length=1, max_length=512),
    authors: str = Form(..., min_length=1, max_length=2000),
    year: int = Form(..., ge=1800),
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    settings = get_settings()
    current_year = datetime.now().year
    if year > current_year + 1:
        raise HTTPException(status_code=422, detail="Publication year is out of range")

    original_name = Path(file.filename or "").name
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted")
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", original_name)
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="PDF exceeds the configured size limit")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="Uploaded file is not a valid PDF")

    try:
        authors_list = _parse_authors(authors)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    paper_id = str(uuid4())
    upload_directory = Path(settings.upload_directory)
    await asyncio.to_thread(upload_directory.mkdir, parents=True, exist_ok=True)
    temporary_pdf = upload_directory / f"{paper_id}.pdf"
    pipeline = None
    indexed = False
    committed = False
    try:
        await asyncio.to_thread(temporary_pdf.write_bytes, content)
        try:
            chunks = await asyncio.to_thread(parse_pdf, temporary_pdf)
        except (RuntimeError, ValueError, OSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        pipeline = await asyncio.to_thread(get_retrieval_pipeline)
        indexed = True
        indexed_count = await asyncio.to_thread(
            pipeline.index_documents,
            chunks,
            paper_id=paper_id,
            paper_title=title.strip(),
        )
        db.add(
            Paper(
                id=paper_id,
                title=title.strip(),
                authors=authors_list,
                year=year,
                file_name=safe_name,
            )
        )
        await db.commit()
        committed = True
    finally:
        if indexed and not committed and pipeline is not None:
            await db.rollback()
            await asyncio.to_thread(pipeline.delete_paper, paper_id)
        await file.close()
        if temporary_pdf.exists():
            await asyncio.to_thread(temporary_pdf.unlink)

    return IngestResponse(
        paper_id=UUID(paper_id),
        title=title.strip(),
        year=year,
        chunks_indexed=indexed_count,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(require_api_key)],
)
async def ask(
    request: AskRequest,
    db: AsyncSession = Depends(get_db),
) -> AskResponse:
    session_id = await get_or_create_session(
        db,
        str(request.session_id) if request.session_id is not None else None,
    )
    history = await load_history(db, session_id)
    await save_message(
        db,
        session_id=session_id,
        role="user",
        content=request.question,
    )
    await db.commit()

    try:
        result = await answer_question(
            db,
            question=request.question,
            history=history,
        )
    except (
        StructuredOutputError,
        CitationValidationError,
        ResponseError,
        httpx.HTTPError,
    ) as error:
        raise HTTPException(
            status_code=502,
            detail="Research model failed to produce a valid grounded answer",
        ) from error

    await save_message(
        db,
        session_id=session_id,
        role="assistant",
        content=result.answer.answer,
        citations=[
            citation.model_dump(exclude_none=True)
            for citation in result.answer.citations
        ],
        grounded=result.answer.grounded,
    )
    await db.commit()
    return AskResponse(
        session_id=UUID(session_id),
        tool_used=result.tool_used,
        answer=result.answer,
        attempts=result.attempts,
    )


@router.get(
    "/history/{session_id}",
    response_model=HistoryResponse,
    dependencies=[Depends(require_api_key)],
)
async def history(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> HistoryResponse:
    session = await db.get(SessionRecord, str(session_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await get_recent_messages(
        db,
        str(session_id),
        limit=MAX_HISTORY_MESSAGES,
    )
    return HistoryResponse(
        session_id=session_id,
        messages=[HistoryMessage.model_validate(message) for message in messages],
    )
