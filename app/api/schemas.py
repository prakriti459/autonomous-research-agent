from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings
from app.schemas import AnswerOutput


class AskRequest(BaseModel):
    session_id: UUID | None = None
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def enforce_question_limit(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Question must not be empty")
        if len(normalized) > get_settings().max_question_length:
            raise ValueError("Question exceeds the configured maximum length")
        return normalized


class AskResponse(BaseModel):
    session_id: UUID
    tool_used: str
    answer: AnswerOutput
    attempts: int


class IngestResponse(BaseModel):
    paper_id: UUID
    title: str
    year: int
    chunks_indexed: int


class HistoryMessage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    citations: list[dict[str, object]] | None
    grounded: bool | None
    created_at: datetime


class HistoryResponse(BaseModel):
    session_id: UUID
    messages: list[HistoryMessage]
