from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    chunk_id: str | None = Field(default=None, min_length=1)

class AnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool


class ToolChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["search_papers", "lookup_paper_metadata"]
    query: str = Field(min_length=1, max_length=2000)
