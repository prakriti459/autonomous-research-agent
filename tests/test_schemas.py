import pytest
from pydantic import ValidationError

from app.schemas import AnswerOutput, Citation, ToolChoice


def test_metadata_citation_can_reference_paper_without_page() -> None:
    citation = Citation(paper="Example paper")

    assert citation.page is None
    assert citation.chunk_id is None


def test_answer_accepts_chunk_citation() -> None:
    answer = AnswerOutput(
        answer="The method uses retrieval.",
        citations=[{"paper": "Example paper", "chunk_id": "chunk_4"}],
        grounded=True,
    )

    assert answer.citations[0].chunk_id == "chunk_4"


def test_tool_choice_rejects_unlisted_tool() -> None:
    with pytest.raises(ValidationError):
        ToolChoice(tool="run_sql", query="SELECT * FROM papers")
