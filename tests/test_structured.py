from types import SimpleNamespace

import pytest

from app.agent.structured import StructuredOutputError, generate_structured
from app.schemas import ToolChoice


class FakeOllama:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    async def chat(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        content = next(self.responses)
        return SimpleNamespace(message=SimpleNamespace(content=content))


@pytest.mark.asyncio
async def test_structured_output_retries_validation_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.agent.structured.get_settings",
        lambda: SimpleNamespace(model_name="phi3"),
    )
    client = FakeOllama(
        [
            '{"tool":"not_allowed","query":"recent papers"}',
            '{"tool":"lookup_paper_metadata","query":"papers from 2023"}',
        ]
    )

    result = await generate_structured(
        client,  # type: ignore[arg-type]
        messages=[{"role": "user", "content": "Which papers are from 2023?"}],
        output_schema=ToolChoice,
    )

    assert result.tool == "lookup_paper_metadata"
    assert client.calls == 2


@pytest.mark.asyncio
async def test_structured_output_fails_after_one_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.agent.structured.get_settings",
        lambda: SimpleNamespace(model_name="phi3"),
    )
    client = FakeOllama(['{"tool":"bad","query":"x"}', "not json"])

    with pytest.raises(StructuredOutputError):
        await generate_structured(
            client,  # type: ignore[arg-type]
            messages=[{"role": "user", "content": "Search"}],
            output_schema=ToolChoice,
        )

    assert client.calls == 2
