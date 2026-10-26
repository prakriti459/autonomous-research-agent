import json
from typing import TypeVar

from ollama import AsyncClient
from pydantic import BaseModel, ValidationError

from app.config import get_settings


OutputT = TypeVar("OutputT", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    pass


async def generate_structured(
    client: AsyncClient,
    *,
    messages: list[dict[str, str]],
    output_schema: type[OutputT],
) -> OutputT:
    settings = get_settings()
    retry_messages = messages
    last_error: Exception | None = None

    for attempt in range(2):
        response = await client.chat(
            model=settings.model_name,
            messages=retry_messages,
            format=output_schema.model_json_schema(),
        )
        content = response.message.content

        try:
            return output_schema.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as error:
            last_error = error
            if attempt == 0:
                retry_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not match the required JSON schema. "
                            f"Validation error: {error}. Return corrected JSON only."
                        ),
                    },
                ]

    raise StructuredOutputError(
        f"Model returned invalid {output_schema.__name__} JSON after one retry"
    ) from last_error
