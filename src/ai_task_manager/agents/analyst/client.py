import json
from importlib.resources import files

import httpx

from .schema import AnalystInput, AnalystResult
from .validation import validate_task_references


def analyze(
    data: AnalystInput,
    *,
    model: str,
    client: httpx.Client,
) -> AnalystResult:
    """Получает разбор запроса от модели и проверяет его."""
    system_prompt = (
        files("ai_task_manager")
        .joinpath("skills", "task-extraction", "SKILL.md")
        .read_text(encoding="utf-8")
    )
    user_message = json.dumps(data.model_dump(mode="json"), ensure_ascii=False)

    response = client.post(
        "/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "format": AnalystResult.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0},
        },
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]
    result = AnalystResult.model_validate_json(content)
    return validate_task_references(data, result)
