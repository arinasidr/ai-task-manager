import json
from importlib.resources import files

from ai_task_manager.agents.llm import JsonLLM

from .schema import AnalystInput, AnalystResult
from .validation import validate_task_references


def analyze(data: AnalystInput, *, llm: JsonLLM) -> AnalystResult:
    """Получает разбор запроса от модели и проверяет его."""
    system_prompt = (
        files("ai_task_manager")
        .joinpath("skills", "task-extraction", "SKILL.md")
        .read_text(encoding="utf-8")
    )
    user_message = json.dumps(data.model_dump(mode="json"), ensure_ascii=False)

    answer = llm.complete_json(
        system_prompt,
        user_message,
        AnalystResult.model_json_schema(),
    )
    result = AnalystResult.model_validate(answer)
    return validate_task_references(data, result)
