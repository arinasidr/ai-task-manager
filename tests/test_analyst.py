import json
from typing import Any

import pytest
from pydantic import ValidationError

from ai_task_manager.agents.analyst.client import analyze
from ai_task_manager.agents.analyst.schema import AnalystInput


def make_input(request: str, tasks: list[dict] | None = None) -> AnalystInput:
    return AnalystInput.model_validate(
        {
            "request": request,
            "current_datetime": "2026-10-25T12:00:00+03:00",
            "timezone": "Europe/Moscow",
            "tasks": tasks or [],
        }
    )


class FakeLLM:
    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer = answer

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        assert system.startswith("# SOUL - AI Task Manager")
        assert "# task-extraction" in system
        assert json.loads(user)["request"]
        assert schema["title"] == "AnalystResult"
        return self.answer


def test_analyst_extracts_task() -> None:
    answer = {
        "intent": "create_task",
        "tasks": [{"title": "Купить молоко", "priority": None, "deadline": "2026-10-26"}],
    }

    result = analyze(make_input("Добавь задачу купить молоко завтра"), llm=FakeLLM(answer))

    assert result.intent == "create_task"
    assert result.tasks[0].deadline == "2026-10-26"


def test_analyst_rejects_unknown_task_id() -> None:
    tasks = [{"id": 1, "title": "Отчёт", "status": "todo"}]

    with pytest.raises(ValueError, match="Неизвестные ID"):
        analyze(
            make_input("Удали задачу #99", tasks),
            llm=FakeLLM({"intent": "delete_task", "task_id": 99}),
        )


def test_analyst_asks_for_id_not_mentioned_by_user() -> None:
    tasks = [{"id": 1, "title": "Отчёт", "status": "todo"}]

    result = analyze(
        make_input("Удали отчёт", tasks), llm=FakeLLM({"intent": "delete_task", "task_id": 1})
    )

    assert result.intent == "clarification_required"
    assert result.missing_fields == ["task_id"]
    assert result.task_id is None


def test_analyst_rejects_empty_create_result() -> None:
    with pytest.raises(ValidationError):
        analyze(make_input("Добавь задачу купить молоко"), llm=FakeLLM({"intent": "create_task"}))
