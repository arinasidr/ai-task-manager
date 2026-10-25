import json

import httpx
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


def fake_ollama(answer: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"][0]["content"].startswith("# task-extraction")
        assert request.url.path == "/api/chat"
        assert payload["model"] == "gemma4:e2b"
        assert payload["format"]["title"] == "AnalystResult"

        return httpx.Response(
            200,
            json={"message": {"content": json.dumps(answer, ensure_ascii=False)}},
        )

    return httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )


def test_analyst_extracts_task() -> None:
    answer = {
        "intent": "create_task",
        "tasks": [{"title": "Купить молоко", "priority": None, "deadline": "2026-10-26"}],
    }

    with fake_ollama(answer) as client:
        result = analyze(
            make_input("Добавь задачу купить молоко завтра"),
            model="gemma4:e2b",
            client=client,
        )

    assert result.intent == "create_task"
    assert result.tasks[0].deadline == "2026-10-26"


def test_analyst_rejects_unknown_task_id() -> None:
    tasks = [{"id": 1, "title": "Отчёт", "status": "todo"}]

    with fake_ollama({"intent": "delete_task", "task_id": 99}) as client:
        with pytest.raises(ValueError, match="Неизвестные ID"):
            analyze(
                make_input("Удали задачу #99", tasks),
                model="gemma4:e2b",
                client=client,
            )


def test_analyst_asks_for_id_not_mentioned_by_user() -> None:
    tasks = [{"id": 1, "title": "Отчёт", "status": "todo"}]

    with fake_ollama({"intent": "delete_task", "task_id": 1}) as client:
        result = analyze(make_input("Удали отчёт", tasks), model="gemma4:e2b", client=client)

    assert result.intent == "clarification_required"
    assert result.missing_fields == ["task_id"]
    assert result.task_id is None


def test_analyst_rejects_empty_create_result() -> None:
    with fake_ollama({"intent": "create_task"}) as client:
        with pytest.raises(ValidationError):
            analyze(
                make_input("Добавь задачу купить молоко"),
                model="gemma4:e2b",
                client=client,
            )
