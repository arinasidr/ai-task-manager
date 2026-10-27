"""Сквозные тесты workflow: настоящие агенты, граф и БД; модели — поддельные, по ролям."""

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from ai_task_manager.agents.llm import LLMError
from ai_task_manager.agents.planner import PLANNER_OUTPUT_SCHEMA
from ai_task_manager.agents.validator import VALIDATOR_OUTPUT_SCHEMA
from ai_task_manager.api import chat as chat_api
from ai_task_manager.api.app import create_app
from ai_task_manager.config import Settings, get_settings
from ai_task_manager.db.crud_func import add_dependency, create_task, get_task, list_tasks
from ai_task_manager.db.database import connect, init_db
from ai_task_manager.db.preferences import update_preference
from ai_task_manager.memory.checkpoint import open_checkpointer
from ai_task_manager.prompting import load_skill, skill_metadata
from ai_task_manager.workflow.graph import MAX_VALIDATION_ATTEMPTS, build_graph
from ai_task_manager.workflow.service import create_workflow, run_chat

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=ZoneInfo("Europe/Moscow"))
ACCEPT = {"decision": "accept", "issues": []}


def reject(detail: str) -> dict[str, Any]:
    return {"decision": "reject", "issues": [{"code": "other", "detail": detail}]}


class Models:
    """Поддельные модели: ответы ставятся в очередь по ролям, вызовы записываются."""

    def __init__(self) -> None:
        self.queues: dict[str, list[Any]] = defaultdict(list)
        self.calls: list[dict[str, Any]] = []

    def answer(self, role: str, *answers: Any) -> None:
        self.queues[role].extend(answers)

    def llm_for(self, model: str) -> "Models._Client":
        return Models._Client(self, model)

    def roles(self) -> list[str]:
        return [call["role"] for call in self.calls]

    class _Client:
        def __init__(self, owner: "Models", model: str) -> None:
            self.owner, self.model = owner, model

        def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict:
            if schema is PLANNER_OUTPUT_SCHEMA:
                role = "planner"
            elif schema is VALIDATOR_OUTPUT_SCHEMA:
                role = "validator"
            else:
                role = "analyst"
            self.owner.calls.append(
                {"role": role, "model": self.model, "user": json.loads(user), "system": system}
            )
            queue = self.owner.queues[role]
            if not queue:
                pytest.fail(f"Неожиданный вызов модели в роли {role}")
            answer = queue.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "tasks.db"
    init_db(path)
    return path


@pytest.fixture
def db(db_path: Path) -> Iterator[sqlite3.Connection]:
    with closing(connect(db_path)) as connection:
        yield connection


@pytest.fixture
def models() -> Models:
    return Models()


@pytest.fixture
def settings(db_path: Path) -> Settings:
    return Settings(_env_file=None, database_path=str(db_path))


@pytest.fixture
def chat(settings: Settings, db_path: Path, models: Models):
    graph = create_workflow(settings, connect=lambda: connect(db_path), llm_for=models.llm_for)

    def send(message: str):
        return run_chat(graph, settings, message, now=NOW)

    return send


def add(db: sqlite3.Connection, title: str, **fields: Any) -> int:
    task_id = create_task(db, title, **fields)
    db.commit()
    return task_id


# ---------------------------------------------------------------- действия


def test_create_task_end_to_end(chat, models: Models, db: sqlite3.Connection) -> None:
    models.answer(
        "analyst",
        {"intent": "create_task", "tasks": [{"title": "Купить молоко", "deadline": "2026-10-15"}]},
    )
    models.answer("validator", ACCEPT)

    result = chat("Добавь задачу купить молоко на завтра")

    assert result.reply.startswith("Готово: создана задача")
    assert result.action_id is None
    assert [(t["title"], t["due_date"]) for t in list_tasks(db)] == [
        ("Купить молоко", "2026-10-15")
    ]
    # аналитик и планировщик — Gemma, валидатор — Mistral; планировщик по правилам, без модели
    assert models.roles() == ["analyst", "validator"]
    assert result.models == {
        "analyst": "gemma4:e2b",
        "planner": "gemma4:e2b",
        "validator": "ministral-3:3b",
    }


def test_delete_waits_for_confirmation(
    chat, models: Models, db: sqlite3.Connection, settings: Settings
) -> None:
    task = add(db, "Отчёт")
    models.answer("analyst", {"intent": "delete_task", "task_id": task})
    models.answer("validator", ACCEPT)

    result = chat(f"Удали задачу #{task}")

    assert result.action_id
    assert "Что изменится" in result.reply
    assert get_task(db, task) is not None  # без подтверждения ничего не удалено


def test_rules_plan_rejected_by_validator_is_not_retried(
    chat, models: Models, db: sqlite3.Connection
) -> None:
    task = add(db, "Отчёт")
    models.answer(
        "analyst", {"intent": "update_task", "task_id": task, "updates": {"priority": "high"}}
    )
    models.answer("validator", reject("Пользователь не просил менять приоритет"))

    result = chat(f"Сделай задачу #{task} срочной")

    assert models.roles() == ["analyst", "validator"]  # без повторов
    assert "Пользователь не просил менять приоритет" in result.reply
    assert "Данные не изменены" in result.reply
    assert get_task(db, task)["priority"] is None


# ---------------------------------------------------------------- планирование и повторы


def order_answer(ids: list[int]) -> dict[str, Any]:
    return {"ordered_task_ids": ids, "recommendations": [], "message": "Начните с первой."}


def retry_graph(verdicts: list[dict], source: str = "llm"):
    """Граф с поддельными узлами: проверяем сам механизм повторов."""
    calls: dict[str, list] = {"planner": [], "validator": 0}

    def analyst(_: dict) -> dict:
        return {"analysis": {"intent": "plan_tasks"}}

    def planner(state: dict) -> dict:
        calls["planner"].append(list(state.get("validator_feedback") or []))
        return {"plan": {"intent": "plan_tasks", "source": source}, "verdict": None}

    def validator(state: dict) -> dict:
        verdict = verdicts[calls["validator"]]
        calls["validator"] += 1
        reasons = [i["detail"] for i in verdict.get("issues", [])]
        return {
            "verdict": verdict,
            "validator_feedback": reasons,
            "validation_attempts": state.get("validation_attempts", 0) + 1,
        }

    graph = build_graph(
        Settings(_env_file=None), analyst=analyst, planner=planner, validator=validator
    )
    state = graph.invoke(
        {
            "request": "С чего начать?",
            "conversation_id": None,
            "current_datetime": NOW.isoformat(),
            "timezone": "Europe/Moscow",
        }
    )
    return state, calls


def test_rejected_llm_plan_is_retried_with_feedback() -> None:
    state, calls = retry_graph([reject("Сначала задача B"), ACCEPT])
    # второй раз планировщик получает замечания валидатора
    assert calls["planner"] == [[], ["Сначала задача B"]]
    assert state["validation_attempts"] == 2
    assert state["verdict"]["decision"] == "accept"


def test_retries_are_limited() -> None:
    state, calls = retry_graph([reject("Плохо 1"), reject("Плохо 2"), ACCEPT])
    assert len(calls["planner"]) == MAX_VALIDATION_ATTEMPTS
    assert state["validation_attempts"] == MAX_VALIDATION_ATTEMPTS
    assert "Плохо 2" in state["reply"]
    assert "Данные не изменены" in state["reply"]


def test_rules_plan_is_not_retried() -> None:
    state, calls = retry_graph([reject("Не то"), ACCEPT], source="rules")
    assert len(calls["planner"]) == 1
    assert "Не то" in state["reply"]


def test_plan_tasks_end_to_end_without_validator_model(
    chat, models: Models, db: sqlite3.Connection
) -> None:
    a, b = add(db, "A", due_date="2026-10-20"), add(db, "B", due_date="2026-10-16")
    models.answer("analyst", {"intent": "plan_tasks"})
    models.answer("planner", order_answer([b, a]))

    result = chat("С чего начать?")

    # порядок проверяет код, модель-валидатор не нужна: план ничего не меняет
    assert models.roles() == ["analyst", "planner"]
    assert result.reply.splitlines()[1].startswith(f"1. #{b}")


def test_preferences_reach_planner(chat, models: Models, db: sqlite3.Connection) -> None:
    a = add(db, "A")
    update_preference(db, "preferred_work_time", "evening")
    db.commit()
    models.answer("analyst", {"intent": "plan_tasks"})
    models.answer("planner", order_answer([a]))
    models.answer("validator", ACCEPT)

    chat("С чего начать?")

    planner_call = next(c for c in models.calls if c["role"] == "planner")
    assert planner_call["user"]["preferences"] == {"preferred_work_time": "evening"}


# ---------------------------------------------------------------- без плана


def test_simple_read_uses_fast_model(chat, models: Models, db: sqlite3.Connection) -> None:
    add(db, "Отчёт")
    models.answer("analyst", {"intent": "get_tasks"})

    result = chat("Покажи задачи")

    assert models.calls[0]["model"] == "qwen3:1.7b"
    assert result.route == "simple_read"
    assert "«Отчёт»" in result.reply
    # план только на чтение принимает код, модель валидатора не вызывается
    assert "validator" not in models.roles()


def test_clarification_from_analyst(chat, models: Models) -> None:
    models.answer(
        "analyst",
        {
            "intent": "clarification_required",
            "missing_fields": ["task_id"],
            "clarification_question": "Укажи номер задачи.",
        },
    )
    result = chat("Удали отчёт")
    assert result.reply == "Укажи номер задачи."
    assert models.roles() == ["analyst"]


def test_unknown_task_id_from_analyst(chat, models: Models, db: sqlite3.Connection) -> None:
    add(db, "Отчёт")
    models.answer("analyst", {"intent": "delete_task", "task_id": 99})
    result = chat("Удали задачу #99")
    assert result.reply.startswith("Задачи #99 нет")
    assert "Неизвестные ID" in result.error


def test_model_unavailable(chat, models: Models) -> None:
    models.answer("analyst", LLMError("Ошибка вызова модели gemma4:e2b: connection refused"))
    result = chat("Добавь задачу")
    assert "недоступна" in result.reply
    assert "Данные не изменены" in result.reply


def test_dialogue_memory_through_checkpointer(
    settings: Settings, db_path: Path, models: Models, tmp_path: Path
) -> None:
    """Workflow + память A11: один conversation_id, история включает итоговые ответы."""
    models.answer(
        "analyst",
        {"intent": "create_task", "tasks": [{"title": "Купить молоко"}]},
        {"intent": "get_tasks"},
    )
    models.answer("validator", ACCEPT)
    with open_checkpointer(tmp_path / "memory.db") as checkpointer:
        graph = create_workflow(
            settings,
            connect=lambda: connect(db_path),
            llm_for=models.llm_for,
            checkpointer=checkpointer,
        )
        first = run_chat(graph, settings, "Добавь задачу купить молоко", now=NOW)
        second = run_chat(graph, settings, "Покажи задачи", first.conversation_id, now=NOW)

    assert second.conversation_id == first.conversation_id
    assert first.history_length == 2
    assert second.history_length == 4
    assert "«Купить молоко»" in second.reply
    assert second.plan["operations"][0]["tool"] == "list_tasks"


def test_repaired_dependency_reaches_validator(
    chat, models: Models, db: sqlite3.Connection
) -> None:
    """Живой прогон: аналитик связал задачу саму с собой, код восстановил порядок по запросу.

    Валидатор должен видеть исправленную связь, иначе он отклоняет верный план.
    """
    titles = ["Собрать данные для отчёта", "Написать отчёт по практике"]
    models.answer(
        "analyst",
        {
            "intent": "create_task",
            "tasks": [{"title": title} for title in titles],
            "dependencies": [{"task": titles[0], "depends_on": titles[0]}],
        },
    )
    models.answer("validator", ACCEPT)

    result = chat("Добавь две задачи: сначала собрать данные для отчёта, потом написать отчёт")

    validator_input = models.calls[-1]["user"]
    assert validator_input["analysis"]["dependencies"] == [
        {"task": titles[1], "depends_on": titles[0]}
    ]
    assert result.reply.startswith("Готово")
    first, second = (task["id"] for task in list_tasks(db))
    assert [
        row["predecessor_task_id"]
        for row in db.execute(
            "SELECT predecessor_task_id FROM task_dependencies WHERE successor_task_id = ?",
            (second,),
        )
    ] == [first]


# ---------------------------------------------------------------- HTTP API


@pytest.fixture
def client(
    db_path: Path, settings: Settings, models: Models, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setenv("ATM_DATABASE_PATH", str(db_path))
    get_settings.cache_clear()
    app = create_app()
    graph = create_workflow(settings, connect=lambda: connect(db_path), llm_for=models.llm_for)
    app.dependency_overrides[chat_api.get_workflow] = lambda: graph
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        get_settings.cache_clear()


def test_chat_api_delete_then_confirm(
    client: TestClient, models: Models, db: sqlite3.Connection
) -> None:
    a, b = add(db, "Собрать данные"), add(db, "Написать отчёт")
    add_dependency(db, a, b)
    db.commit()
    models.answer("analyst", {"intent": "delete_task", "task_id": a})
    models.answer("validator", ACCEPT)

    response = client.post("/api/v1/chat", json={"message": f"Удали задачу #{a}"})
    assert response.status_code == 200
    body = response.json()
    assert body["action_id"]
    assert f"#{a} «Собрать данные»" in body["reply"]
    assert body["conversation_id"]

    confirmed = client.post(f"/api/v1/actions/{body['action_id']}/confirm").json()
    assert confirmed["applied"] is True
    assert get_task(db, a) is None
    assert get_task(db, b) is not None  # удалена только выбранная задача


# ---------------------------------------------------------------- скиллы


@pytest.mark.parametrize("name", ["task-planning", "task-validation"])
def test_skills_have_metadata_and_procedure(name: str) -> None:
    meta = skill_metadata(name)
    assert meta["name"] == name
    assert len(meta["description"]) > 40
    body = load_skill(name)
    assert body.startswith(f"# {name}")
    for section in ("## Назначение", "## Когда применять", "## Порядок работы"):
        assert section in body
    assert "```json" not in body  # без готовых ответов: маленькие модели их копируют


def test_agent_prompts_fit_small_context() -> None:
    from ai_task_manager.agents.planner import load_system_prompt as planner_prompt
    from ai_task_manager.agents.validator import load_system_prompt as validator_prompt

    # контекст моделей 4096 токенов: промпт + задачи + ответ; ~3 символа на токен
    for prompt in (planner_prompt(), validator_prompt()):
        assert len(prompt) < 7000
