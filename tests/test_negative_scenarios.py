"""A17: негативные сценарии через рабочий граф и временную SQLite-базу."""

from collections import defaultdict
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ai_task_manager.agents.planner import PLANNER_OUTPUT_SCHEMA
from ai_task_manager.agents.validator import VALIDATOR_OUTPUT_SCHEMA
from ai_task_manager.config import Settings
from ai_task_manager.db.crud_func import create_task
from ai_task_manager.db.database import connect, init_db
from ai_task_manager.db.preferences import get_preferences
from ai_task_manager.memory.checkpoint import open_checkpointer
from ai_task_manager.workflow.service import create_workflow, run_chat

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=ZoneInfo("Europe/Moscow"))
ACCEPT = {"decision": "accept", "issues": []}


class ScriptedModels:
    """Заранее заданные ответы моделей: тест не зависит от Ollama."""

    def __init__(self) -> None:
        self.answers: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(self, role: str, *answers: dict[str, Any]) -> None:
        self.answers[role].extend(answers)

    def llm_for(self, model: str) -> "ScriptedModels":
        return self

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        if schema is PLANNER_OUTPUT_SCHEMA:
            role = "planner"
        elif schema is VALIDATOR_OUTPUT_SCHEMA:
            role = "validator"
        else:
            role = "analyst"

        if not self.answers[role]:
            raise AssertionError(f"Неожиданный вызов модели: {role}")
        return self.answers[role].pop(0)


def add_task(path: Path, title: str) -> int:
    with closing(connect(path)) as db:
        with db:
            return create_task(db, title)


def task_snapshot(path: Path) -> tuple[list[tuple], list[tuple]]:
    with closing(connect(path)) as db:
        tasks = [tuple(row) for row in db.execute("SELECT * FROM tasks ORDER BY id")]
        links = [
            tuple(row)
            for row in db.execute(
                "SELECT * FROM task_dependencies ORDER BY predecessor_task_id, successor_task_id"
            )
        ]
    return tasks, links


@contextmanager
def scenario(path: Path, models: ScriptedModels) -> Iterator[tuple[Any, Settings]]:
    settings = Settings(
        _env_file=None,
        database_path=str(path),
        log_dir=None,
        phoenix_endpoint=None,
    )
    with open_checkpointer(path.with_name("history.db")) as checkpointer:
        graph = create_workflow(
            settings,
            connect=lambda: connect(path),
            llm_for=models.llm_for,
            checkpointer=checkpointer,
            decision_node=lambda _: {"decision": None, "decision_error": None},
        )
        yield graph, settings


def test_ambiguous_reference_does_not_delete_a_guessed_task(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.db"
    init_db(path)
    first_id = add_task(path, "Написать отчёт")
    add_task(path, "Отправить письмо")
    before = task_snapshot(path)

    models = ScriptedModels()
    models.add(
        "analyst",
        {"intent": "get_tasks"},
        # Модель угадала одну из двух задач. Пользователь этого ID не называл.
        {"intent": "delete_task", "task_id": first_id},
    )

    with scenario(path, models) as (graph, settings):
        listed = run_chat(graph, settings, "Покажи мои задачи", now=NOW)
        result = run_chat(graph, settings, "Удали её", listed.conversation_id, now=NOW)

    assert result.analysis["intent"] == "clarification_required"
    assert "номер" in result.reply.casefold()
    assert result.action_id is None
    assert "act" not in [step["node"] for step in result.trajectory]
    assert task_snapshot(path) == before


def test_invented_dependency_does_not_change_task(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.db"
    init_db(path)
    first_id = add_task(path, "Написать отчёт")
    second_id = add_task(path, "Отправить письмо")
    before = task_snapshot(path)

    models = ScriptedModels()
    models.add(
        "analyst",
        {
            "intent": "update_task",
            "task_id": first_id,
            "updates": {
                "priority": "high",
                # Пользователь не просил связывать задачи.
                "add_dependency_ids": [second_id],
            },
        },
    )

    with scenario(path, models) as (graph, settings):
        result = run_chat(graph, settings, f"Повышай приоритет задачи #{first_id}", now=NOW)

    assert result.analysis["intent"] == "clarification_required"
    assert result.action_id is None
    assert task_snapshot(path) == before


def test_yes_in_chat_does_not_confirm_pending_deletion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.db"
    init_db(path)
    task_id = add_task(path, "Черновик")
    before = task_snapshot(path)

    models = ScriptedModels()
    models.add(
        "analyst",
        {"intent": "delete_task", "task_id": task_id},
        # Даже если модель трактует «да» как новую команду удаления.
        {"intent": "delete_task", "task_id": task_id},
    )
    models.add("validator", ACCEPT)

    with scenario(path, models) as (graph, settings):
        pending = run_chat(graph, settings, f"Удали задачу #{task_id}", now=NOW)
        assert pending.action_id is not None
        answer = run_chat(graph, settings, "Да, удаляй", pending.conversation_id, now=NOW)

    with closing(connect(path)) as db:
        operation = db.execute(
            "SELECT status FROM operations WHERE action_id = ?",
            (pending.action_id,),
        ).fetchone()

    assert answer.analysis["intent"] == "clarification_required"
    assert answer.action_id is None
    assert operation["status"] == "pending"
    assert task_snapshot(path) == before


def test_old_preference_proposal_cannot_override_new_explicit_choice(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.db"
    init_db(path)

    models = ScriptedModels()
    models.add(
        "analyst",
        {"intent": "create_task", "tasks": [{"title": "Созвон с командой"}]},
        {
            "intent": "update_preference",
            "preference": {"kind": "working_days", "value": "morning"},
        },
        {
            "intent": "update_preference",
            "preference": {"kind": "working_days", "value": "morning"},
        },
    )
    models.add("validator", ACCEPT, ACCEPT)

    with scenario(path, models) as (graph, settings):
        first = run_chat(
            graph,
            settings,
            "Добавь задачу созвон с командой, мне удобнее работать вечером",
            now=NOW,
        )
        assert first.preference_id is not None

        changed = run_chat(
            graph,
            settings,
            "Запомни, что мне удобнее работать утром",
            first.conversation_id,
            now=NOW,
        )
        assert changed.analysis["preference"] == {
            "kind": "preferred_work_time",
            "value": "утром",
        }

        answer = run_chat(graph, settings, "Да", changed.conversation_id, now=NOW)
        assert answer.analysis["intent"] == "clarification_required"

    with closing(connect(path)) as db:
        statuses = [
            row["status"] for row in db.execute("SELECT status FROM preferences ORDER BY id")
        ]
        preferences = get_preferences(db)

    assert statuses == ["superseded", "confirmed"]
    assert preferences == {"preferred_work_time": "утром"}
