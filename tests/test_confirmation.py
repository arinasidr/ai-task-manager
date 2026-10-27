"""Подтверждение опасных операций из агентного пути: исполнитель плана, ожидающее действие,
подтверждение, отмена, узел графа и HTTP API. Без LLM и сети."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ai_task_manager.agents.contracts import Plan
from ai_task_manager.agents.planner import Planner
from ai_task_manager.api.app import create_app
from ai_task_manager.config import get_settings
from ai_task_manager.db.crud_func import (
    add_dependency,
    cancel_operation,
    create_task,
    get_dependencies,
    get_operation,
    get_task,
    list_tasks,
    update_task,
)
from ai_task_manager.db.database import connect, init_db
from ai_task_manager.db.preferences import get_preferences
from ai_task_manager.tools.confirmation import (
    ACTION_TYPE,
    confirm_plan_action,
    describe_plan,
    make_action_node,
    prepare_plan_action,
)
from ai_task_manager.tools.executor import (
    ConfirmationRequired,
    confirmation_reasons,
    execute_plan,
    run_plan,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "tasks.db"
    init_db(path)
    return path


@pytest.fixture
def db(db_path: Path) -> Iterator[sqlite3.Connection]:
    with closing(connect(db_path)) as connection:
        yield connection


def add(db: sqlite3.Connection, title: str, **fields: Any) -> int:
    task_id = create_task(db, title, **fields)
    db.commit()
    return task_id


def link(db: sqlite3.Connection, predecessor: int, successor: int) -> None:
    add_dependency(db, predecessor, successor)
    db.commit()


def plan_for(db: sqlite3.Connection, analysis: dict[str, Any]) -> Plan:
    return Planner().plan(analysis, db)


def prepare(db: sqlite3.Connection, plan: Plan, request: str = "запрос") -> str:
    db.execute("BEGIN IMMEDIATE")
    pending = prepare_plan_action(db, plan, request)
    db.commit()
    return pending.action_id


# ---------------------------------------------------------------- исполнитель


def test_create_with_dependency_resolves_refs(db: sqlite3.Connection) -> None:
    plan = plan_for(
        db,
        {
            "intent": "create_task",
            "tasks": [
                {"title": "Изучить материалы", "deadline": "2026-11-01"},
                {"title": "Подготовить доклад", "deadline": "2026-11-03T18:00:00+03:00"},
            ],
            "dependencies": [{"task": "Подготовить доклад", "depends_on": "Изучить материалы"}],
        },
    )
    result = run_plan(db, plan)
    first, second = (task.task_id for task in result.created)
    assert get_dependencies(db, second)["predecessors"] == [first]
    assert get_task(db, second)["due_time"] == "18:00:00"
    assert "создана задача" in result.summary()


def test_safe_update_runs_without_confirmation(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт", due_date="2026-11-01", due_time="10:00")
    plan = plan_for(
        db, {"intent": "update_task", "task_id": task, "updates": {"deadline": "2026-11-05"}}
    )
    run_plan(db, plan)
    stored = get_task(db, task)
    # перенос «на дату» убирает старое время
    assert (stored["due_date"], stored["due_time"]) == ("2026-11-05", None)


def test_preference_is_saved(db: sqlite3.Connection) -> None:
    plan = plan_for(
        db, {"intent": "update_preference", "updates": {"preferred_work_time": "evening"}}
    )
    result = run_plan(db, plan)
    assert get_preferences(db) == {"preferred_work_time": "evening"}
    assert "запомнил" in result.summary()


def test_get_tasks_returns_list_without_writing(db: sqlite3.Connection) -> None:
    add(db, "Отчёт")
    result = run_plan(db, plan_for(db, {"intent": "get_tasks"}))
    assert [task["title"] for task in result.tasks] == ["Отчёт"]


def test_executor_refuses_delete_even_if_plan_says_no_confirmation(
    db: sqlite3.Connection,
) -> None:
    task = add(db, "Отчёт")
    plan = plan_for(db, {"intent": "delete_task", "task_id": task})
    plan.requires_confirmation = False  # подменённый или ошибочный план
    with pytest.raises(ConfirmationRequired, match="удаление задачи"):
        run_plan(db, plan)
    assert get_task(db, task) is not None


def test_executor_refuses_blocked_completion_without_confirmation(
    db: sqlite3.Connection,
) -> None:
    a, b = add(db, "A"), add(db, "B")
    link(db, a, b)
    plan = Plan(
        intent="update_task",
        operations=[{"tool": "change_task_status", "args": {"task_id": b, "status": "done"}}],
    )
    assert confirmation_reasons(db, plan)
    with pytest.raises(ConfirmationRequired):
        run_plan(db, plan)


def test_failed_plan_is_rolled_back_completely(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    plan = Plan(
        intent="update_task",
        operations=[
            {"tool": "update_task", "args": {"task_id": task, "updates": {"priority": "high"}}},
            {"tool": "update_task", "args": {"task_id": 999, "updates": {"priority": "low"}}},
        ],
    )
    with pytest.raises(ValueError, match="#999"):
        run_plan(db, plan)
    assert get_task(db, task)["priority"] is None  # первое изменение тоже откатилось


def test_clarification_plan_is_not_executed(db: sqlite3.Connection) -> None:
    plan = plan_for(db, {"intent": "clarification_required", "missing_fields": ["task_id"]})
    with pytest.raises(ValueError, match="уточнения"):
        execute_plan(db, plan)


# ---------------------------------------------------------------- предварительный план


def test_preview_names_tasks_and_detached_links(db: sqlite3.Connection) -> None:
    a, b = add(db, "Собрать данные"), add(db, "Написать отчёт")
    link(db, a, b)
    preview = describe_plan(db, plan_for(db, {"intent": "delete_task", "task_id": a}))
    assert preview == [f"Удалить #{a} «Собрать данные» безвозвратно; связи с #{b} будут удалены"]


def test_preview_for_blocked_completion_and_update(db: sqlite3.Connection) -> None:
    a, b = add(db, "A"), add(db, "B")
    link(db, a, b)
    preview = describe_plan(
        db,
        plan_for(
            db,
            {
                "intent": "update_task",
                "task_id": b,
                "updates": {"status": "done", "priority": "high", "deadline": "2026-11-05"},
            },
        ),
    )
    assert preview == [
        f"Изменить #{b} «B»: приоритет → высокий, срок → 2026-11-05",
        f"Отметить #{b} «B» выполненной, хотя не завершены задачи #{a}",
    ]


# ---------------------------------------------------------------- ожидающее действие


def test_prepare_does_not_change_tasks(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    action_id = prepare(db, plan_for(db, {"intent": "delete_task", "task_id": task}))
    operation = get_operation(db, action_id)
    assert operation["type"] == ACTION_TYPE
    assert operation["status"] == "pending"
    assert get_task(db, task) is not None
    payload = json.loads(operation["payload_json"])
    assert payload["preview"][0].startswith(f"Удалить #{task}")


def test_confirm_executes_plan_once(db_path: Path, db: sqlite3.Connection) -> None:
    a, b = add(db, "A"), add(db, "B")
    action_id = prepare(db, plan_for(db, {"intent": "delete_tasks", "task_ids": [a, b]}))

    result = confirm_plan_action(action_id, db_path)
    assert result.deleted == [a, b]
    assert list_tasks(db) == []
    assert get_operation(db, action_id)["status"] == "confirmed"
    # повторное подтверждение ничего не делает
    assert confirm_plan_action(action_id, db_path) is None


def test_blocked_completion_after_confirmation(db_path: Path, db: sqlite3.Connection) -> None:
    a, b = add(db, "A"), add(db, "B")
    link(db, a, b)
    plan = plan_for(db, {"intent": "update_task", "task_id": b, "updates": {"status": "done"}})
    assert plan.requires_confirmation
    confirm_plan_action(prepare(db, plan), db_path)
    assert get_task(db, b)["status"] == "done"
    assert get_task(db, a)["status"] == "todo"  # остальные задачи не тронуты


def test_cancelled_action_cannot_be_confirmed(db_path: Path, db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    action_id = prepare(db, plan_for(db, {"intent": "delete_task", "task_id": task}))
    assert cancel_operation(db, action_id)
    db.commit()
    with pytest.raises(ValueError, match="уже обработана"):
        confirm_plan_action(action_id, db_path)
    assert get_task(db, task) is not None


def test_task_changed_after_preview_blocks_confirmation(
    db_path: Path, db: sqlite3.Connection
) -> None:
    task = add(db, "Отчёт")
    action_id = prepare(db, plan_for(db, {"intent": "delete_task", "task_id": task}))
    update_task(db, task, {"title": "Новый отчёт"})  # пользователь видел другую задачу
    db.commit()
    with pytest.raises(ValueError, match="изменилась"):
        confirm_plan_action(action_id, db_path)
    assert get_task(db, task) is not None


def test_expired_action_is_not_executed(db_path: Path, db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    action_id = prepare(db, plan_for(db, {"intent": "delete_task", "task_id": task}))
    past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    db.execute("UPDATE operations SET expires_at = ? WHERE action_id = ?", (past, action_id))
    db.commit()
    with pytest.raises(ValueError, match="истекло"):
        confirm_plan_action(action_id, db_path)
    assert get_task(db, task) is not None


def test_plan_rechecked_on_confirmation(db_path: Path, db: sqlite3.Connection) -> None:
    a, b = add(db, "A"), add(db, "B")
    plan = Plan(
        intent="update_task",
        requires_confirmation=True,
        operations=[
            {"tool": "change_task_status", "args": {"task_id": b, "status": "done"}},
            {"tool": "add_dependency", "args": {"predecessor_task_id": a, "successor_task_id": b}},
        ],
    )
    action_id = prepare(db, plan)
    # пока ждали подтверждения, появилась обратная связь — теперь план создаст цикл;
    # версии задач тоже изменились, поэтому подтверждение отклоняется
    link(db, b, a)
    with pytest.raises(ValueError):
        confirm_plan_action(action_id, db_path)
    assert get_task(db, b)["status"] == "todo"


def test_tampered_stored_plan_fails_recheck(db_path: Path, db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    action_id = prepare(db, plan_for(db, {"intent": "delete_task", "task_id": task}))
    # кто-то подменил сохранённый план в БД: версии задач те же, но план уже без подтверждения
    payload = json.loads(get_operation(db, action_id)["payload_json"])
    payload["plan"]["requires_confirmation"] = False
    db.execute(
        "UPDATE operations SET payload_json = ? WHERE action_id = ?",
        (json.dumps(payload, ensure_ascii=False), action_id),
    )
    db.commit()
    with pytest.raises(ValueError, match="больше не корректен"):
        confirm_plan_action(action_id, db_path)
    assert get_task(db, task) is not None
    assert get_operation(db, action_id)["status"] == "pending"


def test_prepare_rejects_plans_without_existing_tasks(db: sqlite3.Connection) -> None:
    plan = plan_for(db, {"intent": "create_task", "tasks": [{"title": "A"}]})
    with pytest.raises(ValueError):
        prepare_plan_action(db, plan, "x")


# ---------------------------------------------------------------- узел графа


def node_state(plan: Plan, decision: str = "accept", request: str = "запрос") -> dict[str, Any]:
    return {"request": request, "plan": plan.model_dump(), "verdict": {"decision": decision}}


def test_node_puts_dangerous_plan_on_hold(db_path: Path, db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    node = make_action_node(connect=lambda: connect(db_path))
    update = node(node_state(plan_for(db, {"intent": "delete_task", "task_id": task})))
    assert update["pending_action_id"]
    assert "Что изменится" in update["reply"]
    assert "пока не изменены" in update["reply"]
    assert get_task(db, task) is not None


def test_node_executes_safe_plan(db_path: Path, db: sqlite3.Connection) -> None:
    node = make_action_node(connect=lambda: connect(db_path))
    plan = plan_for(db, {"intent": "create_task", "tasks": [{"title": "Купить молоко"}]})
    update = node(node_state(plan))
    assert update["pending_action_id"] is None
    assert "создана задача" in update["reply"]
    assert [t["title"] for t in list_tasks(db)] == ["Купить молоко"]


def test_node_ignores_rejected_plan(db_path: Path, db: sqlite3.Connection) -> None:
    node = make_action_node(connect=lambda: connect(db_path))
    plan = plan_for(db, {"intent": "create_task", "tasks": [{"title": "A"}]})
    assert node(node_state(plan, decision="reject")) == {}
    assert list_tasks(db) == []


def test_node_holds_plan_even_if_flag_was_lost(db_path: Path, db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    plan = plan_for(db, {"intent": "delete_task", "task_id": task})
    plan.requires_confirmation = False
    update = make_action_node(connect=lambda: connect(db_path))(node_state(plan))
    assert update["pending_action_id"]
    assert get_task(db, task) is not None


def test_node_replies_for_read_order_and_clarification(
    db_path: Path, db: sqlite3.Connection
) -> None:
    a = add(db, "Собрать данные", due_date="2026-11-01")
    node = make_action_node(connect=lambda: connect(db_path))
    assert (
        f"#{a} «Собрать данные»" in node(node_state(plan_for(db, {"intent": "get_tasks"})))["reply"]
    )
    order = node(node_state(plan_for(db, {"intent": "plan_tasks"})))["reply"]
    assert f"1. #{a} «Собрать данные»" in order
    clarify = plan_for(db, {"intent": "clarification_required", "missing_fields": ["task_id"]})
    assert "ID" in node(node_state(clarify))["reply"]


# ---------------------------------------------------------------- HTTP API


@pytest.fixture
def client(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("ATM_DATABASE_PATH", str(db_path))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        get_settings.cache_clear()


def test_api_confirms_agent_plan(client: TestClient, db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    action_id = prepare(db, plan_for(db, {"intent": "delete_task", "task_id": task}))

    response = client.post(f"/api/v1/actions/{action_id}/confirm")
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert f"удалена задача #{task}" in body["message"]
    assert get_task(db, task) is None

    again = client.post(f"/api/v1/actions/{action_id}/confirm").json()
    assert again["applied"] is False


def test_api_cancels_agent_plan(client: TestClient, db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    action_id = prepare(db, plan_for(db, {"intent": "delete_task", "task_id": task}))
    assert client.post(f"/api/v1/actions/{action_id}/cancel").json()["applied"] is True
    assert client.post(f"/api/v1/actions/{action_id}/confirm").status_code == 409
    assert get_task(db, task) is not None
