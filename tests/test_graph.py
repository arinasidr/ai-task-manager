from typing import Any

import pytest

from ai_task_manager.config import Settings
from ai_task_manager.workflow.graph import build_graph
from ai_task_manager.workflow.state import WorkflowState


def test_bad_fast_answer_cannot_reuse_old_delete_plan() -> None:
    def analyst(state: WorkflowState) -> dict[str, Any]:
        assert state["analyst_model"] == "fast"
        return {"analysis": {"intent": "create_task"}}

    def must_not_run(_: WorkflowState) -> dict[str, Any]:
        pytest.fail("После ошибочного ответа Qwen планировщик запускаться не должен")

    graph = build_graph(
        Settings(
            fast_model="fast",
            analyst_model="deep",
            planner_model="planner",
            validator_model="validator",
        ),
        analyst=analyst,
        planner=must_not_run,
        validator=must_not_run,
    )

    result = graph.invoke(
        {
            "request": "Покажи задачи",
            "conversation_id": None,
            "current_datetime": "2026-10-26T12:00:00+03:00",
            "timezone": "Europe/Moscow",
            "plan": {"operations": [{"tool": "delete_task", "args": {"task_id": 12}}]},
            "verdict": {"decision": "accept"},
        }
    )

    assert result["route"] == "simple_read"
    assert result["plan"] is None
    assert result["verdict"] is None


def test_valid_read_passes_through_agents_in_order() -> None:
    calls: list[str] = []

    def analyst(state: WorkflowState) -> dict[str, Any]:
        calls.append("analyst")
        assert state["analyst_model"] == "fast"
        return {"analysis": {"intent": "get_tasks"}}

    def planner(state: WorkflowState) -> dict[str, Any]:
        calls.append("planner")
        assert state["analysis"]["intent"] == "get_tasks"
        return {"plan": {"intent": "get_tasks"}}

    def validator(state: WorkflowState) -> dict[str, Any]:
        calls.append("validator")
        assert state["plan"]["intent"] == "get_tasks"
        return {"verdict": {"decision": "accept"}}

    graph = build_graph(
        Settings(fast_model="fast", validator_model="validator"),
        analyst=analyst,
        planner=planner,
        validator=validator,
    )
    result = graph.invoke(
        {
            "request": "Покажи задачи",
            "conversation_id": None,
            "current_datetime": "2026-10-26T12:00:00+03:00",
            "timezone": "Europe/Moscow",
        }
    )

    assert calls == ["analyst", "planner", "validator"]
    assert result["verdict"] == {"decision": "accept"}


def test_write_request_uses_main_models() -> None:
    calls: list[str] = []

    def analyst(state: WorkflowState) -> dict[str, Any]:
        calls.append(state["analyst_model"])
        return {
            "analysis": {
                "intent": "create_task",
                "tasks": [{"title": "Купить молоко"}],
            }
        }

    def planner(state: WorkflowState) -> dict[str, Any]:
        calls.append(state["planner_model"])
        assert state["analysis"]["intent"] == "create_task"
        return {"plan": {"operations": [{"tool": "create_task"}]}}

    def validator(state: WorkflowState) -> dict[str, Any]:
        calls.append(state["validator_model"])
        return {"verdict": {"decision": "accept"}}

    graph = build_graph(
        Settings(
            fast_model="fast",
            analyst_model="gemma",
            planner_model="gemma",
            validator_model="ministral",
        ),
        analyst=analyst,
        planner=planner,
        validator=validator,
    )
    result = graph.invoke(
        {
            "request": "Добавь задачу купить молоко",
            "conversation_id": None,
            "current_datetime": "2026-10-26T12:00:00+03:00",
            "timezone": "Europe/Moscow",
        }
    )

    assert result["route"] == "full"
    assert calls == ["gemma", "gemma", "ministral"]
    assert result["verdict"] == {"decision": "accept"}
