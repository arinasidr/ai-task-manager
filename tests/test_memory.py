from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Any

from langgraph.checkpoint.base import empty_checkpoint

from ai_task_manager.config import Settings
from ai_task_manager.memory.checkpoint import config_for_conversation, open_checkpointer, run_turn
from ai_task_manager.workflow.graph import build_graph
from ai_task_manager.workflow.state import WorkflowState


def test_history_survives_restart_and_isolated_by_conversation(tmp_path: Path) -> None:
    def analyst(state: WorkflowState) -> dict[str, Any]:
        return {
            "analysis": {"intent": "unsupported"},
            "reply": f"Ответ: {state['request']}",
        }

    def unused(_: WorkflowState) -> dict[str, Any]:
        raise AssertionError("Этот агент не должен запускаться")

    def make_graph(checkpointer: Any) -> Any:
        return build_graph(
            Settings(_env_file=None),
            analyst=analyst,
            planner=unused,
            validator=unused,
            checkpointer=checkpointer,
        )

    def message(text: str, conversation_id: str) -> WorkflowState:
        return {
            "request": text,
            "conversation_id": conversation_id,
            "current_datetime": "2026-10-27T10:00:00+03:00",
            "timezone": "Europe/Moscow",
        }

    path = tmp_path / "tasks.db"
    with open_checkpointer(path) as checkpointer:
        make_graph(checkpointer).invoke(
            message("Привет", "dialog-1"),
            config_for_conversation("dialog-1"),
        )

    with open_checkpointer(path) as checkpointer:
        graph = make_graph(checkpointer)
        continued = graph.invoke(
            message("Продолжим", "dialog-1"),
            config_for_conversation("dialog-1"),
        )
        separate = graph.invoke(
            message("Другой диалог", "dialog-2"),
            config_for_conversation("dialog-2"),
        )

        for number in (3, 4, 5):
            bounded = graph.invoke(
                message(f"Сообщение {number}", "dialog-1"),
                config_for_conversation("dialog-1"),
            )

    assert [item["content"] for item in continued["history"]] == [
        "Привет",
        "Ответ: Привет",
        "Продолжим",
        "Ответ: Продолжим",
    ]
    assert [item["content"] for item in separate["history"]] == [
        "Другой диалог",
        "Ответ: Другой диалог",
    ]
    assert [item["content"] for item in bounded["history"]] == [
        "Сообщение 3",
        "Ответ: Сообщение 3",
        "Сообщение 4",
        "Ответ: Сообщение 4",
        "Сообщение 5",
        "Ответ: Сообщение 5",
    ]


def test_expired_conversation_is_deleted_on_open(tmp_path: Path) -> None:
    path = tmp_path / "tasks.db"
    now = datetime.now(UTC)

    with open_checkpointer(path) as checkpointer:
        for thread_id, days_ago in (
            ("expired", 8),
            ("active", 8),
            ("active", 1),
        ):
            checkpoint = empty_checkpoint()
            checkpoint["ts"] = (now - timedelta(days=days_ago)).isoformat()
            checkpointer.put(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
                checkpoint,
                {"source": "input", "step": 0, "writes": {}},
                {},
            )

    with open_checkpointer(path) as checkpointer:
        assert checkpointer.get_tuple(config_for_conversation("expired")) is None
        assert len(list(checkpointer.list(config_for_conversation("active")))) == 2


def test_previous_plan_is_cleared_on_new_request(tmp_path: Path) -> None:
    def analyst(state: WorkflowState) -> dict[str, Any]:
        if state["request"] == "Покажи задачи":
            return {"analysis": {"intent": "get_tasks"}}
        assert state["plan"] is None
        assert state["verdict"] is None
        return {"analysis": {"intent": "unsupported"}}

    def planner(_: WorkflowState) -> dict[str, Any]:
        return {"plan": {"intent": "get_tasks"}}

    def validator(_: WorkflowState) -> dict[str, Any]:
        return {"verdict": {"decision": "accept"}}

    with open_checkpointer(tmp_path / "tasks.db") as checkpointer:
        graph = build_graph(
            Settings(_env_file=None),
            analyst=analyst,
            planner=planner,
            validator=validator,
            checkpointer=checkpointer,
        )
        config = config_for_conversation("dialog-1")
        base: WorkflowState = {
            "request": "Покажи задачи",
            "conversation_id": "dialog-1",
            "current_datetime": "2026-10-27T10:00:00+03:00",
            "timezone": "Europe/Moscow",
        }
        first = graph.invoke(base, config)
        second = graph.invoke({**base, "request": "Помоги"}, config)

    assert first["plan"] == {"intent": "get_tasks"}
    assert second["plan"] is None
    assert second["verdict"] is None


def test_run_turn_creates_id_and_continues_dialogue(tmp_path: Path) -> None:
    def analyst(state: WorkflowState) -> dict[str, Any]:
        return {
            "analysis": {"intent": "unsupported"},
            "reply": f"Ответ: {state['request']}",
        }

    def unused(_: WorkflowState) -> dict[str, Any]:
        raise AssertionError("Этот агент не должен запускаться")

    with open_checkpointer(tmp_path / "tasks.db") as checkpointer:
        graph = build_graph(
            Settings(_env_file=None),
            analyst=analyst,
            planner=unused,
            validator=unused,
            checkpointer=checkpointer,
        )
        first_input: WorkflowState = {
            "request": "Привет",
            "conversation_id": None,
            "current_datetime": "2026-10-27T10:00:00+03:00",
            "timezone": "Europe/Moscow",
        }
        first = run_turn(graph, first_input)
        second = run_turn(
            graph,
            {**first_input, "request": "Продолжим", "conversation_id": first["conversation_id"]},
        )

    assert first_input["conversation_id"] is None
    assert first["conversation_id"]
    assert second["conversation_id"] == first["conversation_id"]
    assert [item["content"] for item in second["history"]] == [
        "Привет",
        "Ответ: Привет",
        "Продолжим",
        "Ответ: Продолжим",
    ]


def test_concurrent_turns_keep_both_messages(tmp_path: Path) -> None:
    first_entered = Event()
    release_first = Event()
    second_started = Event()
    second_entered = Event()

    def analyst(state: WorkflowState) -> dict[str, Any]:
        if state["request"] == "Первое":
            first_entered.set()
            assert release_first.wait(timeout=5)
        else:
            second_entered.set()
        return {"analysis": {"intent": "unsupported"}, "reply": state["request"]}

    def unused(_: WorkflowState) -> dict[str, Any]:
        raise AssertionError("Этот агент не должен запускаться")

    def message(text: str) -> WorkflowState:
        return {
            "request": text,
            "conversation_id": "dialog-1",
            "current_datetime": "2026-10-27T10:00:00+03:00",
            "timezone": "Europe/Moscow",
        }

    path = tmp_path / "tasks.db"
    with open_checkpointer(path) as first_saver, open_checkpointer(path) as second_saver:
        first_graph = build_graph(
            Settings(_env_file=None),
            analyst=analyst,
            planner=unused,
            validator=unused,
            checkpointer=first_saver,
        )
        second_graph = build_graph(
            Settings(_env_file=None),
            analyst=analyst,
            planner=unused,
            validator=unused,
            checkpointer=second_saver,
        )

        def run_second() -> dict[str, Any]:
            second_started.set()
            return run_turn(second_graph, message("Второе"))

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(run_turn, first_graph, message("Первое"))
            assert first_entered.wait(timeout=5)
            second = pool.submit(run_second)
            try:
                assert second_started.wait(timeout=5)
                assert not second_entered.wait(timeout=0.5)
            finally:
                release_first.set()
            first.result(timeout=5)
            second.result(timeout=5)

        saved = first_graph.get_state(config_for_conversation("dialog-1")).values

    assert [item["content"] for item in saved["history"]] == [
        "Первое",
        "Первое",
        "Второе",
        "Второе",
    ]


def test_actual_reply_survives_restart(tmp_path: Path) -> None:
    def analyst(_: WorkflowState) -> dict[str, Any]:
        return {"analysis": {"intent": "get_tasks"}}

    def planner(_: WorkflowState) -> dict[str, Any]:
        return {"plan": {"message_to_user": "Черновик плана"}}

    def validator(_: WorkflowState) -> dict[str, Any]:
        return {"verdict": {"decision": "accept"}}

    def make_graph(checkpointer: Any) -> Any:
        return build_graph(
            Settings(_env_file=None),
            analyst=analyst,
            planner=planner,
            validator=validator,
            checkpointer=checkpointer,
        )

    path = tmp_path / "tasks.db"
    state: WorkflowState = {
        "request": "Покажи задачи",
        "conversation_id": "dialog-1",
        "current_datetime": "2026-10-27T10:00:00+03:00",
        "timezone": "Europe/Moscow",
    }

    with open_checkpointer(path) as checkpointer:
        first = run_turn(
            make_graph(checkpointer),
            state,
            build_reply=lambda _: "Задач пока нет",
        )

    with open_checkpointer(path) as checkpointer:
        second = run_turn(
            make_graph(checkpointer),
            {**state, "request": "Покажи мои задачи"},
            build_reply=lambda _: "По-прежнему нет задач",
        )

    assert first["reply"] == "Задач пока нет"
    assert [item["content"] for item in second["history"]] == [
        "Покажи задачи",
        "Задач пока нет",
        "Покажи мои задачи",
        "По-прежнему нет задач",
    ]


def test_run_turn_discards_expired_dialogue(tmp_path: Path) -> None:
    def analyst(state: WorkflowState) -> dict[str, Any]:
        return {"analysis": {"intent": "unsupported"}, "reply": state["request"]}

    def unused(_: WorkflowState) -> dict[str, Any]:
        raise AssertionError("Этот агент не должен запускаться")

    with open_checkpointer(tmp_path / "tasks.db") as checkpointer:
        old = empty_checkpoint()
        old["ts"] = (datetime.now(UTC) - timedelta(days=8)).isoformat()
        checkpointer.put(
            {"configurable": {"thread_id": "dialog-1", "checkpoint_ns": ""}},
            old,
            {"source": "input", "step": 0, "writes": {}},
            {},
        )

        graph = build_graph(
            Settings(_env_file=None),
            analyst=analyst,
            planner=unused,
            validator=unused,
            checkpointer=checkpointer,
        )
        result = run_turn(
            graph,
            {
                "request": "Новый запрос",
                "conversation_id": "dialog-1",
                "current_datetime": "2026-10-27T10:00:00+03:00",
                "timezone": "Europe/Moscow",
            },
        )
        saved_ids = {
            item.checkpoint["id"] for item in checkpointer.list(config_for_conversation("dialog-1"))
        }

    assert old["id"] not in saved_ids
    assert [item["content"] for item in result["history"]] == [
        "Новый запрос",
        "Новый запрос",
    ]
