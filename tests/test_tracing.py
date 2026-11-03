"""P14: трейсы Tools, JSON-логи, траектория агентов (без Phoenix: спаны в памяти)."""

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai_task_manager.agents.contracts import Plan
from ai_task_manager.config import Settings
from ai_task_manager.db.crud_func import create_task
from ai_task_manager.db.database import connect, init_db
from ai_task_manager.observability.setup import setup_logging, setup_tracing
from ai_task_manager.observability.trajectory import trajectory_node, write_trajectory
from ai_task_manager.tools.executor import ConfirmationRequired, execute_plan, run_plan

EXPORTER = InMemorySpanExporter()
if not isinstance(trace.get_tracer_provider(), TracerProvider):
    _provider = TracerProvider()
    _provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    trace.set_tracer_provider(_provider)
else:  # провайдер уже задан другим тестом — подключаемся к нему
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(EXPORTER))


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    init_db(tmp_path / "tasks.db")
    with closing(connect(tmp_path / "tasks.db")) as connection:
        yield connection


@pytest.fixture
def log_file(tmp_path: Path) -> Iterator[Path]:
    path = setup_logging(Settings(_env_file=None, log_dir=str(tmp_path / "logs")))
    yield path
    for handler in list(logging.getLogger("ai_task_manager").handlers):
        if getattr(handler, "baseFilename", None) == str(path):
            logging.getLogger("ai_task_manager").removeHandler(handler)
            logging.getLogger("uvicorn.error").removeHandler(handler)
            handler.close()


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def spans(name_prefix: str) -> list:
    return [s for s in EXPORTER.get_finished_spans() if s.name.startswith(name_prefix)]


def test_tool_call_is_span_and_json_log(db: sqlite3.Connection, log_file: Path) -> None:
    EXPORTER.clear()
    plan = Plan(
        intent="create_task",
        operations=[{"tool": "create_task", "ref": "new1", "args": {"title": "Купить молоко"}}],
    )

    run_plan(db, plan)

    [span] = spans("tool.")
    assert span.name == "tool.create_task"
    assert span.attributes["openinference.span.kind"] == "TOOL"
    assert json.loads(span.attributes["input.value"]) == {"title": "Купить молоко"}
    assert json.loads(span.attributes["output.value"])["task_id"] == 1
    [line] = [r for r in records(log_file) if r["message"] == "tool_call"]
    # уровень INFO включён — как в работе; поле аргументов не конфликтует с LogRecord.args
    assert line["tool"] == "create_task"
    assert line["tool_args"] == {"title": "Купить молоко"}
    assert line["status"] == "ok"
    assert line["trace_id"] == format(span.context.trace_id, "032x")


def test_failed_tool_is_error_span_and_rollback_event(
    db: sqlite3.Connection, log_file: Path
) -> None:
    EXPORTER.clear()
    plan = Plan(
        intent="update_task",
        operations=[
            {"tool": "update_task", "args": {"task_id": 999, "updates": {"priority": "low"}}}
        ],
    )

    with pytest.raises(ValueError):
        run_plan(db, plan)

    [span] = spans("tool.update_task")
    assert span.status.status_code.name == "ERROR"
    messages = [(r["message"], r.get("status")) for r in records(log_file)]
    assert ("tool_call", "error") in messages
    assert ("tool_plan_rolled_back", None) in messages


def test_blocked_dangerous_plan_is_logged(db: sqlite3.Connection, log_file: Path) -> None:
    task = create_task(db, "Отчёт")
    db.commit()
    plan = Plan(
        intent="delete_task", operations=[{"tool": "delete_task", "args": {"task_id": task}}]
    )

    with pytest.raises(ConfirmationRequired):
        execute_plan(db, plan)

    [line] = [r for r in records(log_file) if r["message"] == "tool_blocked_without_confirmation"]
    assert line["event_tools"] == ["delete_task"]
    assert "необратимо" in line["event_reasons"][0]


def test_trajectory_step_records_node_summary() -> None:
    def planner(_: dict) -> dict:
        return {"plan": {"source": "llm", "operations": [{"tool": "list_tasks"}]}}

    update = trajectory_node("planner", planner)({"trajectory": [{"node": "analyst", "ms": 1}]})

    assert [step["node"] for step in update["trajectory"]] == ["analyst", "planner"]
    step = update["trajectory"][1]
    assert step["source"] == "llm"
    assert step["tools"] == ["list_tasks"]
    assert step["ms"] >= 0


def test_trajectory_file_only_when_log_dir_set(tmp_path: Path) -> None:
    assert write_trajectory(Settings(_env_file=None), {"steps": []}) is None

    path = write_trajectory(
        Settings(_env_file=None, log_dir=str(tmp_path)), {"request": "Покажи задачи", "steps": []}
    )
    [line] = records(path)
    assert line["request"] == "Покажи задачи"
    assert "ts" in line


def test_tracing_is_off_without_endpoint() -> None:
    assert setup_tracing(Settings(_env_file=None)) is False
