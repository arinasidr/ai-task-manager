"""Агент-валидатор: временная SQLite и поддельная LLM, без сети и Ollama."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from ai_task_manager.agents.analyst.schema import AnalystResult
from ai_task_manager.agents.contracts import Plan
from ai_task_manager.agents.llm import LLMError
from ai_task_manager.agents.planner import Planner
from ai_task_manager.agents.validator import (
    VALIDATOR_OUTPUT_SCHEMA,
    Validator,
    check_validator_family,
    load_system_prompt,
    make_validator_node,
    mentioned_dates,
    model_family,
    plan_view,
)
from ai_task_manager.config import Settings
from ai_task_manager.db.crud_func import add_dependency, create_task
from ai_task_manager.db.database import connect, init_db
from ai_task_manager.workflow.graph import build_graph

MSK = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=MSK)
ACCEPT = {"decision": "accept", "issues": []}


class FakeLLM:
    def __init__(self, *answers: dict[str, Any] | Exception) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"system": system, "user": json.loads(user), "schema": schema})
        answer = self.answers.pop(0)
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


def add(db: sqlite3.Connection, title: str, **fields: Any) -> int:
    task_id = create_task(db, title, **fields)
    db.commit()
    return task_id


def link(db: sqlite3.Connection, predecessor: int, successor: int) -> None:
    add_dependency(db, predecessor, successor)
    db.commit()


def planned(db: sqlite3.Connection, analysis: dict[str, Any]) -> dict[str, Any]:
    """План от настоящего планировщика (без LLM)."""
    return Planner().plan(analysis, db, now=NOW).model_dump()


def check(
    db: sqlite3.Connection,
    analysis: dict[str, Any],
    plan: dict[str, Any],
    llm: FakeLLM | None = None,
    request: str = "запрос",
):
    return Validator(llm, model="ministral-3:3b").validate(request, analysis, plan, db, now=NOW)


# ---------------------------------------------------------------- семейства моделей


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("gemma4:e2b", "Google Gemma"),
        ("qwen3:1.7b", "Alibaba Qwen"),
        ("ministral-3:3b", "Mistral"),
        ("phi4-mini:3.8b-q4_K_M", "Microsoft Phi"),
        ("LiquidAI/lfm2.5-1.2b-instruct:q4_k_m", "Liquid LFM"),
    ],
)
def test_model_family(model: str, family: str) -> None:
    assert model_family(model) == family


def test_validator_must_be_other_family() -> None:
    check_validator_family("ministral-3:3b", "gemma4:e2b", "qwen3:1.7b")
    with pytest.raises(ValueError, match="другого семейства"):
        check_validator_family("gemma3:4b", "gemma4:e2b")


def test_default_settings_use_other_family_for_validator() -> None:
    s = Settings(_env_file=None)
    check_validator_family(s.validator_model, s.analyst_model, s.planner_model, s.fast_model)


# ---------------------------------------------------------------- слой 1: код


def test_correct_plan_from_planner_passes_code_and_goes_to_llm(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    analysis = {"intent": "delete_task", "task_id": task}
    llm = FakeLLM(ACCEPT)
    verdict = check(db, analysis, planned(db, analysis), llm)
    assert verdict.accepted
    assert verdict.source == "llm"
    assert len(llm.calls) == 1


def test_delete_without_confirmation_is_rejected_by_code(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    analysis = {"intent": "delete_task", "task_id": task}
    plan = planned(db, analysis)
    plan["requires_confirmation"] = False  # например, prompt injection «без подтверждения»
    llm = FakeLLM()
    verdict = check(db, analysis, plan, llm)
    assert not verdict.accepted
    assert verdict.issues[0].code == "missing_confirmation"
    assert llm.calls == []  # модель не вызывается, если код уже нашёл нарушение


def test_plan_touching_task_not_named_by_user_is_rejected(db: sqlite3.Connection) -> None:
    named, other = add(db, "Отчёт"), add(db, "Диплом")
    analysis = {"intent": "delete_task", "task_id": named}
    plan = planned(db, analysis)
    plan["operations"].append({"tool": "delete_task", "args": {"task_id": other}})
    verdict = check(db, analysis, plan)
    assert any(issue.code == "wrong_task" for issue in verdict.issues)


def test_unknown_task_id_is_rejected(db: sqlite3.Connection) -> None:
    plan = {
        "intent": "update_task",
        "operations": [{"tool": "update_task", "args": {"task_id": 77, "updates": {}}}],
    }
    verdict = check(db, {"intent": "update_task", "task_id": 77, "updates": {"title": "x"}}, plan)
    assert verdict.issues[0].code == "unknown_task"


def test_operation_outside_intent_is_rejected(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    plan = {
        "intent": "get_tasks",
        "operations": [{"tool": "delete_task", "args": {"task_id": task}}],
        "requires_confirmation": True,
    }
    verdict = check(db, {"intent": "get_tasks"}, plan)
    assert any(issue.code == "intent_mismatch" for issue in verdict.issues)


def test_intent_mismatch_with_analysis(db: sqlite3.Connection) -> None:
    plan = planned(db, {"intent": "create_task", "tasks": [{"title": "A"}]})
    verdict = check(db, {"intent": "get_tasks"}, plan)
    assert any(issue.code == "intent_mismatch" for issue in verdict.issues)


def test_extra_created_task_is_rejected(db: sqlite3.Connection) -> None:
    analysis = {"intent": "create_task", "tasks": [{"title": "A"}]}
    plan = planned(db, analysis)
    plan["operations"].append({"tool": "create_task", "ref": "new2", "args": {"title": "B"}})
    verdict = check(db, analysis, plan)
    assert verdict.issues[0].code == "invented_data"


def test_dangling_ref_is_rejected(db: sqlite3.Connection) -> None:
    analysis = {"intent": "create_task", "tasks": [{"title": "A"}]}
    plan = planned(db, analysis)
    plan["operations"].append(
        {
            "tool": "add_dependency",
            "args": {"predecessor_task_id": "$new1", "successor_task_id": "$new9"},
        }
    )
    verdict = check(db, analysis, plan)
    assert any(issue.code == "unknown_task" for issue in verdict.issues)


def test_cycle_is_rejected(db: sqlite3.Connection) -> None:
    a, b = add(db, "A"), add(db, "B")
    link(db, a, b)
    plan = {
        "intent": "update_task",
        "operations": [
            {"tool": "add_dependency", "args": {"predecessor_task_id": b, "successor_task_id": a}}
        ],
    }
    analysis = {"intent": "update_task", "task_id": a, "updates": {"add_dependency_ids": [b]}}
    verdict = check(db, analysis, plan)
    assert any(issue.code == "dependency_cycle" for issue in verdict.issues)


def test_completion_with_unfinished_predecessor_needs_confirmation(
    db: sqlite3.Connection,
) -> None:
    a, b = add(db, "A"), add(db, "B")
    link(db, a, b)
    analysis = {"intent": "update_task", "task_id": b, "updates": {"status": "done"}}
    plan = planned(db, analysis)
    assert check(db, analysis, plan, FakeLLM(ACCEPT)).accepted
    plan["requires_confirmation"] = False
    assert check(db, analysis, plan).issues[0].code == "missing_confirmation"


def test_bad_order_is_rejected(db: sqlite3.Connection) -> None:
    a, b = add(db, "A"), add(db, "B")
    link(db, a, b)
    plan = {"intent": "plan_tasks", "ordered_task_ids": [b, a]}
    verdict = check(db, {"intent": "plan_tasks"}, plan)
    assert verdict.issues[0].code == "bad_order"


def test_order_must_contain_all_open_tasks(db: sqlite3.Connection) -> None:
    a, _ = add(db, "A"), add(db, "B")
    verdict = check(db, {"intent": "plan_tasks"}, {"intent": "plan_tasks", "ordered_task_ids": [a]})
    assert verdict.issues[0].code == "bad_order"


def test_deadline_changed_on_the_way_to_plan_is_caught_by_code(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    analysis = {"intent": "update_task", "task_id": task, "updates": {"deadline": "2026-10-15"}}
    plan = planned(db, analysis)
    plan["operations"][0]["args"]["updates"]["due_date"] = "2026-10-16"
    llm = FakeLLM()
    verdict = check(db, analysis, plan, llm)
    assert verdict.issues[0].code == "wrong_deadline"
    assert llm.calls == []


def test_new_deadline_missing_from_plan_is_caught(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    analysis = {"intent": "update_task", "task_id": task, "updates": {"deadline": "2026-10-15"}}
    plan = {
        "intent": "update_task",
        "operations": [
            {"tool": "update_task", "args": {"task_id": task, "updates": {"title": "Отчёт"}}}
        ],
    }
    assert check(db, analysis, plan).issues[0].code == "missing_data"


def test_create_deadline_must_match_analysis(db: sqlite3.Connection) -> None:
    analysis = {"intent": "create_task", "tasks": [{"title": "A", "deadline": "2026-10-20"}]}
    plan = planned(db, analysis)
    assert check(db, analysis, plan, FakeLLM(ACCEPT)).accepted
    plan["operations"][0]["args"]["due_time"] = "09:00:00"  # выдуманное время
    assert check(db, analysis, plan).issues[0].code == "wrong_deadline"


def test_plan_view_hides_technical_deadline_fields(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт", due_date="2026-10-20", due_time="18:00")
    date_only = Planner().plan(
        {"intent": "update_task", "task_id": task, "updates": {"deadline": "2026-10-15"}},
        db,
        now=NOW,
    )
    args = plan_view(date_only)["operations"][0]["args"]["updates"]
    # у модели нет due_time: null и прочих полей, только понятный срок
    assert args == {"deadline": "2026-10-15"}

    moment = Planner().plan(
        {
            "intent": "create_task",
            "tasks": [{"title": "Созвон", "deadline": "2026-10-20T18:30:00+03:00"}],
        },
        db,
        now=NOW,
    )
    assert plan_view(moment)["operations"][0]["args"] == {
        "title": "Созвон",
        "deadline": "2026-10-20 18:30 (Europe/Moscow)",
    }
    assert "ordered_task_ids" not in plan_view(Plan(intent="get_tasks"))


def test_missing_plan_is_rejected(db: sqlite3.Connection) -> None:
    assert not check(db, {"intent": "get_tasks"}, None).accepted


# ---------------------------------------------------------------- когда модель не нужна


def test_read_only_and_clarification_plans_skip_llm(db: sqlite3.Connection) -> None:
    llm = FakeLLM()
    read = planned(db, {"intent": "get_tasks"})
    clarify = planned(db, {"intent": "clarification_required", "missing_fields": ["task_id"]})
    assert check(db, {"intent": "get_tasks"}, read, llm).accepted
    assert check(db, {"intent": "clarification_required"}, clarify, llm).accepted
    assert llm.calls == []


def test_code_rejection_by_planner_is_a_valid_clarification(db: sqlite3.Connection) -> None:
    # планировщик сам отказал (задачи нет) и спрашивает — проверять моделью нечего
    analysis = {"intent": "update_task", "task_id": 999, "updates": {"status": "done"}}
    verdict = check(db, analysis, planned(db, analysis), FakeLLM())
    assert verdict.accepted


# ---------------------------------------------------------------- слой 2: LLM


def test_llm_reject_returns_reasons_for_planner(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    analysis = {"intent": "update_task", "task_id": task, "updates": {"deadline": "2026-10-17"}}
    llm = FakeLLM(
        {
            "decision": "reject",
            "issues": [{"code": "wrong_deadline", "detail": "Пятница — это 16.10, а не 17.10"}],
        }
    )
    verdict = check(db, analysis, planned(db, analysis), llm, request="Перенеси #1 на пятницу")
    assert not verdict.accepted
    assert verdict.reasons == ["Пятница — это 16.10, а не 17.10"]

    sent = llm.calls[0]
    assert sent["schema"] == VALIDATOR_OUTPUT_SCHEMA
    assert sent["user"]["request"] == "Перенеси #1 на пятницу"
    assert "verified_by_code" not in sent["user"]  # дни недели код не распознаёт
    assert sent["user"]["now"].startswith("2026-10-14")
    # календарь посчитан кодом: 14.10.2026 — среда
    assert sent["user"]["calendar"][:2] == [
        "сегодня: 2026-10-14, среда",
        "завтра: 2026-10-15, четверг",
    ]
    assert sent["user"]["tasks"][0]["id"] == task
    assert sent["user"]["plan"]["operations"][0]["tool"] == "update_task"


def test_plan_tasks_context_has_task_facts(db: sqlite3.Connection) -> None:
    a = add(db, "A", due_date="2026-10-15")
    b = add(db, "B")
    link(db, a, b)
    llm = FakeLLM(ACCEPT)
    plan = Planner().plan({"intent": "plan_tasks"}, db, now=NOW).model_dump()
    check(db, {"intent": "plan_tasks"}, plan, llm)
    facts = {t["id"]: t for t in llm.calls[0]["user"]["tasks"]}
    assert facts[a]["days_left"] == 1
    assert facts[b]["blocked_by"] == [a]


def test_reject_without_issues_is_retried_then_fail_closed(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    analysis = {"intent": "delete_task", "task_id": task}
    llm = FakeLLM({"decision": "reject", "issues": []}, LLMError("timeout"))
    verdict = check(db, analysis, planned(db, analysis), llm)
    assert len(llm.calls) == 2
    assert not verdict.accepted
    assert verdict.source == "unavailable"


def test_accept_with_issues_is_accept(db: sqlite3.Connection) -> None:
    analysis = {"intent": "create_task", "tasks": [{"title": "A"}]}
    llm = FakeLLM({"decision": "accept", "issues": [{"code": "other", "detail": "мелочь"}]})
    verdict = check(db, analysis, planned(db, analysis), llm)
    assert verdict.accepted
    assert verdict.issues == []
    assert verdict.notes


def test_llm_unavailable_accepts_read_only_order(db: sqlite3.Connection) -> None:
    add(db, "A")
    plan = Planner().plan({"intent": "plan_tasks"}, db, now=NOW).model_dump()
    verdict = check(db, {"intent": "plan_tasks"}, plan, FakeLLM(LLMError("x"), LLMError("y")))
    assert verdict.accepted
    assert verdict.source == "unavailable"


def test_no_llm_rejects_write_plans(db: sqlite3.Connection) -> None:
    analysis = {"intent": "create_task", "tasks": [{"title": "A"}]}
    verdict = check(db, analysis, planned(db, analysis), None)
    assert not verdict.accepted
    assert verdict.source == "unavailable"


def test_validator_accepts_analyst_result(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    result = AnalystResult.model_validate(
        {"intent": "update_task", "task_id": task, "updates": {"priority": "high"}}
    )
    plan = Planner().plan(result, db, now=NOW).model_dump()
    verdict = check(db, result.model_dump(), plan, FakeLLM(ACCEPT))
    assert verdict.accepted


def test_validator_never_writes(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    before = db.execute("SELECT * FROM tasks").fetchall()
    analysis = {"intent": "delete_task", "task_id": task}
    check(db, analysis, planned(db, analysis), FakeLLM(ACCEPT))
    assert [tuple(r) for r in db.execute("SELECT * FROM tasks")] == [tuple(r) for r in before]


def test_system_prompt_has_soul_and_role() -> None:
    prompt = load_system_prompt()
    # даты модель берёт из готового календаря, а не считает сама
    assert "`calendar`" in prompt
    assert prompt.startswith("# SOUL - AI Task Manager")
    assert "агент-валидатор" in prompt
    assert "## Примеры ответов пользователю" not in prompt
    assert "```json" not in prompt


# ---------------------------------------------------------------- узел графа


def test_validator_node_in_graph(db_path: Path, db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    llm = FakeLLM(
        {"decision": "reject", "issues": [{"code": "wrong_task", "detail": "Не та задача"}]}
    )
    models: list[str] = []

    def llm_for(model: str) -> FakeLLM:
        models.append(model)
        return llm

    def analyst(_: dict) -> dict:
        return {"analysis": {"intent": "delete_task", "task_id": task}}

    def planner(state: dict) -> dict:
        with closing(connect(db_path)) as connection:
            plan = Planner().plan(state["analysis"], connection, now=NOW)
        return {"plan": plan.model_dump()}

    graph = build_graph(
        Settings(_env_file=None),
        analyst=analyst,
        planner=planner,
        validator=make_validator_node(llm_for=llm_for, connect=lambda: connect(db_path)),
    )
    result = graph.invoke(
        {
            "request": f"Удали задачу #{task}",
            "conversation_id": None,
            "current_datetime": NOW.isoformat(),
            "timezone": "Europe/Moscow",
        }
    )
    assert models == ["ministral-3:3b"]
    assert result["verdict"]["decision"] == "reject"
    assert result["validator_feedback"] == ["Не та задача"]
    assert result["validation_attempts"] == 1


def test_validator_node_refuses_same_family(db_path: Path) -> None:
    node = make_validator_node(llm_for=lambda _: None, connect=lambda: connect(db_path))
    with pytest.raises(ValueError, match="другого семейства"):
        node(
            {
                "request": "x",
                "current_datetime": NOW.isoformat(),
                "timezone": "Europe/Moscow",
                "analyst_model": "gemma4:e2b",
                "planner_model": "gemma4:e2b",
                "validator_model": "gemma3:4b",
            }
        )


# ---------------------------------------------------------------- относительные даты


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("Перенеси задачу #1 на завтра", {"завтра": "2026-10-15"}),
        ("на послезавтра", {"послезавтра": "2026-10-16"}),
        ("сегодня до вечера", {"сегодня": "2026-10-14"}),
        ("через 3 дня", {"через 3 дня": "2026-10-17"}),
        ("через три дня в 10:15", {"через три дня": "2026-10-17"}),
        ("в пятницу", {}),
    ],
)
def test_mentioned_dates(request_text: str, expected: dict[str, str]) -> None:
    found = {word: day.isoformat() for word, day in mentioned_dates(request_text, NOW).items()}
    assert found == expected


def test_analyst_misread_tomorrow_is_caught_by_code(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    # аналитик понял «завтра» как 16.10, план последовательно повторяет его ошибку
    analysis = {"intent": "update_task", "task_id": task, "updates": {"deadline": "2026-10-16"}}
    llm = FakeLLM()
    verdict = check(db, analysis, planned(db, analysis), llm, request="Перенеси #1 на завтра")
    assert verdict.issues[0].code == "wrong_deadline"
    assert "2026-10-15" in verdict.issues[0].detail
    assert llm.calls == []


def test_llm_date_complaint_is_overruled_when_code_verified(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт")
    analysis = {"intent": "update_task", "task_id": task, "updates": {"deadline": "2026-10-15"}}
    llm = FakeLLM(
        {
            "decision": "reject",
            "issues": [{"code": "wrong_deadline", "detail": "должно быть 2026-10-15"}],
        }
    )
    verdict = check(db, analysis, planned(db, analysis), llm, request="Перенеси #1 на завтра")
    assert verdict.accepted
    assert any("проверены кодом" in note for note in verdict.notes)
    assert llm.calls[0]["user"]["verified_by_code"]


def test_other_llm_issues_still_reject_when_dates_verified(db: sqlite3.Connection) -> None:
    analysis = {"intent": "create_task", "tasks": [{"title": "Молоко", "deadline": "2026-10-15"}]}
    llm = FakeLLM(
        {
            "decision": "reject",
            "issues": [
                {"code": "wrong_deadline", "detail": "дата"},
                {"code": "invented_data", "detail": "приоритет выдуман"},
            ],
        }
    )
    verdict = check(db, analysis, planned(db, analysis), llm, request="Купить молоко завтра")
    assert not verdict.accepted
    assert verdict.reasons == ["приоритет выдуман"]
