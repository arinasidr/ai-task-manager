"""P12: поиск возможных предпочтений, ответ «да/нет», применение к плану, API памяти."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from ai_task_manager.agents.contracts import Analysis
from ai_task_manager.agents.planner import Planner
from ai_task_manager.api.app import create_app
from ai_task_manager.config import get_settings
from ai_task_manager.db.crud_func import create_task
from ai_task_manager.db.database import connect, init_db
from ai_task_manager.db.preferences import get_preferences, propose_preference
from ai_task_manager.memory.inference import (
    WEEKDAYS,
    find_candidate,
    from_analysis,
    from_behavior,
    from_text,
    normalize_value,
    same_value,
)
from ai_task_manager.workflow.preferences import classify_answer

MSK = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=MSK)  # среда
SATURDAY = datetime(2026, 10, 17, 12, 0, tzinfo=MSK)


# ---------------------------------------------------------------- поиск догадок


@pytest.mark.parametrize(
    ("request_text", "kind", "value"),
    [
        (
            "Добавь задачу созвон, мне вообще удобнее работать вечером",
            "preferred_work_time",
            "вечером",
        ),
        ("С чего начать? Я обычно работаю по утрам", "preferred_work_time", "утром"),
        ("удобнее всего днём", "preferred_work_time", "днём"),
        ("кстати, я не работаю по выходным", "working_days", WEEKDAYS[:5]),
        (
            "я работаю по понедельникам, средам и пятницам",
            "working_days",
            ["понедельник", "среда", "пятница"],
        ),
        ("Работаю без выходных, добавь задачу", "working_days", WEEKDAYS),
    ],
)
def test_preference_found_in_text(request_text: str, kind: str, value: Any) -> None:
    [candidate] = from_text(request_text)
    assert (candidate.kind, candidate.value, candidate.source) == (kind, value, "text")
    assert "вы написали" in candidate.evidence


@pytest.mark.parametrize(
    "request_text",
    [
        "Добавь задачу пробежка утром",  # время задачи, а не привычка
        "Добавь задачу: работаю над отчётом утром",
        "Перенеси созвон на вечер",
        "Покажи задачи",
    ],
)
def test_task_time_is_not_a_preference(request_text: str) -> None:
    assert from_text(request_text) == []


def test_values_are_normalized() -> None:
    assert normalize_value("preferred_work_time", "evening") == "вечером"
    assert normalize_value("preferred_work_time", "по вечерам") == "вечером"
    assert normalize_value("working_days", ["будни"]) == WEEKDAYS[:5]
    assert same_value("preferred_work_time", "вечер", "evening")
    assert not same_value("preferred_work_time", "утром", "вечером")


def test_analyst_hint_is_kept_apart_from_updates() -> None:
    analysis = Analysis.model_validate(
        {
            "intent": "create_task",
            "tasks": [{"title": "Пробежка"}],
            "preference": {"kind": "preferred_work_time", "value": "утро"},
        }
    )
    assert analysis.updates == {}
    assert analysis.preference_hint == {"kind": "preferred_work_time", "value": "утро"}
    [candidate] = from_analysis(analysis.model_dump())
    assert (candidate.value, candidate.source) == ("утром", "analyst")
    # при явном «запомни» это не догадка, а сохранение (update_preference)
    explicit = Analysis.model_validate(
        {
            "intent": "update_preference",
            "preference": {"kind": "preferred_work_time", "value": "утро"},
        }
    )
    assert from_analysis(explicit.model_dump()) == []


def done_tasks(hours: list[int]) -> list[dict[str, Any]]:
    day = datetime(2026, 10, 1, tzinfo=MSK)
    return [
        {
            "status": "done",
            "updated_at": (day + timedelta(days=i, hours=hour)).astimezone(UTC).isoformat(),
        }
        for i, hour in enumerate(hours)
    ]


def test_habit_found_from_completed_tasks() -> None:
    [candidate] = from_behavior(done_tasks([19, 20, 21, 22, 20]), "Europe/Moscow")
    assert (candidate.value, candidate.source) == ("вечером", "behavior")
    assert "5 из 5" in candidate.evidence


@pytest.mark.parametrize("hours", [[19, 20, 21, 22], [19, 20, 9, 10, 21]])
def test_no_habit_without_enough_evidence(hours: list[int]) -> None:
    assert from_behavior(done_tasks(hours), "Europe/Moscow") == []


def test_candidate_skips_already_confirmed_value() -> None:
    text = "мне удобнее работать вечером"
    assert (
        find_candidate(text, None, [], {"preferred_work_time": "evening"}, timezone="UTC") is None
    )
    changed = find_candidate(text, None, [], {"preferred_work_time": "утром"}, timezone="UTC")
    assert changed is not None and changed.value == "вечером"  # смена значения — снова вопрос
    habit = done_tasks([20, 20, 20, 20, 20])
    # привычка не спорит с тем, что пользователь подтвердил сам
    assert find_candidate("", None, habit, {"preferred_work_time": "утром"}, timezone="UTC") is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Да", True),
        ("да, запомни", True),
        ("ок", True),
        ("Нет", False),
        ("нет, не надо", False),
        ("не запоминай", False),
        ("да и добавь задачу купить молоко", None),
        ("Покажи задачи", None),
        ("данные собрать", None),
    ],
)
def test_answer_to_preference_question(message: str, expected: bool | None) -> None:
    assert classify_answer(message) is expected


# ---------------------------------------------------------------- применение к плану


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    path = tmp_path / "tasks.db"
    init_db(path)
    with closing(connect(path)) as connection:
        yield connection


class FakeLLM:
    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer = answer
        self.users: list[dict[str, Any]] = []

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.users.append(json.loads(user))
        return self.answer


def add(db: sqlite3.Connection, title: str, **fields: Any) -> int:
    task_id = create_task(db, title, **fields)
    db.commit()
    return task_id


def test_preferred_type_breaks_ties_only(db: sqlite3.Connection) -> None:
    report = add(db, "Написать отчёт")
    call = add(db, "Созвон с командой")
    urgent = add(db, "Сдать налоги", due_date="2026-10-15")

    plan = Planner(None).plan(
        {"intent": "plan_tasks"},
        db,
        now=NOW,
        preferences={"preferred_task_types": ["созвоны"]},
    )

    # срочная задача остаётся первой; среди задач без срока созвон раньше отчёта
    assert plan.ordered_task_ids == [urgent, call, report]
    assert f"#{call}" in plan.message_to_user
    assert "Учтены ваши предпочтения" in plan.message_to_user


def test_working_days_facts_reach_model(db: sqlite3.Connection) -> None:
    task = add(db, "Отчёт", due_date="2026-10-19")  # понедельник
    llm = FakeLLM({"ordered_task_ids": [task], "recommendations": [], "message": "Начните."})

    plan = Planner(llm).plan(
        {"intent": "plan_tasks"},
        db,
        now=SATURDAY,
        preferences={"working_days": WEEKDAYS[:5]},
    )

    context = llm.users[0]
    assert context["preferences"]["today_is_working_day"] is False
    # суббота → понедельник: рабочий день до срока один
    assert context["tasks"][0]["working_days_left"] == 1
    assert "non_working_day" in plan.warnings
    assert "нерабочий день" in plan.message_to_user


def test_plan_without_preferences_is_unchanged(db: sqlite3.Connection) -> None:
    add(db, "Отчёт")
    plan = Planner(None).plan({"intent": "plan_tasks"}, db, now=NOW, preferences={})
    assert "Учтены" not in plan.message_to_user
    assert "non_working_day" not in plan.warnings


# ---------------------------------------------------------------- HTTP API


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("ATM_DATABASE_PATH", str(tmp_path / "tasks.db"))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        get_settings.cache_clear()


def proposal(path: str, kind: str, value: Any) -> int:
    with closing(connect(path)) as connection:
        created = propose_preference(connection, kind, value)
        connection.commit()
        return created.id


def test_api_lists_and_confirms_proposal(client: TestClient) -> None:
    path = get_settings().database_path
    proposal_id = proposal(path, "preferred_work_time", "вечером")

    listed = client.get("/api/v1/preferences").json()
    assert listed["confirmed"] == []
    assert listed["proposed"][0]["text"] == "удобнее работать вечером"

    response = client.post(f"/api/v1/preferences/{proposal_id}/confirm")
    assert response.status_code == 200
    assert response.json()["status"] == "confirmed"
    with closing(connect(path)) as connection:
        assert get_preferences(connection) == {"preferred_work_time": "вечером"}

    # повторное подтверждение — конфликт, а не второе сохранение
    assert client.post(f"/api/v1/preferences/{proposal_id}/confirm").status_code == 409


def test_api_reject_change_and_delete(client: TestClient) -> None:
    path = get_settings().database_path
    proposal_id = proposal(path, "working_days", WEEKDAYS[:5])

    assert client.post(f"/api/v1/preferences/{proposal_id}/reject").json()["status"] == "rejected"
    assert client.get("/api/v1/preferences").json()["proposed"] == []

    changed = client.put("/api/v1/preferences/preferred_work_time", json={"value": "evening"})
    assert changed.json()["value"] == "вечером"
    assert changed.json()["source"] == "explicit"

    assert client.delete("/api/v1/preferences/preferred_work_time").json()["deleted"] is True
    assert client.get("/api/v1/preferences").json()["confirmed"] == []
    assert client.delete("/api/v1/preferences/preferred_work_time").status_code == 404
    assert client.put("/api/v1/preferences/unknown", json={"value": "x"}).status_code == 422
