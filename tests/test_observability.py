import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import ai_task_manager.observability.metrics as metrics_module
from ai_task_manager.api import chat as chat_api
from ai_task_manager.api.app import create_app
from ai_task_manager.db.database import connect, init_db
from ai_task_manager.observability.metrics import MetricsRegistry
from ai_task_manager.workflow.service import ChatResult


def test_three_agent_errors_trigger_one_alert(caplog) -> None:
    metrics = MetricsRegistry()

    with caplog.at_level(logging.WARNING):
        metrics.record_agent_error("decision")
        metrics.record_agent_error("decision")
        assert metrics.snapshot()["alerts"] == []

        metrics.record_agent_error("decision")
        metrics.record_agent_error("decision")

    snapshot = metrics.snapshot()
    assert snapshot["agent_errors"]["decision"] == 4
    assert len(snapshot["alerts"]) == 1
    assert snapshot["alerts"][0]["code"] == "agent_error_burst"
    assert "Всплеск ошибок агента decision" in caplog.text


def test_chat_duration_is_visible_in_metrics(caplog) -> None:
    client = TestClient(create_app())

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        assert client.post("/api/v1/chat", json={}).status_code == 422

    snapshot = client.get("/api/v1/metrics").json()
    assert snapshot["requests_total"] == 1
    assert snapshot["request_duration_seconds_sum"] > 0
    chat_logs = [
        record.getMessage()
        for record in caplog.records
        if record.name == "uvicorn.error" and record.message.startswith("chat_request ")
    ]
    assert len(chat_logs) == 1
    assert "status=422" in chat_logs[0]
    assert "duration_seconds=" in chat_logs[0]


def test_chat_records_agent_errors_and_retries(monkeypatch) -> None:
    app = create_app()
    app.dependency_overrides[chat_api.get_workflow] = lambda: object()
    app.dependency_overrides[chat_api.get_db] = lambda: None

    def fake_run_chat(*args, **kwargs):
        return ChatResult(
            conversation_id="test-dialog",
            reply="Попробуйте позже",
            error="analyst: неверный ответ модели",
            decision_error="Tev1 недоступна",
            plan={"source": "fallback"},
            verdict={"source": "unavailable"},
            validation_attempts=3,
            agent_error_counts={"analyst": 1, "decision": 1, "planner": 1, "validator": 1},
        )

    monkeypatch.setattr(chat_api, "run_chat", fake_run_chat)
    client = TestClient(app)

    response = client.post("/api/v1/chat", json={"message": "Покажи задачи"})
    assert response.status_code == 200

    snapshot = client.get("/api/v1/metrics").json()
    assert snapshot["agent_errors"] == {
        "analyst": 1,
        "decision": 1,
        "planner": 1,
        "validator": 1,
    }
    assert snapshot["retries"] == {"validator": 2}


def test_recovered_validator_error_is_visible_in_api_metrics(monkeypatch) -> None:
    app = create_app()
    app.dependency_overrides[chat_api.get_workflow] = lambda: object()
    app.dependency_overrides[chat_api.get_db] = lambda: None
    monkeypatch.setattr(
        chat_api,
        "run_chat",
        lambda *args, **kwargs: ChatResult(
            conversation_id="test-dialog",
            reply="План принят после повторной проверки",
            verdict={"decision": "accept", "source": "llm"},
            validation_attempts=2,
            agent_error_counts={"validator": 1},
        ),
    )
    client = TestClient(app)

    response = client.post("/api/v1/chat", json={"message": "С чего начать?"})

    assert response.status_code == 200
    snapshot = client.get("/api/v1/metrics").json()
    assert snapshot["agent_errors"] == {"validator": 1}
    assert snapshot["retries"] == {"validator": 1}
    assert snapshot["alerts"] == []


def test_model_retries_are_separate_from_graph_retries(monkeypatch) -> None:
    app = create_app()
    app.dependency_overrides[chat_api.get_workflow] = lambda: object()
    app.dependency_overrides[chat_api.get_db] = lambda: None
    monkeypatch.setattr(
        chat_api,
        "run_chat",
        lambda *args, **kwargs: ChatResult(
            conversation_id="test-dialog",
            reply="План готов",
            validation_attempts=2,
            model_retry_counts={"planner": 2, "validator": 1},
        ),
    )
    client = TestClient(app)

    assert client.post("/api/v1/chat", json={"message": "С чего начать?"}).status_code == 200

    snapshot = client.get("/api/v1/metrics").json()
    assert snapshot["retries"] == {"planner_llm": 2, "validator_llm": 1, "validator": 1}
    assert snapshot["agent_errors"] == {}


def test_unknown_user_task_id_does_not_raise_agent_alert(monkeypatch) -> None:
    app = create_app()
    app.dependency_overrides[chat_api.get_workflow] = lambda: object()
    app.dependency_overrides[chat_api.get_db] = lambda: None
    monkeypatch.setattr(
        chat_api,
        "run_chat",
        lambda *args, **kwargs: ChatResult(
            conversation_id="test-dialog",
            reply="Задачи #99 нет. Проверьте номер.",
            error="user_input: Неизвестные ID задач: [99]",
        ),
    )
    client = TestClient(app)

    for _ in range(3):
        assert client.post("/api/v1/chat", json={"message": "Удали задачу #99"}).status_code == 200

    snapshot = client.get("/api/v1/metrics").json()
    assert snapshot["requests_total"] == 3
    assert snapshot["agent_errors"] == {}
    assert snapshot["retries"] == {}
    assert snapshot["alerts"] == []


def test_agent_errors_are_counted_separately() -> None:
    metrics = MetricsRegistry()
    metrics.record_agent_error("analyst")
    metrics.record_agent_error("analyst")
    metrics.record_agent_error("validator")

    assert metrics.snapshot()["alerts"] == []
    metrics.record_agent_error("analyst")

    snapshot = metrics.snapshot()
    assert snapshot["agent_errors"] == {"analyst": 3, "validator": 1}
    assert [(alert["code"], alert["role"]) for alert in snapshot["alerts"]] == [
        ("agent_error_burst", "analyst")
    ]


def test_old_errors_leave_alert_window(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(metrics_module, "monotonic", lambda: clock[0])
    metrics = MetricsRegistry()

    for moment in (0.0, 1.0, 301.1):
        clock[0] = moment
        metrics.record_agent_error("planner")
    assert metrics.snapshot()["alerts"] == []

    for moment in (302.0, 303.0):
        clock[0] = moment
        metrics.record_agent_error("planner")
    assert len(metrics.snapshot()["alerts"]) == 1

    for moment in (604.0, 605.0, 606.0):
        clock[0] = moment
        metrics.record_agent_error("planner")
    assert len(metrics.snapshot()["alerts"]) == 2


def test_concurrent_requests_keep_all_counters() -> None:
    metrics = MetricsRegistry()

    def record(_: int) -> None:
        metrics.record_request(0.25)
        metrics.record_retry("validator")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record, range(200)))

    snapshot = metrics.snapshot()
    assert snapshot["requests_total"] == 200
    assert snapshot["request_duration_seconds_sum"] == pytest.approx(50.0)
    assert snapshot["retries"] == {"validator": 200}


def test_each_api_app_has_its_own_metrics() -> None:
    first = TestClient(create_app())
    second = TestClient(create_app())

    assert first.post("/api/v1/chat", json={}).status_code == 422
    assert first.get("/api/v1/metrics").json()["requests_total"] == 1
    assert second.get("/api/v1/metrics").json()["requests_total"] == 0


def test_runtime_chat_failure_still_records_duration(monkeypatch, caplog) -> None:
    app = create_app()
    app.dependency_overrides[chat_api.get_workflow] = lambda: object()
    app.dependency_overrides[chat_api.get_db] = lambda: None

    def unavailable(*args, **kwargs):
        raise ConnectionError("Модель недоступна")

    monkeypatch.setattr(chat_api, "run_chat", unavailable)
    client = TestClient(app)

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        assert client.post("/api/v1/chat", json={"message": "Покажи задачи"}).status_code == 503
    snapshot = client.get("/api/v1/metrics").json()
    assert snapshot["requests_total"] == 1
    assert snapshot["request_duration_seconds_sum"] > 0
    assert any(
        record.name == "uvicorn.error" and "chat_request status=503" in record.getMessage()
        for record in caplog.records
    )


def test_three_chat_failures_expose_alert_in_api(monkeypatch, caplog) -> None:
    app = create_app()
    app.dependency_overrides[chat_api.get_workflow] = lambda: object()
    app.dependency_overrides[chat_api.get_db] = lambda: None
    monkeypatch.setattr(
        chat_api,
        "run_chat",
        lambda *args, **kwargs: ChatResult(
            conversation_id="test-dialog",
            reply="Ответ через запасной маршрут",
            decision_error="Tev1 недоступна",
            agent_error_counts={"decision": 1},
        ),
    )
    client = TestClient(app)

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert client.post("/api/v1/chat", json={"message": "Покажи задачи"}).status_code == 200

    snapshot = client.get("/api/v1/metrics").json()
    assert snapshot["requests_total"] == 3
    assert snapshot["agent_errors"] == {"decision": 3}
    assert len(snapshot["alerts"]) == 1
    assert snapshot["alerts"][0]["role"] == "decision"
    assert "Всплеск ошибок агента decision" in caplog.text


def test_replayed_chat_does_not_count_agent_error_twice(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "tasks.db"
    init_db(db_path)
    app = create_app()
    app.dependency_overrides[chat_api.get_workflow] = lambda: object()

    def test_db():
        connection = connect(db_path)
        try:
            yield connection
        finally:
            connection.close()

    app.dependency_overrides[chat_api.get_db] = test_db
    calls = []

    def fake_run_chat(*args, **kwargs):
        calls.append(1)
        return ChatResult(
            conversation_id="test-dialog",
            reply="Попробуйте позже",
            decision_error="Tev1 недоступна",
            validation_attempts=2,
            agent_error_counts={"decision": 1},
        )

    monkeypatch.setattr(chat_api, "run_chat", fake_run_chat)
    client = TestClient(app)
    payload = {"message": "Покажи задачи", "request_id": "metrics-replay-1"}

    first = client.post("/api/v1/chat", json=payload)
    repeated = client.post("/api/v1/chat", json=payload)

    assert first.status_code == repeated.status_code == 200
    assert first.json() == repeated.json()
    assert len(calls) == 1
    snapshot = client.get("/api/v1/metrics").json()
    assert snapshot["requests_total"] == 2
    assert snapshot["agent_errors"] == {"decision": 1}
    assert snapshot["retries"] == {"validator": 1}
