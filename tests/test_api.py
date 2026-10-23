from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_task_manager.api import tasks as tasks_api
from ai_task_manager.api.app import create_app
from ai_task_manager.config import get_settings
from ai_task_manager.db.crud_func import add_dependency, create_task
from ai_task_manager.db.database import connect


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("ATM_DATABASE_PATH", str(tmp_path / "tasks.db"))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        get_settings.cache_clear()


def test_delete_requires_confirmation_and_can_be_cancelled(client: TestClient) -> None:
    created = client.post("/api/v1/tasks", json={"title": "  Написать отчёт  "})
    assert created.status_code == 201
    task_id = created.json()["id"]
    assert created.json()["title"] == "Написать отчёт"

    pending = client.post(f"/api/v1/tasks/{task_id}/delete-request")
    assert pending.status_code == 202
    action_id = pending.json()["action_id"]
    assert client.get(f"/api/v1/tasks/{task_id}").status_code == 200

    cancelled = client.post(f"/api/v1/actions/{action_id}/cancel")
    assert cancelled.json()["applied"] is True
    assert client.post(f"/api/v1/actions/{action_id}/confirm").status_code == 409
    assert client.get(f"/api/v1/tasks/{task_id}").status_code == 200

    action_id = client.post(f"/api/v1/tasks/{task_id}/delete-request").json()["action_id"]
    confirmed = client.post(f"/api/v1/actions/{action_id}/confirm")
    assert confirmed.json()["applied"] is True
    assert client.get(f"/api/v1/tasks/{task_id}").status_code == 404
    assert client.post(f"/api/v1/actions/{action_id}/confirm").json()["applied"] is False


def test_blocked_completion_waits_for_confirmation(client: TestClient) -> None:
    first = client.post("/api/v1/tasks", json={"title": "Первая"}).json()["id"]
    second = client.post("/api/v1/tasks", json={"title": "Вторая"}).json()["id"]

    with closing(connect()) as db:
        add_dependency(db, first, second)
        db.commit()

    pending = client.patch(
        f"/api/v1/tasks/{second}/status",
        json={"status": "done"},
    )
    assert pending.status_code == 202
    assert pending.json()["unfinished_predecessor_ids"] == [first]
    assert client.get(f"/api/v1/tasks/{second}").json()["status"] == "todo"

    action_id = pending.json()["action_id"]
    confirmed = client.post(f"/api/v1/actions/{action_id}/confirm")
    assert confirmed.json()["applied"] is True
    assert client.get(f"/api/v1/tasks/{second}").json()["status"] == "done"


def test_dependency_added_during_status_change_prevents_completion(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = client.post("/api/v1/tasks", json={"title": "Предшественник"}).json()["id"]
    second = client.post("/api/v1/tasks", json={"title": "Следующая задача"}).json()["id"]
    change_status = tasks_api.change_task_status

    def add_dependency_now() -> None:
        with closing(connect()) as other_db:
            add_dependency(other_db, first, second)
            other_db.commit()

    class InterleavingConnection:
        def __init__(self, db) -> None:
            self.db = db
            self.injected = False

        def execute(self, sql, parameters=()):
            statement = sql.lstrip().upper()
            if not self.injected and statement.startswith("UPDATE TASKS"):
                add_dependency_now()
                self.injected = True
            cursor = self.db.execute(sql, parameters)
            if not self.injected and statement.startswith("SELECT"):
                add_dependency_now()
                self.injected = True
            return cursor

    def change_with_interleaving(db, task_id: int, status: str) -> bool:
        return change_status(InterleavingConnection(db), task_id, status)

    monkeypatch.setattr(tasks_api, "change_task_status", change_with_interleaving)
    response = client.patch(f"/api/v1/tasks/{second}/status", json={"status": "done"})

    assert response.status_code == 409
    assert client.get(f"/api/v1/tasks/{second}").json()["status"] == "todo"


def test_task_created_by_crud_with_deadline_is_readable_over_api(client: TestClient) -> None:
    with closing(connect()) as db:
        task_id = create_task(
            db,
            "Со сроком",
            due_date="2026-10-23",
            due_time="12:30",
            due_at_utc="2026-10-23T09:30:00+00:00",
        )
        db.commit()

    response = client.get("/api/v1/tasks")
    assert response.status_code == 200
    assert response.json()[0]["id"] == task_id
    assert response.json()[0]["due_date"] == "2026-10-23"
