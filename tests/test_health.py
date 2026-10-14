from fastapi.testclient import TestClient

from ai_task_manager import __version__
from ai_task_manager.api.app import create_app


def test_health_returns_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_openapi_is_available() -> None:
    client = TestClient(create_app())
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "AI Task Manager"
