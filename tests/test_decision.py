from typing import Any

import httpx
import pytest

from ai_task_manager.config import Settings
from ai_task_manager.workflow.decision import DECISION_QUESTIONS, make_decision_node
from ai_task_manager.workflow.decision_client import DecisionModelError, SystemOneClient
from ai_task_manager.workflow.router import route_request


def test_client_calls_systemone_and_returns_answers() -> None:
    sent: dict[str, Any] = {}

    def fake_post(url: str, *, json: dict, timeout: float) -> httpx.Response:
        sent.update(url=url, payload=json, timeout=timeout)
        return httpx.Response(
            200,
            json={"answers": {"route": {"choice": "full"}}},
            request=httpx.Request("POST", url),
        )

    client = SystemOneClient("tev1:test", "http://ollama/", post=fake_post)
    answers = client.ask("Удали задачу", DECISION_QUESTIONS)

    assert sent["url"] == "http://ollama/v1/systemone"
    assert sent["payload"]["model"] == "tev1:test"
    assert sent["payload"]["state"] == "Удали задачу"
    assert answers["route"]["choice"] == "full"


def test_client_rejects_response_without_answers() -> None:
    def fake_post(url: str, *, json: dict, timeout: float) -> httpx.Response:
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    client = SystemOneClient("tev1:test", "http://ollama", post=fake_post)

    with pytest.raises(DecisionModelError, match="без answers"):
        client.ask("Покажи задачи", DECISION_QUESTIONS)


def test_unavailable_tev1_falls_back_to_full_route() -> None:
    def unavailable(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("Ollama недоступна")

    settings = Settings(_env_file=None)
    client = SystemOneClient("tev1:test", "http://ollama", post=unavailable)
    decision = make_decision_node(settings, client=client)({"request": "Удали задачу"})
    route = route_request({"request": "Удали задачу", **decision}, settings)

    assert decision["decision"] is None
    assert decision["decision_error"]
    assert route["route"] == "full"
