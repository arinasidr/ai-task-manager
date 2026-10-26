import pytest

from ai_task_manager.config import Settings
from ai_task_manager.workflow.router import classify_request, route_request


@pytest.mark.parametrize(
    "message",
    ["Покажи задачи", "ПОКАЖИ ЗАДАЧИ?!", "Какие у меня задачи?"],
)
def test_simple_read_uses_fast_route(message: str) -> None:
    assert classify_request(message) == "simple_read"


@pytest.mark.parametrize(
    "message",
    [
        "Добавь задачу",
        "Покажи задачи и удали №12",
        "Покажи задачи; удали №12",
        "Что у меня на сегодня?",
    ],
)
def test_other_requests_use_full_route(message: str) -> None:
    assert classify_request(message) == "full"


def test_empty_request_is_rejected() -> None:
    with pytest.raises(ValueError, match="Пустой запрос"):
        classify_request("  ?!  ")


def test_router_selects_models_without_changing_state() -> None:
    state = {
        "request": "Покажи задачи",
        "conversation_id": None,
        "current_datetime": "2026-10-26T12:00:00+03:00",
        "timezone": "Europe/Moscow",
    }
    original = state.copy()
    settings = Settings(
        fast_model="fast",
        analyst_model="deep",
        planner_model="planner",
        validator_model="validator",
    )

    result = route_request(state, settings)

    assert state == original
    assert result["analyst_model"] == "fast"
    assert result["validator_model"] == "validator"
