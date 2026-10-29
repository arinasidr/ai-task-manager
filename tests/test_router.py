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


def test_tev1_cannot_route_deletion_as_simple_read() -> None:
    state = {
        "request": "Удали задачу",
        "decision": {
            "request_type": {
                "choice": "delete_task",
                "probabilities": {"delete_task": 0.993},
            },
            "route": {
                "choice": "simple_read",
                "probabilities": {"simple_read": 0.698},
            },
        },
    }

    result = route_request(state, Settings(_env_file=None, analyst_model="deep"))

    assert result["route"] == "full"
    assert result["analyst_model"] == "deep"


def test_tev1_cannot_promote_unknown_request_to_fast_model() -> None:
    state = {
        "request": "Выведи список дел",
        "decision": {
            "request_type": {
                "choice": "read_tasks",
                "probabilities": {"read_tasks": 0.95},
            },
            "route": {
                "choice": "simple_read",
                "probabilities": {"simple_read": 0.85},
            },
        },
    }

    result = route_request(state, Settings(_env_file=None, fast_model="fast"))

    assert result["route"] == "full"


def test_tev1_cannot_hide_mutation_in_mixed_request() -> None:
    state = {
        "request": "Покажи задачи и удали №12",
        "decision": {
            "request_type": {
                "choice": "read_tasks",
                "probabilities": {"read_tasks": 0.99},
            },
            "route": {
                "choice": "simple_read",
                "probabilities": {"simple_read": 0.99},
            },
        },
    }

    result = route_request(state, Settings(_env_file=None, fast_model="fast"))

    assert result["route"] == "full"


def test_confident_full_hint_from_tev1_demotes_exact_read() -> None:
    state = {
        "request": "Покажи задачи",
        "decision": {
            "route": {"choice": "full", "probabilities": {"full": 0.91}},
        },
    }

    result = route_request(state, Settings(_env_file=None, analyst_model="deep"))

    assert result["route"] == "full"
    assert result["analyst_model"] == "deep"


@pytest.mark.parametrize(
    "signal",
    ["needs_clarification", "bypass_confirmation"],
)
def test_tev1_warning_forces_full_route(signal: str) -> None:
    state = {
        "request": "Покажи задачи",
        "decision": {
            signal: {"type": "noul", "noul": 0.8},
        },
    }

    result = route_request(
        state,
        Settings(_env_file=None, analyst_model="deep", fast_model="fast"),
    )

    assert result["route"] == "full"
    assert result["analyst_model"] == "deep"
    assert result["routing_reason"].startswith("Tev1:")
