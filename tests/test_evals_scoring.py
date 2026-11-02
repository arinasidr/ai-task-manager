"""Проверки оценщика A16 без моделей и сети."""

from evals.scoring import score_analysis, score_db, score_plan


def db_state(*titles: str) -> dict:
    return {
        "tasks": [{"id": index, "title": title} for index, title in enumerate(titles, start=1)],
        "dependencies": [],
        "preferences": [],
    }


def test_missing_plan_fails_even_without_dependencies() -> None:
    state = db_state("Сдать отчёт")
    case = {"expected": {"plan": {"task_titles": ["Сдать отчёт"], "precedence": []}}}
    run = {
        "before": state,
        "after": state,
        "result": {"plan": {"ordered_task_ids": [], "warnings": []}},
    }

    assert not score_plan(case, run)["passed"]


def test_missing_priority_fails_task_extraction() -> None:
    case = {
        "expected": {
            "analysis": {
                "intent": "create_task",
                "tasks": [{"title": "Разобрать архив", "priority": "low", "deadline": None}],
            }
        }
    }
    run = {
        "result": {
            "analysis": {
                "intent": "create_task",
                "tasks": [{"title": "Разобрать архив", "priority": None, "deadline": None}],
            }
        }
    }

    assert not score_analysis(case, run)["checks"]["tasks"]


def test_pending_deletion_requires_action_id() -> None:
    state = db_state("Черновик")
    case = {"expected": {"pending_action": True, "db_unchanged": True}}
    run = {"before": state, "after": state, "result": {"action_id": None}}

    assert not score_db(case, run)["checks"]["confirmation_correct"]


def test_mutation_before_user_confirmation_fails() -> None:
    before = db_state("Черновик")
    case = {
        "follow_up": "cancel",
        "expected": {
            "pending_action": True,
            "follow_up_status": "cancelled",
            "follow_up_applied": True,
            "db_unchanged": True,
        },
    }
    run = {
        "before": before,
        "after_request": db_state(),
        "after": before,
        "result": {"action_id": "action-1"},
        "follow_up_result": {"status": "cancelled", "applied": True},
    }

    assert not score_db(case, run)["checks"]["unchanged_before_confirmation"]


def test_confirmed_status_without_actual_deletion_fails() -> None:
    before = db_state("Черновик", "Отчёт")
    case = {
        "follow_up": "confirm",
        "expected": {
            "pending_action": True,
            "follow_up_status": "confirmed",
            "follow_up_applied": True,
            "final_db": {
                "tasks": [{"title": "Отчёт"}],
                "dependencies": [],
            },
        },
    }
    run = {
        "before": before,
        "after_request": before,
        "after": before,
        "result": {"action_id": "action-1"},
        "follow_up_result": {"status": "confirmed", "applied": True},
    }

    assert not score_db(case, run)["checks"]["task_titles"]


def test_confirmed_deletion_passes_when_only_target_is_removed() -> None:
    before = db_state("Черновик", "Отчёт")
    case = {
        "follow_up": "confirm",
        "expected": {
            "pending_action": True,
            "follow_up_status": "confirmed",
            "follow_up_applied": True,
            "final_db": {
                "tasks": [{"title": "Отчёт"}],
                "dependencies": [],
            },
        },
    }
    run = {
        "before": before,
        "after_request": before,
        "after": db_state("Отчёт"),
        "result": {"action_id": "action-1"},
        "follow_up_result": {"status": "confirmed", "applied": True},
    }

    assert score_db(case, run)["passed"]
