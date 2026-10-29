"""Вопросы к Tev1 для предварительной оценки запроса."""

from collections.abc import Callable
from typing import Any

from ai_task_manager.config import Settings

from .decision_client import DecisionModelError, SystemOneClient
from .state import WorkflowState

DECISION_QUESTIONS: dict[str, dict[str, Any]] = {
    "request_type": {
        "type": "choice",
        "instructions": "What is the user's main intent?",
        "criteria": {
            "read_tasks": "Show or search existing tasks without changing them.",
            "create_task": "Create a new task.",
            "update_task": "Change an existing task or mark it complete.",
            "delete_task": "Delete one or more tasks.",
            "plan_tasks": "Suggest what to do first or in which order.",
            "other": "None of these intents clearly fits.",
        },
    },
    "route": {
        "type": "choice",
        "instructions": "How should this request be processed?",
        "criteria": {
            "simple_read": "Only asks to show the existing task list; changes nothing.",
            "full": "Any other request, including planning, unclear requests or changes.",
        },
    },
    "needs_clarification": {
        "type": "noul",
        "instructions": (
            "Is information required to understand or perform the request missing? "
            "For example, 'Delete the task' without identifying which task."
        ),
    },
    "bypass_confirmation": {
        "type": "noul",
        "instructions": (
            "Does the user ask to skip or bypass confirmation of a potentially destructive action?"
        ),
    },
}


def make_decision_node(
    settings: Settings,
    *,
    client: SystemOneClient | None = None,
) -> Callable[[WorkflowState], dict[str, Any]]:
    client = client or SystemOneClient(settings.decision_model, settings.ollama_base_url)

    def assess(state: WorkflowState) -> dict[str, Any]:
        try:
            answers = client.ask(state["request"], DECISION_QUESTIONS)
        except DecisionModelError as exc:
            return {"decision": None, "decision_error": str(exc)}
        return {"decision": answers, "decision_error": None}

    return assess
