from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from ai_task_manager.config import Settings

from .router import route_request
from .state import WorkflowState

Node = Callable[[WorkflowState], dict[str, Any]]

_PLANNABLE = frozenset(
    {
        "create_task",
        "get_tasks",
        "update_task",
        "delete_task",
        "delete_tasks",
        "plan_tasks",
        "update_preference",
    }
)


def build_graph(
    settings: Settings,
    *,
    analyst: Node,
    planner: Node,
    validator: Node,
):
    graph = StateGraph(WorkflowState)

    def prepare(_: WorkflowState) -> dict[str, Any]:
        return {
            "analysis": None,
            "plan": None,
            "verdict": None,
            "reply": None,
            "error": None,
            "pending_action_id": None,
            "validator_feedback": [],
            "validation_attempts": 0,
            "fast_retry_done": False,
        }

    graph.add_node("prepare", prepare)
    graph.add_node("route", lambda state: route_request(state, settings))
    graph.add_node("analyst", analyst)
    graph.add_node("planner", planner)
    graph.add_node("validator", validator)

    def after_analyst(state: WorkflowState) -> str:
        analysis = state.get("analysis")
        intent = analysis.get("intent") if isinstance(analysis, dict) else None

        if state.get("route") == "simple_read":
            return "planner" if intent == "get_tasks" else END
        return "planner" if intent in _PLANNABLE else END

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "route")
    graph.add_edge("route", "analyst")
    graph.add_conditional_edges(
        "analyst",
        after_analyst,
        {"planner": "planner", END: END},
    )
    graph.add_edge("planner", "validator")
    graph.add_edge("validator", END)

    return graph.compile()
