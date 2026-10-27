from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
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
    checkpointer: BaseCheckpointSaver | None = None,
):
    graph = StateGraph(WorkflowState)

    def prepare(state: WorkflowState) -> dict[str, Any]:
        history = (state.get("history") or [])[-5:]
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
            "history": [
                *history,
                {
                    "role": "user",
                    "content": state["request"],
                    "created_at": state["current_datetime"],
                },
            ],
        }

    def remember_reply(state: WorkflowState) -> dict[str, Any]:
        reply = state.get("reply")
        if not reply:
            return {}
        return {
            "history": [
                *(state.get("history") or [])[-5:],
                {
                    "role": "assistant",
                    "content": reply,
                    "created_at": state["current_datetime"],
                },
            ]
        }

    graph.add_node("prepare", prepare)
    graph.add_node("route", lambda state: route_request(state, settings))
    graph.add_node("analyst", analyst)
    graph.add_node("planner", planner)
    graph.add_node("validator", validator)
    graph.add_node("remember_reply", remember_reply)

    def after_analyst(state: WorkflowState) -> str:
        analysis = state.get("analysis")
        intent = analysis.get("intent") if isinstance(analysis, dict) else None

        if state.get("route") == "simple_read":
            return "planner" if intent == "get_tasks" else "remember_reply"
        return "planner" if intent in _PLANNABLE else "remember_reply"

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "route")
    graph.add_edge("route", "analyst")
    graph.add_conditional_edges(
        "analyst",
        after_analyst,
        {"planner": "planner", "remember_reply": "remember_reply"},
    )
    graph.add_edge("planner", "validator")
    graph.add_edge("validator", "remember_reply")
    graph.add_edge("remember_reply", END)

    return graph.compile(checkpointer=checkpointer)
