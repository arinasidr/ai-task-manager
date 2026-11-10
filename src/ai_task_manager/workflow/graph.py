from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ai_task_manager.config import Settings
from ai_task_manager.observability.trajectory import trajectory_node

from .nodes import finish_node
from .router import route_request
from .state import WorkflowState

Node = Callable[[WorkflowState], dict[str, Any]]


def _count_agent_error(
    role: str,
    node: Node,
    is_error: Callable[[dict[str, Any]], bool],
) -> Node:
    def run(state: WorkflowState) -> dict[str, Any]:
        update = node(state)
        if not is_error(update):
            return update

        counts = dict(state.get("agent_error_counts") or {})
        counts[role] = counts.get(role, 0) + 1
        return {**update, "agent_error_counts": counts}

    return run


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


# Сколько раз валидатор может проверить план за один запрос: первая проверка + один повтор
# планировщика с замечаниями. Дальше - ответ пользователю, без зацикливания (ТЗ, FR-20).
MAX_VALIDATION_ATTEMPTS = 2


def build_graph(
    settings: Settings,
    *,
    analyst: Node,
    planner: Node,
    validator: Node,
    act: Node | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    decision: Node | None = None,
    preference_answer: Node | None = None,
    propose_preference: Node | None = None,
):
    """Граф: prepare → [preference_answer] → [decision] → route → analyst → planner → validator →
    act → finish → [propose_preference] → remember_reply."""
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
            "decision": None,
            "decision_error": None,
            # pending_preference не сбрасываем: это вопрос из прошлого хода
            "preference_id": None,
            "agent_error_counts": {},
            "model_retry_counts": {},
            "trajectory": [],
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

    def add_node(name: str, node: Node) -> None:
        # каждый шаг (кроме prepare, который начинает траекторию заново) попадает в
        # state["trajectory"] с длительностью и кратким итогом
        graph.add_node(name, trajectory_node(name, node))

    graph.add_node("prepare", prepare)
    if preference_answer is not None:
        add_node("preference_answer", preference_answer)
    if propose_preference is not None:
        add_node("propose_preference", propose_preference)
    if decision is not None:
        add_node(
            "decision",
            _count_agent_error(
                "decision",
                decision,
                lambda update: update.get("decision_error") is not None,
            ),
        )
    add_node("route", lambda state: route_request(state, settings))
    add_node(
        "analyst",
        _count_agent_error(
            "analyst",
            analyst,
            lambda update: str(update.get("error") or "").startswith("analyst:"),
        ),
    )
    add_node(
        "planner",
        _count_agent_error(
            "planner",
            planner,
            lambda update: (update.get("plan") or {}).get("source") == "fallback",
        ),
    )
    add_node(
        "validator",
        _count_agent_error(
            "validator",
            validator,
            lambda update: (update.get("verdict") or {}).get("source") == "unavailable",
        ),
    )
    add_node("finish", finish_node)
    if act is not None:
        add_node("act", act)
    add_node("remember_reply", remember_reply)

    def after_analyst(state: WorkflowState) -> str:
        analysis = state.get("analysis")
        intent = analysis.get("intent") if isinstance(analysis, dict) else None

        if state.get("route") == "simple_read":
            return "planner" if intent == "get_tasks" else "finish"
        return "planner" if intent in _PLANNABLE else "finish"

    def after_validator(state: WorkflowState) -> str:
        verdict = state.get("verdict") or {}
        if verdict.get("decision") == "accept":
            return "act" if act is not None else "finish"
        plan = state.get("plan") or {}
        attempts = state.get("validation_attempts", 0)
        if plan.get("source") == "llm" and attempts < MAX_VALIDATION_ATTEMPTS:
            return "planner"
        return "finish"

    graph.add_edge(START, "prepare")
    first_step = "decision" if decision is not None else "route"
    if preference_answer is not None:
        graph.add_edge("prepare", "preference_answer")
        graph.add_conditional_edges(
            "preference_answer",
            lambda state: "remember_reply" if state.get("reply") else first_step,
            {"remember_reply": "remember_reply", first_step: first_step},
        )
    else:
        graph.add_edge("prepare", first_step)
    if decision is not None:
        graph.add_edge("decision", "route")
    graph.add_edge("route", "analyst")
    graph.add_conditional_edges(
        "analyst",
        after_analyst,
        {"planner": "planner", "finish": "finish"},
    )
    graph.add_edge("planner", "validator")
    validator_targets = {"planner": "planner", "finish": "finish"}
    if act is not None:
        validator_targets["act"] = "act"
        graph.add_edge("act", "finish")
    graph.add_conditional_edges("validator", after_validator, validator_targets)
    if propose_preference is not None:
        graph.add_edge("finish", "propose_preference")
        graph.add_edge("propose_preference", "remember_reply")
    else:
        graph.add_edge("finish", "remember_reply")
    graph.add_edge("remember_reply", END)

    return graph.compile(checkpointer=checkpointer)
