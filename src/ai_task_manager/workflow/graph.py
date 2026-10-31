from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ai_task_manager.config import Settings

from .nodes import finish_node
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


# Сколько раз валидатор может проверить план за один запрос: первая проверка + один повтор
# планировщика с замечаниями. Дальше — ответ пользователю, без зацикливания (ТЗ, FR-20).
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
    """Граф: prepare → [preference_answer] → [decision] → route → analyst → planner →
    validator → act → finish → [propose_preference] → remember_reply.

    Узлы в скобках можно пропустить в изолированных тестах. `preference_answer` отвечает на
    «да/нет» по предложенному предпочтению и сразу ведёт к `remember_reply` (P12).

    После отказа валидатора план возвращается планировщику с замечаниями, если план строила
    модель (`source = "llm"`) и лимит проверок не исчерпан. План, построенный кодом, от
    замечаний не изменится — повтор бессмыслен, сразу ответ пользователю. `finish` гарантирует
    ответ, `remember_reply` сохраняет его в историю диалога.
    """
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
            # pending_preference не сбрасываем: это вопрос из прошлого хода (P12)
            "preference_id": None,
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
    if preference_answer is not None:
        graph.add_node("preference_answer", preference_answer)
    if propose_preference is not None:
        graph.add_node("propose_preference", propose_preference)
    if decision is not None:
        graph.add_node("decision", decision)
    graph.add_node("route", lambda state: route_request(state, settings))
    graph.add_node("analyst", analyst)
    graph.add_node("planner", planner)
    graph.add_node("validator", validator)
    graph.add_node("finish", finish_node)
    if act is not None:
        graph.add_node("act", act)
    graph.add_node("remember_reply", remember_reply)

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
