"""Сборка рабочего workflow из настоящих агентов и обработка одного сообщения пользователя.

    создать граф:  create_workflow(settings)
    обработать:    run_chat(graph, settings, "Удали задачу #5")

Порядок узлов: prepare → route (выбор моделей) → analyst → planner → validator →
(при отказе — снова planner с замечаниями, не больше MAX_VALIDATION_ATTEMPTS проверок) →
act (выполнить или поставить на подтверждение) → finish (ответ пользователю).
"""

import sqlite3
import uuid
from collections.abc import Callable
from datetime import datetime
from functools import cache
from typing import Any
from zoneinfo import ZoneInfo

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from ai_task_manager.agents.llm import JsonLLM, OllamaJsonLLM
from ai_task_manager.agents.validator import make_validator_node
from ai_task_manager.config import Settings
from ai_task_manager.db.database import connect as db_connect
from ai_task_manager.memory.checkpoint import run_turn
from ai_task_manager.tools.confirmation import make_action_node

from .graph import build_graph
from .nodes import make_analyst_node, make_planner_node


class ChatResult(BaseModel):
    conversation_id: str
    reply: str
    action_id: str | None = None
    # для трейсов и отладки: какие модели работали и чем закончилась проверка
    route: str | None = None
    models: dict[str, str] = {}
    verdict: dict[str, Any] | None = None
    validation_attempts: int = 0
    error: str | None = None
    analysis: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    history_length: int = 0


def ollama_factory(settings: Settings) -> Callable[[str], JsonLLM]:
    """Один клиент на модель на всё время работы приложения."""

    @cache
    def llm_for(model: str) -> JsonLLM:
        return OllamaJsonLLM(model, settings.ollama_base_url)

    return llm_for


def create_workflow(
    settings: Settings,
    *,
    connect: Callable[[], sqlite3.Connection] | None = None,
    llm_for: Callable[[str], JsonLLM] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Граф из настоящих агентов. С `checkpointer` (A11) граф помнит диалог по conversation_id."""
    connect = connect or (lambda: db_connect(settings.database_path))
    llm_for = llm_for or ollama_factory(settings)
    return build_graph(
        settings,
        analyst=make_analyst_node(llm_for=llm_for, connect=connect),
        planner=make_planner_node(llm_for=llm_for, connect=connect),
        validator=make_validator_node(llm_for=llm_for, connect=connect),
        act=make_action_node(connect=connect),
        checkpointer=checkpointer,
    )


def run_chat(
    graph: Any,
    settings: Settings,
    message: str,
    conversation_id: str | None = None,
    *,
    now: datetime | None = None,
) -> ChatResult:
    now = now or datetime.now(ZoneInfo(settings.timezone))
    request = {
        "request": message,
        "conversation_id": conversation_id,
        "current_datetime": now.isoformat(),
        "timezone": settings.timezone,
    }
    if isinstance(graph.checkpointer, SqliteSaver):
        # память диалога (A11): история по conversation_id, ходы одного диалога по очереди
        state = run_turn(graph, request)
    else:
        request["conversation_id"] = conversation_id or str(uuid.uuid4())
        state = graph.invoke(request)
    return ChatResult(
        conversation_id=state["conversation_id"],
        reply=state.get("reply") or "",
        action_id=state.get("pending_action_id"),
        route=state.get("route"),
        models={
            role: state[f"{role}_model"]
            for role in ("analyst", "planner", "validator")
            if state.get(f"{role}_model")
        },
        verdict=state.get("verdict"),
        validation_attempts=state.get("validation_attempts", 0),
        error=state.get("error"),
        analysis=state.get("analysis"),
        plan=state.get("plan"),
        history_length=len(state.get("history") or []),
    )
