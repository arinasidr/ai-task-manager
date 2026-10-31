"""HTTP-маршрут общения с агентами."""

import sqlite3
from contextlib import ExitStack
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from ai_task_manager.api.dependencies import get_db
from ai_task_manager.api.schemas import ChatRequest, ChatResponse
from ai_task_manager.config import get_settings
from ai_task_manager.db.chat_requests import RequestConflict, claim_request, complete_request
from ai_task_manager.memory.checkpoint import open_checkpointer
from ai_task_manager.workflow.service import create_workflow, run_chat

router = APIRouter(prefix="/api/v1", tags=["chat"])


# Чекпоинтер (память диалога, A11) держит своё соединение с SQLite, пока работает приложение.
_resources = ExitStack()


@lru_cache
def get_workflow() -> Any:
    """Граф собирается один раз; в тестах подменяется через dependency_overrides."""
    settings = get_settings()
    checkpointer = _resources.enter_context(open_checkpointer(settings.database_path))
    return create_workflow(settings, checkpointer=checkpointer)


Workflow = Annotated[Any, Depends(get_workflow)]
Db = Annotated[sqlite3.Connection, Depends(get_db)]


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={503: {"description": "Модели недоступны (Ollama не запущена?)"}},
)
def chat(payload: ChatRequest, workflow: Workflow, db: Db) -> ChatResponse:
    """Сообщение пользователя проходит аналитика, планировщика и валидатора.

    Безопасные действия выполняются сразу. Опасные возвращают `action_id`: подтвердить —
    `POST /api/v1/actions/{action_id}/confirm`, отменить — `/cancel`.
    """
    if payload.request_id is not None:
        try:
            saved = claim_request(db, payload.request_id, payload.message, payload.conversation_id)
        except RequestConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if saved is not None:
            return ChatResponse.model_validate(saved)

    try:
        result = run_chat(workflow, get_settings(), payload.message, payload.conversation_id)
    except ConnectionError as exc:  # pragma: no cover — защита на случай сбоя вне агентов
        raise HTTPException(status_code=503, detail=f"Модели недоступны: {exc}") from exc
    response = ChatResponse(
        conversation_id=result.conversation_id,
        reply=result.reply,
        action_id=result.action_id,
        preference_id=result.preference_id,
    )
    if payload.request_id is not None:
        complete_request(db, payload.request_id, response.model_dump())
    return response
