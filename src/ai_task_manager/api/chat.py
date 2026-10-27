"""HTTP-маршрут общения с агентами."""

from contextlib import ExitStack
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from ai_task_manager.api.schemas import ChatRequest, ChatResponse
from ai_task_manager.config import get_settings
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


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={503: {"description": "Модели недоступны (Ollama не запущена?)"}},
)
def chat(payload: ChatRequest, workflow: Workflow) -> ChatResponse:
    """Сообщение пользователя проходит аналитика, планировщика и валидатора.

    Безопасные действия выполняются сразу. Опасные возвращают `action_id`: подтвердить —
    `POST /api/v1/actions/{action_id}/confirm`, отменить — `/cancel`.
    """
    try:
        result = run_chat(workflow, get_settings(), payload.message, payload.conversation_id)
    except ConnectionError as exc:  # pragma: no cover — защита на случай сбоя вне агентов
        raise HTTPException(status_code=503, detail=f"Модели недоступны: {exc}") from exc
    return ChatResponse(
        conversation_id=result.conversation_id,
        reply=result.reply,
        action_id=result.action_id,
    )
