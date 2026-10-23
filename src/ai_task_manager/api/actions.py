"""HTTP-маршруты подтверждения действий."""

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ai_task_manager.api.dependencies import get_db
from ai_task_manager.api.schemas import CancelActionResponse, ConfirmActionResponse
from ai_task_manager.db.crud_func import cancel_operation, get_operation
from ai_task_manager.db.operation_service import (
    confirm_blocked_completion,
    confirm_delete_task,
)

router = APIRouter(prefix="/api/v1/actions", tags=["actions"])
Db = Annotated[sqlite3.Connection, Depends(get_db)]


@router.post("/{action_id}/confirm", response_model=ConfirmActionResponse)
def confirm_action(action_id: str, db: Db) -> dict:
    """Подтверждает ожидающее действие по его типу."""
    operation = get_operation(db, action_id)

    if operation is None:
        raise HTTPException(status_code=404, detail="Действие не найдено")

    if operation["type"] == "complete_task_with_blockers":
        confirm = confirm_blocked_completion
    elif operation["type"] == "delete_task":
        confirm = confirm_delete_task
    else:
        raise HTTPException(status_code=409, detail="Неизвестный тип действия")

    try:
        applied = confirm(action_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "action_id": action_id,
        "status": "confirmed",
        "applied": applied,
    }


@router.post("/{action_id}/cancel", response_model=CancelActionResponse)
def cancel_action(action_id: str, db: Db) -> dict:
    """Отменяет ожидающее действие, не меняя задачи."""
    operation = get_operation(db, action_id)
    if operation is None:
        raise HTTPException(status_code=404, detail="Действие не найдено")

    if operation["status"] == "cancelled":
        return {"action_id": action_id, "status": "cancelled", "applied": False}

    if operation["status"] != "pending":
        raise HTTPException(status_code=409, detail="Действие уже обработано")

    if not cancel_operation(db, action_id):
        raise HTTPException(status_code=409, detail="Не удалось отменить действие")

    db.commit()
    return {"action_id": action_id, "status": "cancelled", "applied": True}
