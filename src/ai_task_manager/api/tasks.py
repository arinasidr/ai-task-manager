"""HTTP-маршруты для задач."""

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ai_task_manager.api.dependencies import get_db
from ai_task_manager.api.schemas import (
    DeletePendingResponse,
    PendingActionResponse,
    TaskCreate,
    TaskResponse,
    TaskStatusUpdate,
    TaskUpdate,
)
from ai_task_manager.db.crud_func import (
    change_task_status,
    create_pending_operation,
    create_task,
    get_dependencies,
    get_task,
    get_unfinished_predecessors,
    list_tasks,
    update_task,
)
from ai_task_manager.db.operation_service import prepare_blocked_completion

router = APIRouter(prefix="/api/v1", tags=["tasks"])
Db = Annotated[sqlite3.Connection, Depends(get_db)]


@router.get("/tasks", response_model=list[TaskResponse])
def read_tasks(db: Db) -> list[dict]:
    """Возвращает список задач."""
    return list_tasks(db)


@router.post("/tasks", response_model=TaskResponse, status_code=201)
def add_task(payload: TaskCreate, db: Db) -> dict:
    """Создаёт задачу и возвращает её вместе с ID."""
    data = payload.model_dump(mode="json", exclude_none=True)
    task_id = create_task(db, **data)
    db.commit()
    return get_task(db, task_id)


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def read_task(task_id: int, db: Db) -> dict:
    """Возвращает задачу по ID."""
    task = get_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task


@router.patch("/tasks/{task_id}", response_model=TaskResponse)
def edit_task(task_id: int, payload: TaskUpdate, db: Db) -> dict:
    updates = payload.model_dump(mode="json", exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=422, detail="Не указано, что изменить")

    try:
        updated = update_task(db, task_id, updates)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not updated:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    db.commit()
    return get_task(db, task_id)


@router.patch(
    "/tasks/{task_id}/status",
    response_model=TaskResponse,
    responses={202: {"model": PendingActionResponse}},
)
def edit_task_status(task_id: int, payload: TaskStatusUpdate, db: Db) -> dict | JSONResponse:
    task = get_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    if task["status"] == payload.status:
        return task

    if payload.status == "done" and get_unfinished_predecessors(db, task_id):
        try:
            pending = prepare_blocked_completion(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return JSONResponse(
            status_code=202,
            content={
                **pending,
                "message": "Есть незавершённые зависимости. Подтвердите завершение задачи.",
            },
        )

    try:
        changed = change_task_status(db, task_id, payload.status)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if not changed:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    db.commit()
    return get_task(db, task_id)


@router.post(
    "/tasks/{task_id}/delete-request",
    response_model=DeletePendingResponse,
    status_code=202,
)
def request_task_deletion(task_id: int, db: Db) -> dict:
    task = get_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    dependencies = get_dependencies(db, task_id)

    try:
        action_id = create_pending_operation(
            db,
            "delete_task",
            {"task_id": task_id},
            [task_id],
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()
    return {
        "action_id": action_id,
        "task_id": task_id,
        "dependencies": dependencies,
        "message": "Удалить задачу и её связи? Подтвердите действие.",
    }
