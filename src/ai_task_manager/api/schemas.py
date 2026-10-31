"""Схемы запросов и ответов HTTP API."""

from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Priority = Literal["low", "medium", "high"]
Status = Literal["todo", "in_progress", "done"]


class TaskCreate(BaseModel):
    title: str = Field(min_length=1)
    description: str | None = None
    priority: Priority | None = None
    due_date: date | None = None
    due_time: time | None = None
    due_timezone: str | None = None
    due_at_utc: datetime | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Название задачи не может быть пусым")
        return value


class TaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    priority: Priority | None = None
    due_date: date | None = None
    due_time: time | None = None
    due_timezone: str | None = None
    due_at_utc: datetime | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str:
        if value is None or not value.strip():
            raise ValueError("Название задачи не может быть пустым")
        return value.strip()


class TaskResponse(BaseModel):
    id: int
    title: str
    description: str | None
    status: Status
    priority: Priority | None
    due_date: date | None
    due_time: time | None
    due_timezone: str | None
    due_at_utc: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime


class TaskStatusUpdate(BaseModel):
    status: Status


class PendingActionResponse(BaseModel):
    action_id: str
    task_id: int
    unfinished_predecessor_ids: list[int]
    message: str


class ConfirmActionResponse(BaseModel):
    action_id: str
    status: Literal["confirmed"]
    applied: bool
    # итог для пользователя (заполняется для действий, подготовленных агентами)
    message: str | None = None


class DeletePendingResponse(BaseModel):
    action_id: str
    task_id: int
    dependencies: dict[str, list[int]]
    message: str


class CancelActionResponse(BaseModel):
    action_id: str
    status: Literal["cancelled"]
    applied: bool


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    request_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Сообщение не может быть пустым")
        return value


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    action_id: str | None = None
    # Предложенное предпочтение (P12): ответьте «да»/«нет» в чате с тем же conversation_id
    # или POST /api/v1/preferences/{preference_id}/confirm | /reject.
    preference_id: int | None = None
