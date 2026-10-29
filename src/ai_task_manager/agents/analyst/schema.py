from datetime import date, datetime
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Priority = Literal["low", "medium", "high"]
Status = Literal["todo", "in_progress", "done"]
Intent = Literal[
    "create_task",
    "get_tasks",
    "update_task",
    "delete_task",
    "delete_tasks",
    "plan_tasks",
    "update_preference",
    "clarification_required",
    "unsupported",
]


def normalize_deadline(value: str) -> str:
    if "T" in value:
        try:
            moment = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("Неккоректные дата и время") from exc

        if moment.utcoffset() is None:
            raise ValueError("Для срока со временем нужен часовой пояс")

        return moment.isoformat()

    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError("Некорректная дата") from exc


Deadline = Annotated[str, AfterValidator(normalize_deadline)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")  # запрещаем неожиданные поля


class TaskSnapshot(StrictModel):
    id: int = Field(gt=0)
    title: str
    status: Status
    priority: Priority | None = None
    due_date: str | None = None
    due_time: str | None = None
    due_timezone: str | None = None


class DialogueContext(StrictModel):
    role: Literal["user", "assistant"]
    content: str


class AnalystInput(StrictModel):
    request: str = Field(min_length=1)
    current_datetime: AwareDatetime
    timezone: str
    tasks: list[TaskSnapshot] = Field(default_factory=list)
    history: list[DialogueContext] = Field(default_factory=list)


class ExtractedTask(StrictModel):
    title: str = Field(min_length=1)
    priority: Priority | None = None
    deadline: Deadline | None = None


class ExtractedDependency(StrictModel):
    task: str
    depends_on: str


class ExtractedChanges(StrictModel):
    title: str | None = None
    deadline: Deadline | None = None
    priority: Priority | None = None
    status: Status | None = None
    add_dependency_ids: list[int] = Field(default_factory=list)
    remove_dependency_ids: list[int] = Field(default_factory=list)


class ExtractedPreference(StrictModel):
    kind: Literal["preferred_work_time", "working_days", "preferred_task_types"]
    value: str | list[str]


class AnalystResult(StrictModel):
    intent: Intent
    tasks: list[ExtractedTask] = Field(default_factory=list)
    dependencies: list[ExtractedDependency] = Field(default_factory=list)
    task_id: int | None = Field(default=None, gt=0)
    task_ids: list[int] = Field(default_factory=list)
    updates: ExtractedChanges | None = None
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    clarification_question: str | None = None

    preference: ExtractedPreference | None = None

    @model_validator(mode="after")
    def check_intent_payload(self) -> Self:
        if self.intent == "create_task" and not self.tasks:
            raise ValueError("Для создания нужна хотя бы одна задача")

        if self.intent == "update_task":
            if self.task_id is None or self.updates is None:
                raise ValueError("Для изменения нужны ID задачи и изменения")
            values = self.updates.model_dump(exclude_none=True).values()
            if not any(value != [] and value != "" for value in values):
                raise ValueError("Список изменений пуст")

        if self.intent == "delete_task" and self.task_id is None:
            raise ValueError("Для удаления нужен ID задачи")

        if self.intent == "delete_tasks" and not self.task_ids:
            raise ValueError("Для удаления нужны ID задач")

        if self.intent == "update_preference" and self.preference is None:
            raise ValueError("Не указано предпочтение")

        if self.intent == "clarification_required":
            if not self.missing_fields and not self.clarification_question:
                raise ValueError("Нужно указать, что уточнить")

        return self
