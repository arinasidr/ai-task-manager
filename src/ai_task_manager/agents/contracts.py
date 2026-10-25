"""Контракты между агентами: что аналитик передаёт планировщику и что планировщик отдаёт дальше.

`Analysis` — вход планировщика. Принимает и результат аналитика (`AnalystResult` из
`agents/analyst/schema.py`), и формат ответа из бенчмарка (`benchmarks/schema.py`): различия
(пустые поля в `updates`, `preference`, `clarification_question`) выравниваются при разборе.
`Plan` — выход планировщика: его проверяют валидатор и код, а исполняют Tools.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Priority = Literal["low", "medium", "high"]

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

# Коды предупреждений общие с бенчмарком, чтобы аналитик, планировщик и валидатор говорили на
# одном языке.
Warning = Literal[
    "task_not_found",
    "dependency_cycle",
    "incomplete_dependencies",
    "dependent_tasks_will_be_detached",
    "dependency_deadline_conflict",
    "invalid_date",
    "ambiguous_local_time",
    "past_deadline",
    "ambiguous_request",
]

ToolName = Literal[
    "list_tasks",
    "create_task",
    "update_task",
    "change_task_status",
    "delete_task",
    "add_dependency",
    "remove_dependency",
    "update_preference",
]

# Операции, которые меняют данные. list_tasks только читает.
WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "create_task",
        "update_task",
        "change_task_status",
        "delete_task",
        "add_dependency",
        "remove_dependency",
        "update_preference",
    }
)


class ExtractedTask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    priority: Priority | None = None
    deadline: str | None = None


class ExtractedDependency(BaseModel):
    """Связь между задачами из одного сообщения: `task` выполняется после `depends_on`."""

    model_config = ConfigDict(extra="ignore")

    task: str
    depends_on: str


class Analysis(BaseModel):
    """Разбор запроса аналитиком (тот же формат, что в бенчмарке)."""

    model_config = ConfigDict(extra="ignore")

    intent: Intent
    tasks: list[ExtractedTask] = Field(default_factory=list)
    dependencies: list[ExtractedDependency] = Field(default_factory=list)
    task_id: int | None = None
    task_ids: list[int] = Field(default_factory=list)
    updates: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    message: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_analyst_result(cls, data: Any) -> Any:
        """Приводит результат аналитика к единому виду.

        - `AnalystResult` (Pydantic-модель) превращается в словарь;
        - из `updates` убираются незаполненные поля: у аналитика `None` значит «не менять»;
        - `preference = {kind, value}` переносится в `updates` (как в бенчмарке);
        - `clarification_question` становится `message`.
        """
        if isinstance(data, BaseModel):
            data = data.model_dump()
        if not isinstance(data, dict):
            return data
        data = dict(data)
        updates = {
            key: value
            for key, value in (data.get("updates") or {}).items()
            if value is not None and value != []
        }
        preference = data.pop("preference", None)
        if isinstance(preference, dict) and preference.get("kind"):
            updates[preference["kind"]] = preference.get("value")
        data["updates"] = updates
        question = data.pop("clarification_question", None)
        if question and not data.get("message"):
            data["message"] = question
        return data


class Operation(BaseModel):
    """Один вызов Tool. Новые задачи ещё не имеют ID, поэтому на них ссылаются через `ref`.

    Пример: `create_task` с `ref="new1"`, затем `add_dependency` с
    `args={"predecessor_task_id": "$new1", "successor_task_id": "$new2"}`.
    """

    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    ref: str | None = None


class Recommendation(BaseModel):
    task_id: int
    reason: str
    recommended_priority: Priority | None = None


class Plan(BaseModel):
    intent: Intent
    operations: list[Operation] = Field(default_factory=list)
    ordered_task_ids: list[int] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    needs_clarification: bool = False
    message_to_user: str = ""
    # Пояснения для валидатора и трейсов: почему план такой и что поправил код.
    reasoning: list[str] = Field(default_factory=list)
    # rules — план построен кодом; llm — моделью; fallback — модель не справилась, план по правилам.
    source: Literal["rules", "llm", "fallback"] = "rules"

    @property
    def writes(self) -> bool:
        return any(op.tool in WRITE_TOOLS for op in self.operations)
