"""Исполнитель плана: вызывает Tools (функции `db/`) по операциям плана.

- Все операции плана выполняются в одной транзакции: либо весь план, либо ничего.
- Опасность операции определяет **код**, а не флаг в плане: удаление и завершение задачи с
  незавершёнными зависимостями без подтверждения не выполняются, даже если план утверждает,
  что подтверждение не нужно.
- Новые задачи из плана получают ID при создании; ссылки `$new1` в следующих операциях
  заменяются на эти ID.

Исполнитель не делает commit сам: транзакцией управляет вызывающий код (`run_plan`,
`confirmation.confirm_plan_action`).
"""

import sqlite3
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from ai_task_manager.agents.contracts import Plan
from ai_task_manager.db import crud_func
from ai_task_manager.db.preferences import update_preference

_CREATE_ARGS = {
    "title",
    "description",
    "priority",
    "due_date",
    "due_time",
    "due_timezone",
    "due_at_utc",
}


class ConfirmationRequired(PermissionError):
    """План содержит опасные операции, а подтверждения пользователя нет."""


class CreatedTask(BaseModel):
    ref: str | None
    task_id: int
    title: str


class ExecutionResult(BaseModel):
    created: list[CreatedTask] = Field(default_factory=list)
    updated: list[int] = Field(default_factory=list)
    status_changed: list[tuple[int, str]] = Field(default_factory=list)
    deleted: list[int] = Field(default_factory=list)
    dependencies_added: list[tuple[int, int]] = Field(default_factory=list)
    dependencies_removed: list[tuple[int, int]] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    tasks: list[dict[str, Any]] | None = None

    def summary(self) -> str:
        """Ответ пользователю по факту выполнения (а не по намерению плана)."""
        parts: list[str] = []
        for task in self.created:
            parts.append(f"создана задача #{task.task_id} «{task.title}»")
        parts += [f"изменена задача #{task_id}" for task_id in self.updated]
        parts += [
            f"задача #{task_id}: статус {_STATUS_NAMES.get(status, status)}"
            for task_id, status in self.status_changed
        ]
        parts += [f"удалена задача #{task_id}" for task_id in self.deleted]
        parts += [f"задача #{s} теперь выполняется после #{p}" for p, s in self.dependencies_added]
        parts += [f"убрана связь #{p} → #{s}" for p, s in self.dependencies_removed]
        parts += [f"запомнил: {item}" for item in self.preferences]
        if not parts:
            return "Ничего не изменено."
        return "Готово: " + "; ".join(parts) + "."


_STATUS_NAMES = {"todo": "к выполнению", "in_progress": "в работе", "done": "выполнена"}


# ---------------------------------------------------------------- опасные операции


def confirmation_reasons(connection: sqlite3.Connection, plan: Plan) -> list[str]:
    """Почему план требует подтверждения — по фактическим данным БД, а не по флагу плана."""
    reasons: list[str] = []
    for op in plan.operations:
        task_id = op.args.get("task_id")
        if op.tool == "delete_task":
            reasons.append(f"удаление задачи #{task_id} необратимо")
        elif op.tool == "change_task_status" and op.args.get("status") == "done":
            if isinstance(task_id, int):
                blockers = crud_func.get_unfinished_predecessors(connection, task_id)
                if blockers:
                    reasons.append(
                        f"задача #{task_id} зависит от незавершённых "
                        + ", ".join(f"#{b}" for b in blockers)
                    )
    return reasons


# ---------------------------------------------------------------- выполнение


def _resolve(value: Any, refs: dict[str, int]) -> int:
    if isinstance(value, str) and value.startswith("$"):
        if value[1:] not in refs:
            raise ValueError(f"Ссылка {value} на несозданную задачу")
        return refs[value[1:]]
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError(f"Некорректный ID задачи: {value!r}")


def _force_complete(connection: sqlite3.Connection, task_id: int) -> bool:
    """Завершение вопреки незавершённым зависимостям — только после подтверждения."""
    cursor = connection.execute(
        """
        UPDATE tasks SET status = 'done', updated_at = ?, version = version + 1
        WHERE id = ? AND status != 'done'
        """,
        (datetime.now(UTC).isoformat(), task_id),
    )
    return cursor.rowcount == 1


def execute_plan(
    connection: sqlite3.Connection, plan: Plan, *, confirmed: bool = False
) -> ExecutionResult:
    """Выполняет операции плана без commit. При ошибке исключение — откатывает вызывающий."""
    if plan.needs_clarification:
        raise ValueError("План ждёт уточнения от пользователя, выполнять нечего")
    reasons = confirmation_reasons(connection, plan)
    if (reasons or plan.requires_confirmation) and not confirmed:
        raise ConfirmationRequired(
            "Нужно подтверждение пользователя: " + "; ".join(reasons or ["так решил планировщик"])
        )

    result = ExecutionResult()
    refs: dict[str, int] = {}
    for op in plan.operations:
        args = op.args
        if op.tool == "list_tasks":
            result.tasks = crud_func.list_tasks(connection)

        elif op.tool == "create_task":
            values = {key: value for key, value in args.items() if key in _CREATE_ARGS}
            title = values.pop("title")
            task_id = crud_func.create_task(connection, title, **values)
            if op.ref:
                refs[op.ref] = task_id
            result.created.append(CreatedTask(ref=op.ref, task_id=task_id, title=title.strip()))

        elif op.tool == "update_task":
            task_id = _resolve(args.get("task_id"), refs)
            if not crud_func.update_task(connection, task_id, dict(args.get("updates") or {})):
                raise ValueError(f"Задача #{task_id} не найдена")
            result.updated.append(task_id)

        elif op.tool == "change_task_status":
            task_id = _resolve(args.get("task_id"), refs)
            status = args.get("status")
            blocked = status == "done" and crud_func.get_unfinished_predecessors(
                connection, task_id
            )
            changed = (
                _force_complete(connection, task_id)
                if blocked
                else crud_func.change_task_status(connection, task_id, status)
            )
            if not changed:
                raise ValueError(f"Не удалось сменить статус задачи #{task_id}")
            result.status_changed.append((task_id, status))

        elif op.tool == "delete_task":
            task_id = _resolve(args.get("task_id"), refs)
            if not crud_func.delete_task(connection, task_id):
                raise ValueError(f"Задача #{task_id} не найдена")
            result.deleted.append(task_id)

        elif op.tool == "add_dependency":
            pair = (
                _resolve(args.get("predecessor_task_id"), refs),
                _resolve(args.get("successor_task_id"), refs),
            )
            crud_func.add_dependency(connection, *pair)
            result.dependencies_added.append(pair)

        elif op.tool == "remove_dependency":
            pair = (
                _resolve(args.get("predecessor_task_id"), refs),
                _resolve(args.get("successor_task_id"), refs),
            )
            crud_func.remove_dependency(connection, *pair)
            result.dependencies_removed.append(pair)

        elif op.tool == "update_preference":
            for kind, value in args.items():
                saved = update_preference(connection, kind, value)
                shown = ", ".join(saved.value) if isinstance(saved.value, list) else saved.value
                result.preferences.append(f"{_PREFERENCE_NAMES.get(kind, kind)} — {shown}")

        else:  # pragma: no cover — ToolName в контракте не даст сюда попасть
            raise ValueError(f"Неизвестная операция {op.tool}")
    return result


_PREFERENCE_NAMES = {
    "preferred_work_time": "удобное время работы",
    "working_days": "рабочие дни",
    "preferred_task_types": "предпочтительные типы задач",
}


def run_plan(connection: sqlite3.Connection, plan: Plan) -> ExecutionResult:
    """Выполняет безопасный план одной транзакцией (commit или полный откат)."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        result = execute_plan(connection, plan)
    except BaseException:
        connection.rollback()
        raise
    if plan.writes:
        connection.commit()
    else:
        connection.rollback()
    return result
