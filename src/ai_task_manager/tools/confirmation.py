"""Подтверждение опасных операций из агентного пути.

Путь: валидатор принял план → если план опасный, он сохраняется как ожидающее действие
(`operations`, тип `agent_plan`) вместе с предварительным просмотром → пользователь видит, что
именно изменится, и подтверждает (`POST /api/v1/actions/{id}/confirm`) или отменяет (`/cancel`).

Почему ожидающее действие в БД, а не пауза графа (`interrupt()`): подтверждение приходит
отдельным HTTP-запросом, возможно после перезапуска сервера. Таблица `operations` (A07) уже
хранит срок действия (10 минут) и версии затронутых задач, поэтому, если задачу изменили после
показа плана, подтверждение не сработает.

При подтверждении план проверяется кодом ещё раз и выполняется одной транзакцией.
"""

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ai_task_manager.agents.contracts import Plan
from ai_task_manager.agents.planner import _deadline_text
from ai_task_manager.agents.validator import Validator
from ai_task_manager.db import crud_func
from ai_task_manager.db.database import connect
from ai_task_manager.tools.executor import (
    ExecutionResult,
    confirmation_reasons,
    execute_plan,
    run_plan,
)

ACTION_TYPE = "agent_plan"

_STATUS_TEXT = {"todo": "к выполнению", "in_progress": "в работе", "done": "выполненной"}
_LIST_STATUS = {"todo": "к выполнению", "in_progress": "в работе", "done": "выполнена"}
_PRIORITY_TEXT = {"low": "низкий", "medium": "средний", "high": "высокий"}
_PREFERENCE_TEXT = {
    "preferred_work_time": "удобное время работы",
    "working_days": "рабочие дни",
    "preferred_task_types": "предпочтительные типы задач",
}


# ---------------------------------------------------------------- предварительный план


def _task_name(connection: sqlite3.Connection, value: Any, titles: dict[str, str]) -> str:
    if isinstance(value, str) and value.startswith("$"):
        return f"«{titles.get(value[1:], value)}» (новая)"
    task = crud_func.get_task(connection, value) if isinstance(value, int) else None
    return f"#{value} «{task['title']}»" if task else f"#{value}"


def _changes_text(updates: dict[str, Any]) -> str:
    parts = []
    if "title" in updates:
        parts.append(f"название → «{updates['title']}»")
    if "priority" in updates:
        parts.append(f"приоритет → {_PRIORITY_TEXT.get(updates['priority'], 'без приоритета')}")
    if any(key.startswith("due_") for key in updates):
        parts.append(f"срок → {_deadline_text(updates) or 'без срока'}")
    if "description" in updates:
        parts.append("описание")
    return ", ".join(parts) or "без изменений"


def describe_plan(connection: sqlite3.Connection, plan: Plan) -> list[str]:
    """Что именно изменится — по строке на операцию, с названиями задач из БД."""
    titles = {op.ref: op.args.get("title", "") for op in plan.operations if op.ref}
    lines: list[str] = []
    for op in plan.operations:
        args = op.args
        if op.tool == "create_task":
            extra = []
            if args.get("due_date"):
                extra.append(f"срок {_deadline_text(args)}")
            if args.get("priority"):
                extra.append(f"приоритет {_PRIORITY_TEXT[args['priority']]}")
            suffix = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"Создать задачу «{args.get('title')}»{suffix}")
        elif op.tool == "update_task":
            name = _task_name(connection, args.get("task_id"), titles)
            lines.append(f"Изменить {name}: {_changes_text(args.get('updates') or {})}")
        elif op.tool == "change_task_status":
            task_id = args.get("task_id")
            line = (
                f"Отметить {_task_name(connection, task_id, titles)} "
                f"{_STATUS_TEXT.get(args.get('status'), args.get('status'))}"
            )
            if args.get("status") == "done" and isinstance(task_id, int):
                blockers = crud_func.get_unfinished_predecessors(connection, task_id)
                if blockers:
                    line += ", хотя не завершены задачи " + ", ".join(f"#{b}" for b in blockers)
            lines.append(line)
        elif op.tool == "delete_task":
            task_id = args.get("task_id")
            line = f"Удалить {_task_name(connection, task_id, titles)} безвозвратно"
            if isinstance(task_id, int) and crud_func.get_task(connection, task_id):
                links = crud_func.get_dependencies(connection, task_id)
                linked = sorted({*links["predecessors"], *links["successors"]})
                if linked:
                    line += "; связи с " + ", ".join(f"#{i}" for i in linked) + " будут удалены"
            lines.append(line)
        elif op.tool == "add_dependency":
            before = _task_name(connection, args.get("predecessor_task_id"), titles)
            after = _task_name(connection, args.get("successor_task_id"), titles)
            lines.append(f"{after} — выполнять после {before}")
        elif op.tool == "remove_dependency":
            before = _task_name(connection, args.get("predecessor_task_id"), titles)
            after = _task_name(connection, args.get("successor_task_id"), titles)
            lines.append(f"Убрать зависимость {after} от {before}")
        elif op.tool == "update_preference":
            for kind, value in args.items():
                shown = ", ".join(value) if isinstance(value, list) else value
                lines.append(f"Запомнить: {_PREFERENCE_TEXT.get(kind, kind)} — {shown}")
    return lines


# ---------------------------------------------------------------- ожидающее действие


class PendingPlan(BaseModel):
    action_id: str
    preview: list[str]
    reasons: list[str]
    expires_at: str


def _affected_task_ids(plan: Plan) -> list[int]:
    ids = {
        value
        for op in plan.operations
        for key, value in op.args.items()
        if key in ("task_id", "predecessor_task_id", "successor_task_id")
        and isinstance(value, int)
        and not isinstance(value, bool)
    }
    return sorted(ids)


def prepare_plan_action(connection: sqlite3.Connection, plan: Plan, request: str) -> PendingPlan:
    """Сохраняет план как ожидающее действие (commit делает вызывающий код)."""
    if not plan.writes:
        raise ValueError("План ничего не меняет — подтверждать нечего")
    affected = _affected_task_ids(plan)
    if not affected:
        raise ValueError("Подтверждение нужно только для действий с существующими задачами")
    preview = describe_plan(connection, plan)
    reasons = confirmation_reasons(connection, plan)
    action_id = crud_func.create_pending_operation(
        connection,
        ACTION_TYPE,
        {"plan": plan.model_dump(), "request": request, "preview": preview},
        affected,
    )
    operation = crud_func.get_operation(connection, action_id)
    return PendingPlan(
        action_id=action_id,
        preview=preview,
        reasons=reasons,
        expires_at=operation["expires_at"],
    )


def confirm_plan_action(
    action_id: str, db_path: str | Path | None = None
) -> ExecutionResult | None:
    """Выполняет подтверждённый план. None — действие уже было подтверждено раньше.

    Перед выполнением проверяется: действие ждёт подтверждения, срок не истёк, затронутые задачи
    не менялись после показа плана (A07), и план всё ещё проходит проверки кода валидатора.
    """
    connection = connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        operation = crud_func.get_operation(connection, action_id)
        if operation is not None and operation["status"] == "confirmed":
            if operation["type"] != ACTION_TYPE:
                raise ValueError("Тип операции не совпадает")
            connection.rollback()
            return None

        payload, _ = crud_func.validate_pending_operation(connection, action_id, ACTION_TYPE)
        plan = Plan.model_validate(payload["plan"])

        issues = Validator(system_prompt="-").hard_checks(plan, None, connection)
        if issues:
            raise ValueError("План больше не корректен: " + "; ".join(d for _, d in issues))

        result = execute_plan(connection, plan, confirmed=True)

        cursor = connection.execute(
            """
            UPDATE operations
            SET status = 'confirmed', result_json = ?, confirmed_at = ?
            WHERE action_id = ? AND status = 'pending'
            """,
            (
                json.dumps(result.model_dump(exclude={"tasks"}), ensure_ascii=False),
                datetime.now(UTC).isoformat(),
                action_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Не удалось подтвердить операцию")
        connection.commit()
        return result
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


# ---------------------------------------------------------------- ответы пользователю


def _tasks_reply(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        return "Задач пока нет."
    lines = ["Ваши задачи:"]
    for task in tasks:
        details = [_LIST_STATUS.get(task["status"], task["status"])]
        if task.get("priority"):
            details.append(f"приоритет {_PRIORITY_TEXT[task['priority']]}")
        if task.get("due_date"):
            details.append(f"срок {_deadline_text(task)}")
        lines.append(f"#{task['id']} «{task['title']}» — {', '.join(details)}")
    return "\n".join(lines)


def _order_reply(connection: sqlite3.Connection, plan: Plan) -> str:
    reasons = {r.task_id: r.reason for r in plan.recommendations}
    lines = [plan.message_to_user] if plan.message_to_user else []
    for index, task_id in enumerate(plan.ordered_task_ids, start=1):
        name = _task_name(connection, task_id, {})
        reason = f" — {reasons[task_id]}" if task_id in reasons else ""
        lines.append(f"{index}. {name}{reason}")
    return "\n".join(lines)


def pending_reply(plan: Plan, pending: PendingPlan) -> str:
    lines = [plan.message_to_user or "Нужно подтверждение.", "", "Что изменится:"]
    lines += [f"- {line}" for line in pending.preview]
    lines += ["", "Задачи пока не изменены. Подтвердить или отменить?"]
    return "\n".join(lines)


# ---------------------------------------------------------------- узел графа


def make_action_node(
    *, connect: Callable[[], sqlite3.Connection]
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Узел LangGraph после валидатора: выполняет принятый план или ставит его на подтверждение.

    Пишет в State `reply` и `pending_action_id`. Отклонённые планы не трогает — повторы и ответ
    при отказе решает workflow.
    """

    def action_node(state: dict[str, Any]) -> dict[str, Any]:
        verdict = state.get("verdict") or {}
        if verdict.get("decision") != "accept" or not state.get("plan"):
            return {}
        plan = Plan.model_validate(state["plan"])

        if plan.needs_clarification:
            return {"reply": plan.message_to_user, "pending_action_id": None}

        with closing(connect()) as connection:
            if plan.intent == "plan_tasks":
                return {"reply": _order_reply(connection, plan), "pending_action_id": None}

            if plan.writes and (
                plan.requires_confirmation or confirmation_reasons(connection, plan)
            ):
                connection.execute("BEGIN IMMEDIATE")
                try:
                    pending = prepare_plan_action(connection, plan, state["request"])
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
                return {
                    "reply": pending_reply(plan, pending),
                    "pending_action_id": pending.action_id,
                }

            result = run_plan(connection, plan)
        if result.tasks is not None:
            return {"reply": _tasks_reply(result.tasks), "pending_action_id": None}
        return {"reply": result.summary(), "pending_action_id": None}

    return action_node
