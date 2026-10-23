import json
from datetime import UTC, datetime
from pathlib import Path

from ai_task_manager.db.crud_func import (
    create_pending_operation,
    delete_task,
    get_operation,
    get_task,
    get_unfinished_predecessors,
    validate_pending_operation,
)
from ai_task_manager.db.database import connect


def confirm_delete_task(
    action_id: str,
    db_path: str | Path | None = None,
) -> bool:
    """Удаляет задачу и подтверждает операцию в одной транзакции.

    Вызывается после согласия пользователя; повторное подтверждение возвращает False.
    """
    connection = connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")

        operation = get_operation(connection, action_id)
        if operation is not None and operation["status"] == "confirmed":
            connection.rollback()
            return False

        payload, task_ids = validate_pending_operation(connection, action_id, "delete_task")
        if len(task_ids) != 1 or payload.get("task_id") != task_ids[0]:
            raise ValueError("Операция удаления содержит неверный ID задачи")

        task_id = task_ids[0]
        if not delete_task(connection, task_id):
            raise ValueError(f"Задача {task_id} не найдена")

        cursor = connection.execute(
            """
            UPDATE operations
            SET status = 'confirmed', result_json = ?, confirmed_at = ?
            WHERE action_id = ? AND status = 'pending'
            """,
            (
                json.dumps({"deleted_task_id": task_id}),
                datetime.now(UTC).isoformat(),
                action_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Не удалось подтвердить операцию")

        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def prepare_blocked_completion(
    task_id: int,
    db_path: str | Path | None = None,
) -> dict:
    """Готовит подтверждение завершения, сохраняя список незакрытых зависимостей."""
    connection = connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")

        task = get_task(connection, task_id)
        if task is None:
            raise ValueError(f"Задача {task_id} не найдена")
        if task["status"] == "done":
            raise ValueError("Задача уже завершена")

        blockers = get_unfinished_predecessors(connection, task_id)
        if not blockers:
            raise ValueError("Подтверждение не требуется")

        action_id = create_pending_operation(
            connection,
            "complete_task_with_blockers",
            {"task_id": task_id, "unfinished_predecessor_ids": blockers},
            [task_id],
        )
        connection.commit()
        return {
            "action_id": action_id,
            "task_id": task_id,
            "unfinished_predecessor_ids": blockers,
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def confirm_blocked_completion(
    action_id: str,
    db_path: str | Path | None = None,
) -> bool:
    """Проверяет подтверждение и атомарно завершает задачу вопреки зависимостям."""
    connection = connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")

        operation = get_operation(connection, action_id)
        if operation is not None and operation["status"] == "confirmed":
            if operation["type"] != "complete_task_with_blockers":
                raise ValueError("Тип операции не совпадает")
            connection.rollback()
            return False

        payload, task_ids = validate_pending_operation(
            connection, action_id, "complete_task_with_blockers"
        )
        if len(task_ids) != 1 or payload.get("task_id") != task_ids[0]:
            raise ValueError("Операция содержит неверный ID задачи")

        expected_blockers = payload.get("unfinished_predecessor_ids")
        if not isinstance(expected_blockers, list) or not expected_blockers:
            raise ValueError("Не сохранён список незавершённых зависимостей")

        task_id = task_ids[0]
        current_blockers = get_unfinished_predecessors(connection, task_id)
        if current_blockers != expected_blockers:
            raise ValueError("Зависимости изменились; нужно запросить подтверждение заново")

        now = datetime.now(UTC).isoformat()
        cursor = connection.execute(
            """
            UPDATE tasks
            SET status = 'done', updated_at = ?, version = version + 1
            WHERE id = ? AND status != 'done'
            """,
            (now, task_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Задача {task_id} уже завершена или не найдена")

        cursor = connection.execute(
            """
            UPDATE operations
            SET status = 'confirmed', result_json = ?, confirmed_at = ?
            WHERE action_id = ? AND status = 'pending'
            """,
            (
                json.dumps({"completed_task_id": task_id}),
                now,
                action_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Не удалось подтвердить операцию")

        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
