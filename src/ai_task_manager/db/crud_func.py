import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta

_UPDATABLE_FIELDS = {
    "title",
    "description",
    "priority",
    "due_date",
    "due_time",
    "due_timezone",
    "due_at_utc",
}


def get_task(connection: sqlite3.Connection, task_id: int) -> dict | None:
    """Возвращает задачу по ID или None, если такой задачи нет."""
    row = connection.execute(
        "SELECT * FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def create_task(
    connection: sqlite3.Connection,
    title: str,
    *,
    description: str | None = None,
    priority: str | None = None,
    due_date: str | None = None,
    due_time: str | None = None,
    due_timezone: str | None = None,
    due_at_utc: str | None = None,
) -> int:
    """Создаёт задачу и возвращает её ID без фиксации транзакции."""
    if not title.strip():
        raise ValueError("Название задачи не может быть пустым")
    if priority not in (None, "low", "medium", "high"):
        raise ValueError("Недопустимый приоритет")

    now = datetime.now(UTC).isoformat()
    cursor = connection.execute(
        """
        INSERT INTO tasks (
            title, description, priority,
            due_date, due_time, due_timezone, due_at_utc,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            title.strip(),
            description,
            priority,
            due_date,
            due_time,
            due_timezone,
            due_at_utc,
            now,
            now,
        ),
    )
    return int(cursor.lastrowid)


def list_tasks(connection: sqlite3.Connection) -> list[dict]:
    """Возвращает все задачи в порядке их ID."""
    rows = connection.execute("SELECT * FROM tasks ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def update_task(
    connection: sqlite3.Connection,
    task_id: int,
    updates: dict[str, str | None],
) -> bool:
    """Обновляет разрешённые поля задачи и повышает её версию."""
    if task_id <= 0:
        raise ValueError("Нужен положительный ID задачи")
    if not updates:
        raise ValueError("Не указано, что изменить")
    if set(updates) - _UPDATABLE_FIELDS:
        raise ValueError("Недопустимое поле для изменения")

    changes = updates.copy()
    if "title" in changes:
        title = changes["title"]
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Название задачи не может быть пустым и должно быть строкой")
        changes["title"] = title.strip()

    if "priority" in changes and changes["priority"] not in (None, "low", "medium", "high"):
        raise ValueError("Недопустимый приоритет")

    assignments = ", ".join(f"{field} = ?" for field in changes)
    cursor = connection.execute(
        f"UPDATE tasks SET {assignments}, updated_at = ?, version = version + 1 WHERE id = ?",
        [*changes.values(), datetime.now(UTC).isoformat(), task_id],
    )
    return cursor.rowcount == 1


def change_task_status(
    connection: sqlite3.Connection,
    task_id: int,
    status: str,
) -> bool:
    """Меняет статус, запрещая завершение при незакрытых зависимостях."""
    if task_id <= 0:
        raise ValueError("Нужен положительный ID задачи")
    if status not in ("todo", "in_progress", "done"):
        raise ValueError("Недопустимый статус")

    if status == "done":
        ids = get_unfinished_predecessors(connection, task_id)
        if ids:
            raise ValueError(f"Есть незавершённые зависимости {ids}; требуется подтверждение")

    cursor = connection.execute(
        """
        UPDATE tasks
        SET status = ?, updated_at = ?, version = version + 1
        WHERE id = ?
        """,
        (status, datetime.now(UTC).isoformat(), task_id),
    )
    return cursor.rowcount == 1


def delete_task(connection: sqlite3.Connection, task_id: int) -> bool:
    """Удаляет задачу; подтверждение проверяет вызывающий сервис."""
    if task_id <= 0:
        raise ValueError("Нужен положительный ID задачи")

    cursor = connection.execute(
        "DELETE FROM tasks WHERE id = ?",
        (task_id,),
    )
    return cursor.rowcount == 1


def get_dependencies(
    connection: sqlite3.Connection,
    task_id: int,
) -> dict[str, list[int]]:
    """Возвращает ID предшественников и последователей задачи."""
    if get_task(connection, task_id) is None:
        raise ValueError(f"Задача {task_id} не найдена")

    predecessors = connection.execute(
        """
        SELECT predecessor_task_id
        FROM task_dependencies
        WHERE successor_task_id = ?
        ORDER BY predecessor_task_id
        """,
        (task_id,),
    ).fetchall()

    successors = connection.execute(
        """
        SELECT successor_task_id
        FROM task_dependencies
        WHERE predecessor_task_id = ?
        ORDER BY successor_task_id
        """,
        (task_id,),
    ).fetchall()

    return {
        "predecessors": [row["predecessor_task_id"] for row in predecessors],
        "successors": [row["successor_task_id"] for row in successors],
    }


def add_dependency(
    connection: sqlite3.Connection,
    predecessor_task_id: int,
    successor_task_id: int,
) -> bool:
    """Добавляет связь между задачами, отклоняя отсутствующие ID и циклы."""
    if predecessor_task_id == successor_task_id:
        raise ValueError("Задача не может зависеть от самой себя")

    for task_id in (predecessor_task_id, successor_task_id):
        if get_task(connection, task_id) is None:
            raise ValueError(f"Задача {task_id} не найдена")

    existing = connection.execute(
        """
        SELECT 1 FROM task_dependencies
        WHERE predecessor_task_id = ? AND successor_task_id = ?
        """,
        (predecessor_task_id, successor_task_id),
    ).fetchone()
    if existing:
        return False

    cycle = connection.execute(
        """
        WITH RECURSIVE reachable(task_id) AS (
            SELECT successor_task_id
            FROM task_dependencies
            WHERE predecessor_task_id = ?

            UNION

            SELECT d.successor_task_id
            FROM task_dependencies AS d
            JOIN reachable AS r ON d.predecessor_task_id = r.task_id
        )
        SELECT 1 FROM reachable WHERE task_id = ? LIMIT 1
        """,
        (successor_task_id, predecessor_task_id),
    ).fetchone()
    if cycle:
        raise ValueError("Эта зависимость создаст цикл")

    connection.execute(
        """
        INSERT INTO task_dependencies (predecessor_task_id, successor_task_id)
        VALUES (?, ?)
        """,
        (predecessor_task_id, successor_task_id),
    )
    _bump_task_versions(connection, (predecessor_task_id, successor_task_id))
    return True


def remove_dependency(
    connection: sqlite3.Connection,
    predecessor_task_id: int,
    successor_task_id: int,
) -> bool:
    """Удаляет связь между задачами и обновляет их версии."""
    if predecessor_task_id <= 0 or successor_task_id <= 0:
        raise ValueError("Нужны положительные ID задач")

    cursor = connection.execute(
        """
        DELETE FROM task_dependencies
        WHERE predecessor_task_id = ? AND successor_task_id = ?
        """,
        (predecessor_task_id, successor_task_id),
    )
    if cursor.rowcount == 0:
        return False

    _bump_task_versions(connection, (predecessor_task_id, successor_task_id))
    return True


def create_pending_operation(
    connection: sqlite3.Connection,
    operation_type: str,
    payload: dict,
    affected_task_ids: list[int],
) -> str:
    """Сохраняет ожидающее действие и версии задач на десять минут."""
    if not operation_type.strip():
        raise ValueError("Не указан тип операции")
    if not affected_task_ids:
        raise ValueError("Не указаны затрагиваемые задачи")

    expected_versions = {}
    for task_id in affected_task_ids:
        task = get_task(connection, task_id)
        if task is None:
            raise ValueError(f"Задача {task_id} не найдена")
        expected_versions[str(task_id)] = task["version"]

    prepared_payload = dict(payload)
    prepared_payload["_expected_versions"] = expected_versions

    action_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    connection.execute(
        """
        INSERT INTO operations (
            action_id, type, payload_json, affected_task_ids_json,
            status, created_at, expires_at
        ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
        """,
        (
            action_id,
            operation_type,
            json.dumps(prepared_payload, ensure_ascii=False),
            json.dumps(affected_task_ids),
            now.isoformat(),
            (now + timedelta(minutes=10)).isoformat(),
        ),
    )
    return action_id


def get_operation(
    connection: sqlite3.Connection,
    action_id: str,
) -> dict | None:
    """Возвращает запись операции по action_id или None."""
    if not action_id.strip():
        raise ValueError("Не указан action_id")

    row = connection.execute(
        "SELECT * FROM operations WHERE action_id = ?",
        (action_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def cancel_operation(
    connection: sqlite3.Connection,
    action_id: str,
) -> bool:
    """Отменяет только ожидающую операцию, не меняя задачи."""
    if not action_id.strip():
        raise ValueError("Не указан action_id")

    cursor = connection.execute(
        """
        UPDATE operations
        SET status = 'cancelled'
        WHERE action_id = ? AND status = 'pending'
        """,
        (action_id,),
    )
    return cursor.rowcount == 1


def _bump_task_versions(
    connection: sqlite3.Connection,
    task_ids: tuple[int, int],
) -> None:
    """Повышает версии после смены связей, делая старые подтверждения недействительными."""
    now = datetime.now(UTC).isoformat()
    connection.executemany(
        """
        UPDATE tasks
        SET version = version + 1, updated_at = ?
        WHERE id = ?
        """,
        [(now, task_id) for task_id in set(task_ids)],
    )


def validate_pending_operation(
    connection: sqlite3.Connection,
    action_id: str,
    expected_type: str,
) -> tuple[dict, list[int]]:
    """Проверяет тип, срок и версии задач перед применением операции."""
    operation = get_operation(connection, action_id)
    if operation is None:
        raise ValueError("Операция не найдена")
    if operation["type"] != expected_type:
        raise ValueError("Тип операции не совпадает")
    if operation["status"] != "pending":
        raise ValueError("Операция уже обработана")

    expires_at = operation["expires_at"]
    if expires_at is None or datetime.fromisoformat(expires_at) <= datetime.now(UTC):
        raise ValueError("Время подтверждения истекло")

    payload = json.loads(operation["payload_json"])
    task_ids = json.loads(operation["affected_task_ids_json"])
    expected_versions = payload.get("_expected_versions")

    if not isinstance(expected_versions, dict):
        raise ValueError("Не сохранены версии задач")

    for task_id in task_ids:
        task = get_task(connection, task_id)
        if task is None or task["version"] != expected_versions.get(str(task_id)):
            raise ValueError(f"Задача {task_id} изменилась; нужно запросить подтверждение заново")

    return payload, task_ids


def get_unfinished_predecessors(
    connection: sqlite3.Connection,
    task_id: int,
) -> list[int]:
    """Возвращает ID незавершённых предшественников задачи."""
    rows = connection.execute(
        """
        SELECT t.id
        FROM task_dependencies AS d
        JOIN tasks AS t ON t.id = d.predecessor_task_id
        WHERE d.successor_task_id = ? AND t.status != 'done'
        ORDER BY t.id
        """,
        (task_id,),
    ).fetchall()
    return [row["id"] for row in rows]
