import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from ai_task_manager.memory.models import Preference, PreferenceInput, PreferenceKind


@contextmanager
def _atomic_write(connection: sqlite3.Connection) -> Iterator[None]:
    """Откатывает только текущую операцию, оставляя commit вызывающему коду."""
    started_transaction = not connection.in_transaction
    changes_before = connection.total_changes
    if started_transaction:
        connection.execute("BEGIN IMMEDIATE")

    try:
        connection.execute("SAVEPOINT preference_write")
        try:
            if not started_transaction:
                # Резервируем запись до чтения даже внутри чужой DEFERRED-транзакции.
                connection.execute("UPDATE preferences SET id = id WHERE 0")
            yield
        except BaseException:
            connection.execute("ROLLBACK TO SAVEPOINT preference_write")
            raise
        finally:
            connection.execute("RELEASE SAVEPOINT preference_write")
    except BaseException:
        if started_transaction:
            connection.rollback()
        raise
    else:
        if started_transaction and connection.total_changes == changes_before:
            # Читающий или повторный вызов не должен удерживать блокировку записи.
            connection.rollback()


def _value_key(value: str | list[str]) -> tuple[str, ...]:
    """Сравнивает значения без учёта регистра, лишних пробелов и порядка списка."""
    items = [value] if isinstance(value, str) else value
    return tuple(sorted({" ".join(item.casefold().split()) for item in items}))


def _from_row(row: sqlite3.Row) -> Preference:
    data = dict(row)
    data["value"] = json.loads(data.pop("value_json"))
    return Preference.model_validate(data)


def _matching_rows(
    connection: sqlite3.Connection, kind: PreferenceKind, value: str | list[str]
) -> list[Preference]:
    key = _value_key(value)
    rows = connection.execute(
        "SELECT * FROM preferences WHERE kind = ? ORDER BY id DESC", (kind,)
    ).fetchall()
    matches: list[Preference] = []
    for row in rows:
        preference = _from_row(row)
        if _value_key(preference.value) == key:
            matches.append(preference)
    return matches


def get_preferences(
    connection: sqlite3.Connection,
) -> dict[PreferenceKind, str | list[str]]:
    """Возвращает только подтверждённые предпочтения для планировщика."""
    rows = connection.execute(
        "SELECT * FROM preferences WHERE status = 'confirmed' ORDER BY id"
    ).fetchall()

    result = {}
    for row in rows:
        preference = _from_row(row)
        result[preference.kind] = preference.value
    return result


def update_preference(
    connection: sqlite3.Connection,
    kind: PreferenceKind,
    value: str | list[str],
) -> Preference:
    """Записывает явное предпочтение; commit делает вызывающий код."""
    data = PreferenceInput(kind=kind, value=value)
    now = datetime.now(UTC).isoformat()

    with _atomic_write(connection):
        connection.execute(
            """
            UPDATE preferences
            SET status = 'superseded', updated_at = ?
            WHERE kind = ? AND status IN ('confirmed', 'proposed')
            """,
            (now, data.kind),
        )
        cursor = connection.execute(
            """
            INSERT INTO preferences (
                kind, value_json, status, source, created_at, updated_at
            ) VALUES (?, ?, 'confirmed', 'explicit', ?, ?)
            """,
            (data.kind, json.dumps(data.value, ensure_ascii=False), now, now),
        )
        row = connection.execute(
            "SELECT * FROM preferences WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        if row is None:
            raise RuntimeError("Сохранённое предпочтение не найдено")
        return _from_row(row)


def propose_preference(
    connection: sqlite3.Connection,
    kind: PreferenceKind,
    value: str | list[str],
) -> Preference:
    """Сохраняет предположение без применения к планированию."""
    data = PreferenceInput(kind=kind, value=value)
    value_json = json.dumps(data.value, ensure_ascii=False)
    with _atomic_write(connection):
        matches = _matching_rows(connection, data.kind, data.value)
        if any(item.status == "confirmed" for item in matches):
            raise ValueError("Это предпочтение уже действует")
        if any(item.status == "rejected" for item in matches):
            raise ValueError("Пользователь уже отклонил это предпочтение")
        existing = next((item for item in matches if item.status == "proposed"), None)
        if existing is not None:
            return existing

        now = datetime.now(UTC).isoformat()
        cursor = connection.execute(
            """
            INSERT INTO preferences (
                kind, value_json, status, source, created_at, updated_at
            ) VALUES (?, ?, 'proposed', 'inferred', ?, ?)
            """,
            (data.kind, value_json, now, now),
        )
        row = connection.execute(
            "SELECT * FROM preferences WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        if row is None:
            raise RuntimeError("Предложенное предпочтение не найдено")
        return _from_row(row)


def confirm_preference(
    connection: sqlite3.Connection,
    preference_id: int,
) -> Preference:
    """Подтверждает предложение и заменяет прежнее действующее значение."""
    with _atomic_write(connection):
        row = connection.execute(
            "SELECT * FROM preferences WHERE id = ?", (preference_id,)
        ).fetchone()
        if row is None or row["status"] != "proposed":
            raise ValueError("Предложение не найдено или уже обработано")

        now = datetime.now(UTC).isoformat()
        kind = row["kind"]
        connection.execute(
            """
            UPDATE preferences
            SET status = 'superseded', updated_at = ?
            WHERE kind = ? AND status IN ('confirmed', 'proposed') AND id != ?
            """,
            (now, kind, preference_id),
        )
        cursor = connection.execute(
            """
            UPDATE preferences
            SET status = 'confirmed', updated_at = ?
            WHERE id = ? AND status = 'proposed'
            """,
            (now, preference_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("Предложение уже обработано")

        confirmed = connection.execute(
            "SELECT * FROM preferences WHERE id = ?", (preference_id,)
        ).fetchone()
        if confirmed is None:
            raise RuntimeError("Подтверждённое предпочтение не найдено")
        return _from_row(confirmed)


def reject_preference(connection: sqlite3.Connection, preference_id: int) -> bool:
    """Отклоняет ещё не подтверждённое предложение."""
    with _atomic_write(connection):
        row = connection.execute(
            "SELECT * FROM preferences WHERE id = ?", (preference_id,)
        ).fetchone()
        if row is None or row["status"] != "proposed":
            return False

        target = _from_row(row)
        now = datetime.now(UTC).isoformat()
        for proposal in _matching_rows(connection, target.kind, target.value):
            if proposal.status == "proposed":
                connection.execute(
                    """
                    UPDATE preferences SET status = 'rejected', updated_at = ?
                    WHERE id = ? AND status = 'proposed'
                    """,
                    (now, proposal.id),
                )
        return True


def delete_preference(connection: sqlite3.Connection, kind: PreferenceKind) -> bool:
    """Удаляет предпочтение этого типа вместе с его историей и предложениями."""
    with _atomic_write(connection):
        cursor = connection.execute("DELETE FROM preferences WHERE kind = ?", (kind,))
        return cursor.rowcount > 0
