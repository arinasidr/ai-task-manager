"""Защита POST /chat от повторного выполнения после потери ответа клиентом."""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime


class RequestConflict(ValueError):
    """Ключ запроса уже занят другим сообщением или запрос ещё выполняется."""


def _fingerprint(message: str, conversation_id: str | None) -> str:
    content = json.dumps(
        {"message": message, "conversation_id": conversation_id},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def claim_request(
    db: sqlite3.Connection,
    request_id: str,
    message: str,
    conversation_id: str | None,
) -> dict | None:
    """Резервирует ключ; для завершённого повтора возвращает сохранённый ответ."""
    request_hash = _fingerprint(message, conversation_id)
    try:
        db.execute(
            """
            INSERT INTO chat_requests (request_id, request_hash, status, created_at)
            VALUES (?, ?, 'processing', ?)
            """,
            (request_id, request_hash, datetime.now(UTC).isoformat()),
        )
        db.commit()
        return None
    except sqlite3.IntegrityError:
        db.rollback()

    row = db.execute(
        "SELECT request_hash, status, response_json FROM chat_requests WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    if row is None:
        raise RequestConflict("Не удалось найти сохранённый запрос")
    if row["request_hash"] != request_hash:
        raise RequestConflict("Этот request_id уже использован для другого сообщения")
    if row["status"] != "completed":
        raise RequestConflict(
            "Запрос ещё выполняется или его результат неизвестен. "
            "Повторите позже с тем же request_id; не создавайте новый запрос."
        )
    return json.loads(row["response_json"])


def complete_request(db: sqlite3.Connection, request_id: str, response: dict) -> None:
    """Сохраняет ответ для безопасных повторов с тем же ключом."""
    cursor = db.execute(
        """
        UPDATE chat_requests
        SET status = 'completed', response_json = ?
        WHERE request_id = ? AND status = 'processing'
        """,
        (json.dumps(response, ensure_ascii=False), request_id),
    )
    if cursor.rowcount != 1:
        db.rollback()
        raise RuntimeError("Не удалось сохранить результат запроса")
    db.commit()
