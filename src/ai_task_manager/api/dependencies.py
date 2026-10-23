"""Зависимости маршрутов FastAPI."""

import sqlite3
from collections.abc import Iterator

from ai_task_manager.db.database import connect


def get_db() -> Iterator[sqlite3.Connection]:
    """Открывает соединение на время запроса и закрывает его после."""
    connection = connect()
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()
