import sqlite3
from pathlib import Path

from ai_task_manager.config import get_settings


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Открывает SQLite с доступом к полям по имени и проверкой внешних ключей."""
    path = Path(db_path) if db_path is not None else Path(get_settings().database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(db_path: str | Path | None = None) -> None:
    """Создаёт таблицы задач, зависимостей и операций, если их ещё нет."""
    connection = connect(db_path)
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL CHECK (length(trim(title)) > 0),
                description TEXT,
                status TEXT NOT NULL DEFAULT 'todo'
                    CHECK (status IN ('todo', 'in_progress', 'done')),
                priority TEXT
                    CHECK (priority IN ('low', 'medium', 'high') OR priority IS NULL),
                due_date TEXT,
                due_time TEXT,
                due_timezone TEXT,
                due_at_utc TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS task_dependencies (
                predecessor_task_id INTEGER NOT NULL,
                successor_task_id INTEGER NOT NULL,

                PRIMARY KEY (predecessor_task_id, successor_task_id),
                CHECK (predecessor_task_id <> successor_task_id),

                FOREIGN KEY (predecessor_task_id)
                    REFERENCES tasks(id) ON DELETE CASCADE,
                FOREIGN KEY (successor_task_id)
                    REFERENCES tasks(id) ON DELETE CASCADE
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_dependencies_successor
            ON task_dependencies(successor_task_id)
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS operations (
                action_id TEXT PRIMARY KEY NOT NULL,
                type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                affected_task_ids_json TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK (status IN ('pending', 'confirmed', 'cancelled', 'expired')),
                result_json TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                confirmed_at TEXT
            )
        """)
        connection.commit()
    finally:
        connection.close()
