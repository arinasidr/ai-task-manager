import json
from contextlib import closing
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from ai_task_manager.config import Settings
from ai_task_manager.db.crud_func import (
    add_dependency,
    cancel_operation,
    change_task_status,
    create_task,
    get_operation,
)
from ai_task_manager.db.database import connect, init_db
from ai_task_manager.db.preferences import update_preference
from ai_task_manager.tools.confirmation import confirm_plan_action
from ai_task_manager.workflow.service import create_workflow, run_chat


def load_cases() -> list[dict]:
    path = Path(__file__).with_name("cases.jsonl")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def seed_case(db_path: Path, case: dict) -> dict[str, int]:
    """Создать исходные задачи сценария; вернуть их ID по названиям."""
    init_db(db_path)
    task_ids: dict[str, int] = {}
    initial = case["initial_db"]

    with closing(connect(db_path)) as connection:
        with connection:
            for task in initial["tasks"]:
                title = task["title"]
                if title in task_ids:
                    raise ValueError(f"Повторяющееся название: {title}")

                task_id = create_task(
                    connection,
                    title,
                    priority=task.get("priority"),
                    due_date=task.get("due_date"),
                    due_time=task.get("due_time"),
                    due_timezone=task.get("due_timezone"),
                    due_at_utc=task.get("due_at_utc"),
                )
                task_ids[title] = task_id

                if task.get("status", "todo") != "todo":
                    change_task_status(connection, task_id, task["status"])

            for link in initial["dependencies"]:
                add_dependency(
                    connection,
                    task_ids[link["depends_on"]],
                    task_ids[link["task"]],
                )

            for preference in initial.get("preferences", []):
                if preference["status"] != "confirmed":
                    raise ValueError("Пока поддерживаются только подтверждённые предпочтения")
                update_preference(
                    connection,
                    preference["kind"],
                    preference["value"],
                )

    return task_ids


def snapshot_db(db_path: Path) -> dict:
    """Состояние задач, связей и предпочтений для сравнения."""
    with closing(connect(db_path)) as connection:
        return {
            "tasks": [dict(row) for row in connection.execute("SELECT * FROM tasks ORDER BY id")],
            "dependencies": [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM task_dependencies "
                    "ORDER BY predecessor_task_id, successor_task_id"
                )
            ],
            "preferences": [
                dict(row) for row in connection.execute("SELECT * FROM preferences ORDER BY id")
            ],
        }


def run_case(case: dict) -> dict:
    """Запустить запрос и, если задано, подтвердить или отменить действие."""
    with TemporaryDirectory() as folder:
        db_path = Path(folder) / "tasks.db"
        seed_ids = seed_case(db_path, case)
        before = snapshot_db(db_path)

        settings = Settings(database_path=str(db_path), timezone=case["timezone"])
        started = perf_counter()
        result = run_chat(
            create_workflow(settings),
            settings,
            case["message"],
            now=datetime.fromisoformat(case["now"]),
        )
        duration = perf_counter() - started

        # Проверяем, что задача не исчезла до ответа пользователя.
        after_request = snapshot_db(db_path)

        follow_up = case.get("follow_up")
        if follow_up not in (None, "confirm", "cancel"):
            raise ValueError(f"Неизвестный follow_up: {follow_up}")

        follow_up_result = None
        if follow_up is not None:
            if result.action_id is None:
                follow_up_result = {"status": None, "applied": False}
            else:
                if follow_up == "confirm":
                    applied = confirm_plan_action(result.action_id, db_path) is not None
                else:
                    with closing(connect(db_path)) as connection:
                        with connection:
                            applied = cancel_operation(connection, result.action_id)

                with closing(connect(db_path)) as connection:
                    operation = get_operation(connection, result.action_id)

                follow_up_result = {
                    "status": operation["status"] if operation else None,
                    "applied": applied,
                }

        after = snapshot_db(db_path)

    return {
        "case_id": case["id"],
        "seed_ids": seed_ids,
        "before": before,
        "after_request": after_request,
        "after": after,
        "follow_up_result": follow_up_result,
        "result": result.model_dump(mode="json"),
        "duration_seconds": duration,
    }
