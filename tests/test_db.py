import json
from contextlib import closing
from pathlib import Path

import pytest

from ai_task_manager.db.crud_func import (
    add_dependency,
    cancel_operation,
    change_task_status,
    create_pending_operation,
    create_task,
    get_dependencies,
    get_operation,
    get_task,
    list_tasks,
    remove_dependency,
    update_task,
)
from ai_task_manager.db.database import connect, init_db
from ai_task_manager.db.operation_service import (
    confirm_blocked_completion,
    confirm_delete_task,
    prepare_blocked_completion,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "tasks.db"
    init_db(path)
    return path


def test_task_crud_keeps_optional_fields_empty_and_updates_only_selected_id(
    db_path: Path,
) -> None:
    with closing(connect(db_path)) as connection:
        first = create_task(connection, "  Отчёт  ")
        second = create_task(connection, "Отчёт")

        assert first != second
        task = get_task(connection, first)
        assert task is not None
        assert task["title"] == "Отчёт"
        assert task["status"] == "todo"
        assert task["priority"] is None
        assert task["due_date"] is None

        assert update_task(connection, first, {"priority": "high"})
        assert change_task_status(connection, first, "in_progress")
        assert not update_task(connection, 9999, {"title": "Нет такой задачи"})
        with pytest.raises(ValueError, match="Недопустимое поле"):
            update_task(connection, first, {"status": "done"})
        connection.commit()

    with closing(connect(db_path)) as connection:
        tasks = list_tasks(connection)
        assert [task["id"] for task in tasks] == [first, second]
        assert tasks[0]["priority"] == "high"
        assert tasks[0]["status"] == "in_progress"
        assert tasks[0]["version"] == 3
        assert tasks[1]["priority"] is None
        assert tasks[1]["status"] == "todo"


def test_dependencies_reject_cycles_and_block_premature_completion(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        first, second, third = (create_task(connection, title) for title in ("A", "B", "C"))
        assert add_dependency(connection, first, second)
        assert add_dependency(connection, second, third)
        assert not add_dependency(connection, first, second)
        assert get_dependencies(connection, second) == {
            "predecessors": [first],
            "successors": [third],
        }

        with pytest.raises(ValueError, match="цикл"):
            add_dependency(connection, third, first)
        with pytest.raises(ValueError, match="не найдена"):
            add_dependency(connection, 9999, first)
        with pytest.raises(ValueError, match="незавершённые зависимости"):
            change_task_status(connection, second, "done")
        assert get_task(connection, second)["status"] == "todo"

        assert change_task_status(connection, first, "done")
        assert change_task_status(connection, second, "done")
        assert remove_dependency(connection, second, third)
        assert not remove_dependency(connection, second, third)
        assert get_dependencies(connection, third)["predecessors"] == []


def test_confirmed_delete_removes_only_task_and_its_links(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        first, second, third = (create_task(connection, title) for title in ("A", "B", "C"))
        add_dependency(connection, first, second)
        add_dependency(connection, second, third)
        action_id = create_pending_operation(connection, "delete_task", {"task_id": first}, [first])
        assert get_task(connection, first) is not None
        assert get_operation(connection, action_id)["status"] == "pending"
        connection.commit()

    assert confirm_delete_task(action_id, db_path)
    assert not confirm_delete_task(action_id, db_path)

    with closing(connect(db_path)) as connection:
        assert get_task(connection, first) is None
        assert get_task(connection, second) is not None
        assert get_task(connection, third) is not None
        assert get_dependencies(connection, second) == {
            "predecessors": [],
            "successors": [third],
        }
        assert get_dependencies(connection, third)["predecessors"] == [second]
        operation = get_operation(connection, action_id)
        assert operation["status"] == "confirmed"
        assert json.loads(operation["result_json"]) == {"deleted_task_id": first}


def test_stale_cancelled_and_expired_confirmations_leave_tasks_untouched(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        stale = create_task(connection, "Изменится")
        stale_action = create_pending_operation(
            connection, "delete_task", {"task_id": stale}, [stale]
        )
        update_task(connection, stale, {"title": "Уже изменена"})

        cancelled = create_task(connection, "Останется")
        cancelled_action = create_pending_operation(
            connection, "delete_task", {"task_id": cancelled}, [cancelled]
        )
        assert cancel_operation(connection, cancelled_action)
        assert not cancel_operation(connection, cancelled_action)

        expired = create_task(connection, "Просрочена")
        expired_action = create_pending_operation(
            connection, "delete_task", {"task_id": expired}, [expired]
        )
        connection.execute(
            "UPDATE operations SET expires_at = ? WHERE action_id = ?",
            ("2000-01-01T00:00:00+00:00", expired_action),
        )
        connection.commit()

    for action_id, message in (
        (stale_action, "изменилась"),
        (cancelled_action, "уже обработана"),
        (expired_action, "истекло"),
    ):
        with pytest.raises(ValueError, match=message):
            confirm_delete_task(action_id, db_path)

    with closing(connect(db_path)) as connection:
        for task_id in (stale, cancelled, expired):
            assert get_task(connection, task_id) is not None
        assert get_operation(connection, cancelled_action)["status"] == "cancelled"


def test_delete_confirmation_rejects_mismatched_task_id(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        first = create_task(connection, "A")
        second = create_task(connection, "B")
        action_id = create_pending_operation(
            connection, "delete_task", {"task_id": second}, [first]
        )
        connection.commit()

    with pytest.raises(ValueError, match="неверный ID"):
        confirm_delete_task(action_id, db_path)

    with closing(connect(db_path)) as connection:
        assert get_task(connection, first) is not None
        assert get_task(connection, second) is not None
        assert get_operation(connection, action_id)["status"] == "pending"


def test_blocked_completion_changes_only_target_after_confirmation(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        first, second, third = (create_task(connection, title) for title in ("A", "B", "C"))
        add_dependency(connection, first, second)
        add_dependency(connection, second, third)
        connection.commit()

    prepared = prepare_blocked_completion(second, db_path)
    assert prepared["task_id"] == second
    assert prepared["unfinished_predecessor_ids"] == [first]

    with closing(connect(db_path)) as connection:
        assert get_task(connection, second)["status"] == "todo"
        assert get_operation(connection, prepared["action_id"])["status"] == "pending"

    assert confirm_blocked_completion(prepared["action_id"], db_path)
    assert not confirm_blocked_completion(prepared["action_id"], db_path)

    with closing(connect(db_path)) as connection:
        assert get_task(connection, first)["status"] == "todo"
        assert get_task(connection, second)["status"] == "done"
        assert get_task(connection, third)["status"] == "todo"
        operation = get_operation(connection, prepared["action_id"])
        assert operation["status"] == "confirmed"
        assert json.loads(operation["result_json"]) == {"completed_task_id": second}


def test_cancelled_blocked_completion_does_not_change_task(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        first = create_task(connection, "A")
        second = create_task(connection, "B")
        add_dependency(connection, first, second)
        connection.commit()

    prepared = prepare_blocked_completion(second, db_path)
    with closing(connect(db_path)) as connection:
        assert cancel_operation(connection, prepared["action_id"])
        connection.commit()

    with pytest.raises(ValueError, match="уже обработана"):
        confirm_blocked_completion(prepared["action_id"], db_path)

    with closing(connect(db_path)) as connection:
        assert get_task(connection, second)["status"] == "todo"


def test_blocked_completion_rechecks_predecessor_status(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        first = create_task(connection, "A")
        second = create_task(connection, "B")
        add_dependency(connection, first, second)
        connection.commit()

    prepared = prepare_blocked_completion(second, db_path)
    with closing(connect(db_path)) as connection:
        assert change_task_status(connection, first, "done")
        connection.commit()

    with pytest.raises(ValueError, match="Зависимости изменились"):
        confirm_blocked_completion(prepared["action_id"], db_path)

    with closing(connect(db_path)) as connection:
        assert get_task(connection, second)["status"] == "todo"
        assert get_operation(connection, prepared["action_id"])["status"] == "pending"


def test_blocked_completion_rejects_new_dependency(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        first, second, third = (create_task(connection, title) for title in ("A", "B", "C"))
        add_dependency(connection, first, second)
        connection.commit()

    prepared = prepare_blocked_completion(second, db_path)
    with closing(connect(db_path)) as connection:
        add_dependency(connection, third, second)
        connection.commit()

    with pytest.raises(ValueError, match="изменилась"):
        confirm_blocked_completion(prepared["action_id"], db_path)

    with closing(connect(db_path)) as connection:
        assert get_task(connection, second)["status"] == "todo"
