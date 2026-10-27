"""Демонстрация подтверждения опасных операций (без LLM, на временной БД).

    python -m ai_task_manager.tools.confirmation_demo

Сценарии: удаление с предварительным планом и подтверждением, отмена, завершение задачи с
незавершённой зависимостью, задача изменилась после показа плана, безопасное действие без
подтверждения. План строит настоящий планировщик; решение валидатора считается «accept»
(сам валидатор проверен в P08).
"""

import tempfile
from contextlib import closing
from pathlib import Path

from ai_task_manager.agents.planner import Planner
from ai_task_manager.db.crud_func import (
    add_dependency,
    cancel_operation,
    create_task,
    get_task,
    update_task,
)
from ai_task_manager.db.database import connect, init_db
from ai_task_manager.tools.confirmation import confirm_plan_action, make_action_node


def _show(title: str, text: str) -> None:
    print(f"\n=== {title}")
    print(text)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.db"
        init_db(path)
        node = make_action_node(connect=lambda: connect(path))
        planner = Planner()

        with closing(connect(path)) as db:
            data = create_task(db, "Собрать данные")
            report = create_task(db, "Написать отчёт")
            notes = create_task(db, "Разобрать заметки")
            add_dependency(db, data, report)
            db.commit()

            def run(request: str, analysis: dict) -> dict:
                plan = planner.plan(analysis, db)
                state = {
                    "request": request,
                    "plan": plan.model_dump(),
                    "verdict": {"decision": "accept"},
                }
                return node(state)

            # 1. удаление: предварительный план → подтверждение
            update = run(f"Удали задачу #{data}", {"intent": "delete_task", "task_id": data})
            _show("1. «Удали задачу» — ответ системы", update["reply"])
            print(f"\nЗадача #{data} ещё есть: {get_task(db, data) is not None}")
            result = confirm_plan_action(update["pending_action_id"], path)
            _show("   пользователь подтвердил", result.summary())
            print(f"Задача #{data} есть: {get_task(db, data) is not None}")

            # 2. отмена
            update = run(f"Удали задачу #{notes}", {"intent": "delete_task", "task_id": notes})
            cancel_operation(db, update["pending_action_id"])
            db.commit()
            _show(
                "2. «Удали задачу» → пользователь отменил",
                f"Задача #{notes} на месте: {get_task(db, notes) is not None}",
            )

            # 3. завершение задачи, у которой не завершена зависимость
            blocker = create_task(db, "Согласовать план")
            add_dependency(db, blocker, report)
            db.commit()
            update = run(
                f"Отметь задачу #{report} выполненной",
                {"intent": "update_task", "task_id": report, "updates": {"status": "done"}},
            )
            _show("3. Завершение вопреки зависимости — ответ системы", update["reply"])
            confirm_plan_action(update["pending_action_id"], path)
            print(f"\nПосле подтверждения статус #{report}: {get_task(db, report)['status']}")

            # 4. задача изменилась после показа плана
            update = run(f"Удали задачу #{notes}", {"intent": "delete_task", "task_id": notes})
            update_task(db, notes, {"title": "Разобрать заметки (важное!)"})
            db.commit()
            try:
                confirm_plan_action(update["pending_action_id"], path)
                outcome = "удалена — ОШИБКА"
            except ValueError as exc:
                outcome = f"подтверждение отклонено: {exc}"
            _show("4. Задачу изменили после показа плана", outcome)

            # 5. безопасное действие выполняется сразу
            update = run(
                "Добавь задачу купить молоко",
                {"intent": "create_task", "tasks": [{"title": "Купить молоко"}]},
            )
            _show("5. Создание задачи — подтверждение не нужно", update["reply"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
