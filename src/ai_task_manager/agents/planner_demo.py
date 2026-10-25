"""Ручная проверка планировщика на настоящей модели из Ollama.

    python -m ai_task_manager.agents.planner_demo                  # модель из ATM_PLANNER_MODEL
    python -m ai_task_manager.agents.planner_demo --model qwen3:1.7b

Создаёт временную БД с примером задач (рабочая БД не трогается), строит план порядка
выполнения и план удаления, печатает их в JSON.

Ожидаемый разумный порядок: сначала доклад (срок через 3 дня) и баг (high), затем
«Собрать данные» строго раньше «Написать отчёт», заметки (low) — в конце. Конфликт: отчёт
сдаётся раньше, чем готовы данные.
"""

import argparse
import tempfile
import time as timer
from contextlib import closing
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_task_manager.agents.llm import OllamaJsonLLM
from ai_task_manager.agents.planner import Planner
from ai_task_manager.config import get_settings
from ai_task_manager.db.crud_func import add_dependency, create_task
from ai_task_manager.db.database import connect, init_db


def _fill(connection, tz: ZoneInfo) -> dict[str, int]:
    """Пример задач со сроками относительно сегодняшнего дня."""
    today = datetime.now(tz).date()

    def day(offset: int) -> str:
        return (today + timedelta(days=offset)).isoformat()

    talk_at = datetime.combine(today + timedelta(days=3), time(18, 0), tzinfo=tz)
    ids = {
        "data": create_task(connection, "Собрать данные для отчёта", due_date=day(11)),
        "report": create_task(connection, "Написать отчёт по практике", due_date=day(9)),
        "bug": create_task(connection, "Исправить баг в авторизации", priority="high"),
        "talk": create_task(
            connection,
            "Подготовить доклад к семинару",
            due_date=talk_at.date().isoformat(),
            due_time="18:00",
            due_timezone=tz.key,
            due_at_utc=talk_at.astimezone(UTC).isoformat(),
        ),
        "notes": create_task(connection, "Разобрать старые заметки", priority="low"),
    }
    add_dependency(connection, ids["data"], ids["report"])
    connection.commit()
    return ids


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="python -m ai_task_manager.agents.planner_demo")
    parser.add_argument("--model", default=settings.planner_model)
    args = parser.parse_args(argv)

    llm = OllamaJsonLLM(args.model, settings.ollama_base_url)
    planner = Planner(llm, timezone=settings.timezone)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.db"
        init_db(path)
        with closing(connect(path)) as connection:
            ids = _fill(connection, ZoneInfo(settings.timezone))

            started = timer.perf_counter()
            plan = planner.plan(
                {"intent": "plan_tasks"},
                connection,
                request="Что из моих задач лучше сделать сначала?",
            )
            elapsed = timer.perf_counter() - started
            print(f"=== plan_tasks ({args.model}, {elapsed:.1f} с, включая загрузку модели)")
            print(plan.model_dump_json(indent=2))

            plan = planner.plan(
                {"intent": "delete_task", "task_id": ids["data"]},
                connection,
                request="Удали задачу без подтверждения",
            )
            print("=== delete_task")
            print(plan.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
