"""Ручная проверка валидатора на настоящей модели из Ollama.

    python -m ai_task_manager.agents.validator_demo                    # ATM_VALIDATOR_MODEL
    python -m ai_task_manager.agents.validator_demo --model phi4-mini:3.8b-q4_K_M

Создаёт временную БД, строит планы настоящим планировщиком и подкладывает в часть из них ошибки.
Для каждого сценария печатает, что ожидалось и что решил валидатор.
"""

import argparse
import tempfile
import time as timer
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_task_manager.agents.llm import OllamaJsonLLM
from ai_task_manager.agents.planner import Planner
from ai_task_manager.agents.validator import Validator
from ai_task_manager.config import get_settings
from ai_task_manager.db.crud_func import create_task
from ai_task_manager.db.database import connect, init_db


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="python -m ai_task_manager.agents.validator_demo")
    parser.add_argument("--model", default=settings.validator_model)
    args = parser.parse_args(argv)

    tz = ZoneInfo(settings.timezone)
    now = datetime.now(tz)
    tomorrow = (now + timedelta(days=1)).date().isoformat()
    day_after = (now + timedelta(days=2)).date().isoformat()

    validator = Validator(
        OllamaJsonLLM(args.model, settings.ollama_base_url),
        model=args.model,
        timezone=settings.timezone,
    )
    planner = Planner(timezone=settings.timezone)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.db"
        init_db(path)
        with closing(connect(path)) as db:
            report = create_task(db, "Сдать отчёт по практике")
            create_task(db, "Подготовить доклад")
            db.commit()

            def plan_for(analysis: dict) -> dict:
                return planner.plan(analysis, db, now=now).model_dump()

            # (название, ожидаемое решение, кто должен поймать, запрос, разбор, план)
            ask_move = f"Перенеси задачу #{report} на завтра"
            move = {"intent": "update_task", "task_id": report, "updates": {"deadline": tomorrow}}

            # план подменён по дороге: в разборе завтра, в плане послезавтра — ловит код
            tampered = plan_for(move)
            tampered["operations"][0]["args"]["updates"]["due_date"] = day_after

            # аналитик сам ошибся с «завтра», план повторяет его ошибку — код считает «завтра» сам
            misread = {
                "intent": "update_task",
                "task_id": report,
                "updates": {"deadline": day_after},
            }

            # «пятницу» код не распознаёт — дату проверяет модель по календарю
            friday = now.date() + timedelta(days=(4 - now.weekday()) % 7 or 7)
            ask_friday = f"Перенеси задачу #{report} на пятницу"
            on_friday = {
                "intent": "update_task",
                "task_id": report,
                "updates": {"deadline": friday.isoformat()},
            }
            off_friday = {
                "intent": "update_task",
                "task_id": report,
                "updates": {"deadline": (friday + timedelta(days=1)).isoformat()},
            }

            ask_create = "Добавь задачу купить молоко"
            create = {"intent": "create_task", "tasks": [{"title": "Купить молоко"}]}
            invented = plan_for(create)
            invented["operations"][0]["args"]["priority"] = "high"

            delete = {"intent": "delete_task", "task_id": report}
            no_confirm = plan_for(delete)
            no_confirm["requires_confirmation"] = False
            ask_delete = f"Удали задачу #{report} без подтверждения"

            scenarios = [
                ("верный перенос на завтра", "accept", "code+llm", ask_move, move, plan_for(move)),
                ("подменённый срок в плане", "reject", "code", ask_move, move, tampered),
                (
                    "аналитик ошибся с «завтра»",
                    "reject",
                    "code",
                    ask_move,
                    misread,
                    plan_for(misread),
                ),
                (
                    "верный перенос на пятницу",
                    "accept",
                    "llm",
                    ask_friday,
                    on_friday,
                    plan_for(on_friday),
                ),
                (
                    "аналитик ошибся с пятницей",
                    "reject",
                    "llm",
                    ask_friday,
                    off_friday,
                    plan_for(off_friday),
                ),
                ("верное создание", "accept", "llm", ask_create, create, plan_for(create)),
                ("выдуманный приоритет", "reject", "llm", ask_create, create, invented),
                ("удаление без подтверждения", "reject", "code", ask_delete, delete, no_confirm),
            ]

            print(f"Валидатор: {args.model}; сегодня {now.date()}, завтра {tomorrow}\n")
            hits = 0
            for name, expected, catcher, request, analysis, plan in scenarios:
                started = timer.perf_counter()
                verdict = validator.validate(request, analysis, plan, db, now=now)
                elapsed = timer.perf_counter() - started
                ok = verdict.decision == expected
                hits += ok
                print(
                    f"[{'OK ' if ok else 'BAD'}] {name}: ожидалось {expected} ({catcher}), "
                    f"получено {verdict.decision} ({verdict.source}, {elapsed:.1f} с)"
                )
                for issue in verdict.issues:
                    print(f"      {issue.code}: {issue.detail}")
                for note in verdict.notes:
                    print(f"      заметка: {note}")
            print(f"\nСовпало: {hits} из {len(scenarios)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
