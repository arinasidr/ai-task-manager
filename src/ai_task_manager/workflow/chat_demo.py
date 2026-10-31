"""Полный диалог через workflow на настоящих моделях (нужна запущенная Ollama).

    python -m ai_task_manager.workflow.chat_demo

Временная БД, рабочая не трогается. Каждое сообщение проходит роутер → аналитик →
планировщик → валидатор → действие; печатаются ответ, модели, вердикт и время.
Удаление подтверждается автоматически, чтобы показать весь путь.
В середине диалога — долгосрочная память (P12): предпочтение, сказанное мимоходом, система
предлагает запомнить, ответ «да» его сохраняет, и «С чего начать?» уже его учитывает.
"""

import tempfile
import time
from contextlib import closing
from pathlib import Path

from ai_task_manager.config import get_settings
from ai_task_manager.db.database import connect, init_db
from ai_task_manager.db.preferences import get_preferences
from ai_task_manager.memory.checkpoint import open_checkpointer
from ai_task_manager.tools.confirmation import confirm_plan_action
from ai_task_manager.workflow.service import create_workflow, run_chat

MESSAGES = [
    "Добавь две задачи: сначала собрать данные для отчёта, потом написать отчёт по практике",
    "Добавь срочную задачу исправить баг в авторизации на завтра",
    # P12: предпочтение мимоходом → вопрос «запомнить?» → «да» → план его учитывает
    "Добавь задачу созвон с командой, мне вообще удобнее работать вечером",
    "да",
    "Покажи задачи",
    "С чего мне лучше начать?",
    "Удали задачу #1",
    "Удали отчёт",
]


def _debug(result) -> None:
    """Что понял аналитик и почему план такой — чтобы разбирать неудачные ответы."""
    if result.analysis is None:
        print("[аналитик не вызывался: ответ разобран кодом]")
        return
    analysis = result.analysis
    parts = [f"intent={analysis.get('intent')}"]
    if analysis.get("tasks"):
        parts.append("задачи=" + str([task.get("title") for task in analysis["tasks"]]))
    if analysis.get("dependencies"):
        deps = [f"{d.get('task')} ← {d.get('depends_on')}" for d in analysis["dependencies"]]
        parts.append(f"связи={deps}")
    if analysis.get("task_id") or analysis.get("task_ids"):
        parts.append(f"ID={analysis.get('task_id') or analysis.get('task_ids')}")
    if analysis.get("updates"):
        updates = {k: v for k, v in analysis["updates"].items() if v not in (None, [])}
        if updates:
            parts.append(f"изменения={updates}")
    print(f"[аналитик: {'; '.join(parts)}]")
    plan = result.plan or {}
    if plan and (
        plan.get("needs_clarification") or (result.verdict or {}).get("decision") != "accept"
    ):
        print(f"[план: source={plan.get('source')}; reasoning={plan.get('reasoning')}]")
    for note in (result.verdict or {}).get("notes") or []:
        print(f"[валидатор: {note}]")


def _run_dialogue(settings, path: Path) -> None:
    with open_checkpointer(path) as checkpointer:
        graph = create_workflow(settings, connect=lambda: connect(path), checkpointer=checkpointer)
        conversation_id = None  # один диалог: история копится в чекпоинтере (A11)

        for message in MESSAGES:
            started = time.perf_counter()
            result = run_chat(graph, settings, message, conversation_id)
            conversation_id = result.conversation_id
            elapsed = time.perf_counter() - started
            verdict = (result.verdict or {}).get("decision", "—")
            print(f"\n>>> {message}")
            print(
                f"[{elapsed:.1f} с; маршрут {result.route}; модели {result.models}; "
                f"валидатор: {verdict}, проверок {result.validation_attempts}; "
                f"реплик в истории {result.history_length}]"
            )
            if result.preference_id:
                print(f"[предложено предпочтение #{result.preference_id}, ждёт «да/нет»]")
            if result.error:
                print(f"[ошибка: {result.error}]")
            _debug(result)
            print(result.reply)

            if result.action_id:
                done = confirm_plan_action(result.action_id, path)
                print(f"\n>>> (пользователь подтвердил)\n{done.summary() if done else '—'}")


def main() -> int:
    base = get_settings()
    # ignore_cleanup_errors: на Windows SQLite может ещё держать файл в момент удаления
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "demo.db"
        init_db(path)
        settings = base.model_copy(update={"database_path": str(path)})
        _run_dialogue(settings, path)
        with closing(connect(path)) as db:
            titles = [row["title"] for row in db.execute("SELECT title FROM tasks ORDER BY id")]
            preferences = get_preferences(db)
        print(f"\nВ БД осталось: {titles}")
        print(f"Подтверждённые предпочтения: {preferences}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
