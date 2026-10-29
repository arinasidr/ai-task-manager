"""Узлы графа с настоящими агентами: аналитик, планировщик, итоговый ответ.

Узел — функция State → обновление State. Модель для каждого агента выбрал роутер (A10) и
записал в State (`analyst_model`, `planner_model`, `validator_model`); узел получает клиента
этой модели через `llm_for(model)`. Валидатор и узел действия — в `agents/validator.py` и
`tools/confirmation.py`.
"""

import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from ai_task_manager.agents.analyst.client import analyze
from ai_task_manager.agents.analyst.schema import AnalystInput
from ai_task_manager.agents.llm import JsonLLM, LLMError
from ai_task_manager.agents.planner import Planner
from ai_task_manager.db.crud_func import list_tasks
from ai_task_manager.db.preferences import get_preferences

Connect = Callable[[], sqlite3.Connection]
LLMFor = Callable[[str], JsonLLM]

_SNAPSHOT_FIELDS = ("id", "title", "status", "priority", "due_date", "due_time", "due_timezone")

NOT_UNDERSTOOD = (
    "Не получилось разобрать запрос. Переформулируйте, пожалуйста, — данные не изменены."
)


def make_analyst_node(*, llm_for: LLMFor, connect: Connect) -> Callable[[dict], dict]:
    """Аналитик получает запрос, текущее время и актуальный список задач из БД."""

    def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
        with closing(connect()) as connection:
            tasks = [
                {key: task[key] for key in _SNAPSHOT_FIELDS} for task in list_tasks(connection)
            ]
        history = state.get("history") or []
        if history and history[-1]["role"] == "user" and history[-1]["content"] == state["request"]:
            history = history[:-1]
        data = AnalystInput.model_validate(
            {
                "request": state["request"],
                "current_datetime": state["current_datetime"],
                "timezone": state["timezone"],
                "tasks": tasks,
                # prepare уже добавил текущий запрос; аналитику нужны только прошлые реплики.
                "history": [
                    {"role": item["role"], "content": item["content"][:2000]}
                    for item in history[-4:]
                ],
            }
        )
        try:
            result = analyze(data, llm=llm_for(state["analyst_model"]))
        except (LLMError, ValidationError, ValueError) as exc:
            # Модель не ответила, ответ не прошёл схему или сослался на несуществующую задачу:
            # ничего не делаем и просим уточнить.
            question = NOT_UNDERSTOOD
            if isinstance(exc, LLMError) and "Ошибка вызова модели" in str(exc):
                question = "Модель сейчас недоступна, попробуйте позже. Данные не изменены."
            elif isinstance(exc, ValueError) and "Неизвестные ID" in str(exc):
                ids = ", ".join(f"#{n}" for n in re.findall(r"\d+", str(exc)))
                question = f"Задачи {ids} нет. Проверьте номер — могу показать список задач."
            return {
                "analysis": {
                    "intent": "clarification_required",
                    "clarification_question": question,
                },
                "error": f"analyst: {exc}",
            }
        return {"analysis": result.model_dump(mode="json")}

    return analyst_node


def make_planner_node(*, llm_for: LLMFor, connect: Connect) -> Callable[[dict], dict]:
    """Планировщик: план по разбору аналитика, предпочтениям и замечаниям валидатора."""

    def planner_node(state: dict[str, Any]) -> dict[str, Any]:
        planner = Planner(llm_for(state["planner_model"]), timezone=state["timezone"])
        with closing(connect()) as connection:
            plan = planner.plan(
                state["analysis"],
                connection,
                request=state["request"],
                now=datetime.fromisoformat(state["current_datetime"]),
                feedback=state.get("validator_feedback") or None,
                preferences=get_preferences(connection),
            )
        update: dict[str, Any] = {"plan": plan.model_dump(), "verdict": None}
        if planner.repaired_dependencies is not None:
            # аналитик ошибся в связи, код восстановил её по тексту запроса — дальше по графу
            # (валидатор, история) идёт исправленный разбор
            analysis = dict(state["analysis"])
            analysis["dependencies"] = [d.model_dump() for d in planner.repaired_dependencies]
            update["analysis"] = analysis
        # предыдущий вердикт относится к старому плану
        return update

    return planner_node


def finish_node(state: dict[str, Any]) -> dict[str, Any]:
    """Гарантирует ответ пользователю, если ни один узел его не дал."""
    if state.get("reply"):
        return {}

    verdict = state.get("verdict") or {}
    if verdict.get("decision") == "accept":
        # ответ даёт узел действия; без него (графы в тестах) ответ формирует вызывающий код
        return {}
    if verdict.get("decision") == "reject":
        reasons = [issue["detail"] for issue in verdict.get("issues") or []]
        text = "Не удалось подготовить действие, которое точно соответствует запросу"
        if reasons:
            text += ": " + "; ".join(reasons)
        return {"reply": text + ". Данные не изменены. Уточните, пожалуйста, запрос."}

    analysis = state.get("analysis") or {}
    if analysis.get("clarification_question"):
        return {"reply": analysis["clarification_question"]}
    if analysis.get("intent") == "clarification_required":
        return {"reply": "Уточните, пожалуйста, запрос: каких данных не хватает?"}
    if analysis.get("intent") == "unsupported":
        return {
            "reply": "Я умею работать только с задачами: создавать, менять, удалять "
            "и планировать порядок."
        }
    return {"reply": NOT_UNDERSTOOD}
