"""Долгосрочная память в графе (P12): догадка → вопрос пользователю → ответ «да/нет».

Два узла:

    prepare → preference_answer ─┬─ ответ «да/нет» на прошлый вопрос ──────► remember_reply
                                 └─ новый запрос → decision/route → … → finish
    finish → propose_preference → remember_reply

- `propose_preference` после ответа на запрос ищет возможное предпочтение
  (`memory.inference`), сохраняет его как `proposed` и дописывает к ответу вопрос.
  Догадка не влияет на планирование: планировщик читает только `confirmed` (AC-19).
- `preference_answer` в начале следующего хода: короткое «да»/«нет» подтверждает или отклоняет
  предложение кодом, без вызова моделей. Любое другое сообщение — новый запрос: вопрос
  снимается, предложение остаётся неподтверждённым (его можно подтвердить через API).

Ожидающий вопрос хранится в состоянии диалога (`pending_preference`, чекпоинтер A11/A12),
а само предложение — в таблице `preferences` (A13).
"""

import json
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from typing import Any

from ai_task_manager.db.crud_func import list_tasks
from ai_task_manager.db.preferences import (
    confirm_preference,
    get_preferences,
    propose_preference,
    reject_preference,
)
from ai_task_manager.memory.inference import describe, find_candidate, same_value

Connect = Callable[[], sqlite3.Connection]

# Не предлагаем запомнить что-то, если ход закончился вопросом, отказом или ошибкой.
_SKIP_INTENTS = frozenset({"update_preference", "clarification_required", "unsupported"})

_YES = re.compile(
    r"(да|ага|угу|ок|окей|хорошо|конечно|верно|подтверждаю|запомни|сохрани|yes)(\s.*)?"
)
_NO = re.compile(
    r"(нет|не надо|не нужно|не запоминай|не сохраняй|отмена|отклоняю|неверно|no)(\s.*)?"
)
_MAX_ANSWER_WORDS = 4


def classify_answer(message: str) -> bool | None:
    """True — «да», False — «нет», None — это не ответ на вопрос, а новый запрос."""
    text = " ".join(re.sub(r"[^\w\s-]", " ", message.casefold().replace("ё", "е")).split())
    if not text or len(text.split()) > _MAX_ANSWER_WORDS:
        return None
    if _NO.fullmatch(text):
        return False
    if _YES.fullmatch(text):
        return True
    return None


def make_preference_answer_node(*, connect: Connect) -> Callable[[dict], dict]:
    def preference_answer(state: dict[str, Any]) -> dict[str, Any]:
        pending = state.get("pending_preference")
        if not pending:
            return {}
        answer = classify_answer(state["request"])
        if answer is None:
            # новый запрос: вопрос больше не висит, предложение остаётся неподтверждённым
            return {"pending_preference": None}

        with closing(connect()) as connection:
            if answer:
                try:
                    confirmed = confirm_preference(connection, pending["id"])
                except ValueError:
                    reply = "Это предложение уже обработано, ничего не изменил."
                else:
                    reply = (
                        f"Запомнил: вам {describe(confirmed.kind, confirmed.value)}. "
                        "Теперь учитываю это, когда предлагаю порядок задач."
                    )
            else:
                if reject_preference(connection, pending["id"]):
                    reply = "Хорошо, не запоминаю и больше не буду это предлагать."
                else:
                    reply = "Это предложение уже обработано, ничего не изменил."
            connection.commit()
        return {"pending_preference": None, "reply": reply}

    return preference_answer


def _already_proposed(connection: sqlite3.Connection, kind: str, value: Any) -> bool:
    rows = connection.execute(
        "SELECT value_json FROM preferences WHERE kind = ? AND status = 'proposed'", (kind,)
    ).fetchall()
    return any(same_value(kind, json.loads(row["value_json"]), value) for row in rows)


def make_preference_proposal_node(*, connect: Connect) -> Callable[[dict], dict]:
    def propose_preference_node(state: dict[str, Any]) -> dict[str, Any]:
        analysis = state.get("analysis")
        verdict = state.get("verdict") or {}
        if (
            not state.get("reply")
            or not isinstance(analysis, dict)
            or analysis.get("intent") in _SKIP_INTENTS
            or state.get("pending_action_id")  # один вопрос за раз: сначала подтверждение действия
            or state.get("error")
            or verdict.get("decision") == "reject"
        ):
            return {}

        with closing(connect()) as connection:
            candidate = find_candidate(
                state["request"],
                analysis,
                list_tasks(connection),
                get_preferences(connection),
                timezone=state["timezone"],
            )
            if candidate is None:
                return {}
            # догадку из анализа или поведения спрашиваем один раз; повтор в тексте — новый повод
            if candidate.source != "text" and _already_proposed(
                connection, candidate.kind, candidate.value
            ):
                return {}
            try:
                proposal = propose_preference(connection, candidate.kind, candidate.value)
            except ValueError:
                # уже действует или пользователь раньше отказался — не переспрашиваем
                return {}
            connection.commit()

        question = (
            f"Кстати, запомнить, что вам {describe(candidate.kind, candidate.value)}? "
            f"({candidate.evidence[0].upper()}{candidate.evidence[1:]}.) "
            "Ответьте «да» или «нет». Пока вы не ответили, я это не учитываю."
        )
        return {
            "reply": f"{state['reply']}\n\n{question}",
            "pending_preference": {
                "id": proposal.id,
                "kind": proposal.kind,
                "value": proposal.value,
            },
            "preference_id": proposal.id,
        }

    return propose_preference_node
