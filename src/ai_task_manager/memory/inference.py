"""Поиск возможных предпочтений пользователя (P12).

Найденное здесь — только догадка. Она сохраняется как `proposed` (A13) и не влияет на
планирование, пока пользователь её не подтвердит (FR-17, AC-19). Явная просьба «запомни…»
сюда не попадает: её разбирает аналитик (`update_preference`) и сохраняет сразу.

Источники догадок, по убыванию надёжности:
1. **текст запроса** — пользователь мимоходом сказал о себе («мне удобнее работать вечером»),
   но просил о другом (создать задачу, составить план). Ищется кодом по шаблонам;
2. **подсказка аналитика** — модель заполнила `preference` при другом намерении;
3. **поведение** — пользователь обычно завершает задачи в одно и то же время суток.

Значения приводятся к одному виду, чтобы «вечером», «по вечерам» и «evening» считались одним
предпочтением: иначе отклонённая догадка предлагалась бы снова в другой формулировке.
"""

import re
from collections import Counter
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from ai_task_manager.memory.models import PreferenceKind

WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
WORK_TIMES = ["утром", "днём", "вечером", "ночью"]

# Сколько завершённых задач нужно, чтобы говорить о привычке, и какая доля должна совпасть.
MIN_COMPLETED_TASKS = 5
MIN_SHARE = 0.8

_TIME_WORDS = {
    "утро": "утром",
    "утром": "утром",
    "по утрам": "утром",
    "с утра": "утром",
    "morning": "утром",
    "день": "днём",
    "днём": "днём",
    "днем": "днём",
    "в обед": "днём",
    "afternoon": "днём",
    "day": "днём",
    "вечер": "вечером",
    "вечером": "вечером",
    "по вечерам": "вечером",
    "evening": "вечером",
    "ночь": "ночью",
    "ночью": "ночью",
    "по ночам": "ночью",
    "night": "ночью",
}
_DAY_STEMS = {
    "понедельник": "понедельник",
    "вторник": "вторник",
    "сред": "среда",
    "четверг": "четверг",
    "пятниц": "пятница",
    "суббот": "суббота",
    "воскресен": "воскресенье",
    "monday": "понедельник",
    "tuesday": "вторник",
    "wednesday": "среда",
    "thursday": "четверг",
    "friday": "пятница",
    "saturday": "суббота",
    "sunday": "воскресенье",
}

_TIME = r"(по утрам|с утра|утром|днём|днем|в обед|по вечерам|вечером|по ночам|ночью)"
_GAP = r"[^.!?;]{0,30}?"
_WORK_TIME_PATTERNS = [
    # «мне удобнее работать вечером», «я обычно работаю по утрам»
    re.compile(
        r"\b(?:мне|я)\b" + _GAP + r"\b(?:удобн\w*|предпочитаю|люблю|привык\w*|обычно|всегда|"
        r"лучше|продуктивн\w*|работаю|занимаюсь)\b" + _GAP + r"\b" + _TIME + r"\b"
    ),
    # «удобнее работать по вечерам» без «мне»
    re.compile(r"\b(?:удобн\w*|предпочитаю|привык\w*)\b" + _GAP + r"\b" + _TIME + r"\b"),
]
_WEEKDAYS_ONLY = re.compile(
    r"\b(?:не работаю|отдыхаю)\b[^.!?;]{0,15}\b(?:по выходным|в выходные)\b"
    r"|\b(?:работаю|рабочие дни)\b[^.!?;]{0,15}\b(?:по будням|в будни|"
    r"с понедельника по пятницу)\b"
)
_EVERY_DAY = re.compile(r"\bработаю\b[^.!?;]{0,15}\b(?:без выходных|каждый день|всю неделю)\b")
_DAY = r"(?:понедельник|вторник|сред|четверг|пятниц|суббот|воскресень)\w*"
_DAY_LIST = re.compile(
    r"\b(?:работаю|рабочие дни)\b[^.!?;]{0,12}?\b((?:(?:по|в|во)\s+)?"
    + _DAY
    + r"(?:\s*(?:,|и)\s*(?:(?:по|в|во)\s+)?"
    + _DAY
    + r")+)"
)


class PreferenceCandidate(BaseModel):
    kind: PreferenceKind
    value: str | list[str]
    source: Literal["text", "analyst", "behavior"]
    # Чем подтверждается догадка — попадает в вопрос пользователю и в трейсы.
    evidence: str


# ---------------------------------------------------------------- нормализация


def _clean(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


def normalize_work_time(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    key = _clean(value).replace("днем", "днём")
    if key in _TIME_WORDS:
        return _TIME_WORDS[key]
    for word, canonical in _TIME_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", key):
            return canonical
    return None


def normalize_days(value: Any) -> list[str] | None:
    items = value if isinstance(value, list) else [value]
    days: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            return None
        text = _clean(item)
        if re.search(r"будн|weekdays|пн\s*-\s*пт", text):
            days.update(WEEKDAYS[:5])
            continue
        if re.search(r"выходн|weekend", text):
            days.update(WEEKDAYS[5:])
            continue
        found = [day for stem, day in _DAY_STEMS.items() if stem in text]
        if not found:
            return None
        days.update(found)
    return [day for day in WEEKDAYS if day in days] or None


def normalize_value(kind: str, value: Any) -> str | list[str] | None:
    """Единый вид значения; None — значение не распознано, предлагать нечего."""
    if kind == "preferred_work_time":
        return normalize_work_time(value)
    if kind == "working_days":
        return normalize_days(value)
    if kind == "preferred_task_types":
        items = value if isinstance(value, list) else [value]
        cleaned = [_clean(item) for item in items if isinstance(item, str) and item.strip()]
        return cleaned or None
    return None


def same_value(kind: str, left: Any, right: Any) -> bool:
    a, b = normalize_value(kind, left), normalize_value(kind, right)
    if a is None or b is None:
        return False
    if isinstance(a, list) and isinstance(b, list):
        return sorted(a) == sorted(b)
    return a == b


def describe(kind: str, value: Any) -> str:
    """Человекочитаемый вид после «вам …»: для вопроса, ответа и строки «учтены предпочтения»."""
    if kind == "preferred_work_time":
        return f"удобнее работать {normalize_work_time(value) or value}"
    if kind == "working_days":
        days = normalize_days(value)
        if days == WEEKDAYS[:5]:
            return "удобнее работать по будням"
        if days == WEEKDAYS:
            return "удобно работать всю неделю"
        shown = ", ".join(days) if days else str(value)
        return f"удобнее работать в эти дни: {shown}"
    if kind == "preferred_task_types":
        items = value if isinstance(value, list) else [value]
        return "предпочтительнее задачи такого типа: " + ", ".join(str(item) for item in items)
    return f"{kind}: {value}"


# ---------------------------------------------------------------- источники


def from_text(request: str) -> list[PreferenceCandidate]:
    text = _clean(request)
    found: list[PreferenceCandidate] = []
    for pattern in _WORK_TIME_PATTERNS:
        match = pattern.search(text)
        if match:
            found.append(
                PreferenceCandidate(
                    kind="preferred_work_time",
                    value=normalize_work_time(match.group(1)) or match.group(1),
                    source="text",
                    evidence=f"вы написали «{match.group(0)}»",
                )
            )
            break

    days: list[str] | None = None
    match = _WEEKDAYS_ONLY.search(text)
    if match:
        days = WEEKDAYS[:5]
    elif match := _EVERY_DAY.search(text):
        days = list(WEEKDAYS)
    elif match := _DAY_LIST.search(text):
        days = normalize_days(re.split(r",|\bи\b", match.group(1)))
    if match and days:
        found.append(
            PreferenceCandidate(
                kind="working_days",
                value=days,
                source="text",
                evidence=f"вы написали «{match.group(0)}»",
            )
        )
    return found


def from_analysis(analysis: dict[str, Any] | None) -> list[PreferenceCandidate]:
    """Подсказка аналитика: `preference` при другом намерении (см. contracts.Analysis)."""
    if not isinstance(analysis, dict) or analysis.get("intent") == "update_preference":
        return []
    hint = analysis.get("preference_hint") or analysis.get("preference")
    if not isinstance(hint, dict) or not hint.get("kind"):
        return []
    value = normalize_value(hint["kind"], hint.get("value"))
    if value is None:
        return []
    return [
        PreferenceCandidate(
            kind=hint["kind"],
            value=value,
            source="analyst",
            evidence="так понял ваше сообщение аналитик",
        )
    ]


def _part_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "утром"
    if 12 <= hour < 18:
        return "днём"
    if 18 <= hour < 24:
        return "вечером"
    return "ночью"


def from_behavior(tasks: list[dict[str, Any]], timezone: str) -> list[PreferenceCandidate]:
    """Привычка: большинство задач завершено в одно и то же время суток.

    Время завершения — `updated_at` задачи со статусом done (последнее изменение — отметка
    о выполнении). Срок задачи не используется: «сдать до 18:00» не значит «работаю вечером».
    """
    tz = ZoneInfo(timezone)
    parts: list[str] = []
    for task in tasks:
        if task.get("status") != "done" or not task.get("updated_at"):
            continue
        try:
            moment = datetime.fromisoformat(task["updated_at"])
        except ValueError:
            continue
        if moment.tzinfo is None:
            continue
        parts.append(_part_of_day(moment.astimezone(tz).hour))
    if len(parts) < MIN_COMPLETED_TASKS:
        return []
    part, count = Counter(parts).most_common(1)[0]
    if count / len(parts) < MIN_SHARE:
        return []
    return [
        PreferenceCandidate(
            kind="preferred_work_time",
            value=part,
            source="behavior",
            evidence=f"{count} из {len(parts)} последних выполненных задач вы закрыли {part}",
        )
    ]


def find_candidate(
    request: str,
    analysis: dict[str, Any] | None,
    tasks: list[dict[str, Any]],
    confirmed: dict[str, Any],
    *,
    timezone: str,
) -> PreferenceCandidate | None:
    """Одна догадка за ход — чтобы не засыпать пользователя вопросами.

    Пропускаются догадки, совпадающие с уже подтверждённым значением. Поведение не спорит с
    подтверждённым: если пользователь сам сказал «утром», привычка его не переспрашивает.
    """
    for candidate in [*from_text(request), *from_analysis(analysis)]:
        current = confirmed.get(candidate.kind)
        if current is None or not same_value(candidate.kind, current, candidate.value):
            return candidate
    for candidate in from_behavior(tasks, timezone):
        if candidate.kind not in confirmed:
            return candidate
    return None
