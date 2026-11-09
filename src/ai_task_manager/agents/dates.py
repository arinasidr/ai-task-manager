"""Даты, которые код вычисляет из текста запроса сам, не доверяя модели.

«сегодня», «завтра», «послезавтра», «через 3 дня», дни недели с предлогом: «в пятницу»,
«на воскресенье», «до среды», «к четвергу», «в следующий вторник». Остальные формулировки
(«15-го», «в конце месяца») не распознаются — их проверяет модель-валидатор.

Используется планировщиком (исправить срок, если аналитик ошибся в дне) и валидатором
(сверить срок плана с запросом и отбросить ошибки счёта модели).
"""

import re
from datetime import date, datetime, timedelta

_RELATIVE_WORDS = {"сегодня": 0, "завтра": 1, "послезавтра": 2}
_NUMBER_WORDS = {
    "один": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
}  # fmt: skip
_IN_DAYS = re.compile(r"\bчерез\s+(\d+|[а-яё]+)\s+(?:день|дня|дней)\b", re.IGNORECASE)
_WORD = re.compile(r"\b(послезавтра|завтра|сегодня)\b", re.IGNORECASE)

# Предлог обязателен: «настроить среду разработки» — не день недели. «к среде» — день недели,
# а «в среде Linux» — нет, поэтому форма «среде» допустима только после «к/ко».
_WEEKDAY_STEMS = ("понедельник", "вторник", "сред", "четверг", "пятниц", "суббот", "воскресень")
_WEEKDAY = re.compile(
    r"\b(в|во|на|к|ко|до)\s+(?:(эт[уоа]\w*|ближайш\w*|следующ\w*)\s+)?"
    r"(понедельник\w*|вторник\w*|сред[уаые]|четверг\w*|пятниц[уаые]|суббот[уаые]"
    r"|воскресень[еяю])\b",
    re.IGNORECASE,
)


def weekday_dates(request: str, today: date) -> dict[str, list[date]]:
    """«в воскресенье» → [ближайшее воскресенье]. Если сегодня тот же день недели или сказано
    «следующий», допустимы обе даты: ближайшая и через неделю."""
    found: dict[str, list[date]] = {}
    for match in _WEEKDAY.finditer(request):
        preposition, qualifier, word = (g.casefold() if g else "" for g in match.groups())
        if word == "среде" and preposition not in ("к", "ко"):
            continue
        weekday = next(i for i, stem in enumerate(_WEEKDAY_STEMS) if word.startswith(stem))
        nearest = today + timedelta(days=(weekday - today.weekday()) % 7)
        options = [nearest]
        if nearest == today or qualifier.startswith("следующ"):
            options.append(nearest + timedelta(days=7))
        found[match.group(0).casefold()] = options
    return found


def mentioned_dates(request: str, now: datetime) -> dict[str, date]:
    """Даты, которые код может вычислить из запроса: фраза → дата (день недели — ближайший)."""
    found: dict[str, date] = {}
    today = now.date()
    for phrase, options in weekday_dates(request, today).items():
        found[phrase] = options[0]
    for match in _WORD.finditer(request):
        word = match.group(1).casefold()
        found[word] = today + timedelta(days=_RELATIVE_WORDS[word])
    for match in _IN_DAYS.finditer(request):
        raw = match.group(1).casefold()
        days = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw)
        if days is not None:
            found[match.group(0).casefold()] = today + timedelta(days=days)
    return found


def corrected_deadline(value: str | None, request: str, now: datetime) -> tuple[str, str] | None:
    """Если в запросе ровно одна вычислимая дата, а срок от аналитика попал на другой день,
    возвращает (исправленный срок, пояснение). Время из срока сохраняется. Иначе None."""
    if not value or not str(value).strip():
        return None
    mentioned = mentioned_dates(request, now)
    if len(mentioned) != 1:
        return None
    ((phrase, expected),) = mentioned.items()
    allowed = weekday_dates(request, now.date()).get(phrase, [expected])
    text = str(value).strip()
    try:
        current = date.fromisoformat(text[:10])
    except ValueError:
        return None
    if current in allowed:
        return None
    fixed = expected.isoformat() + text[10:]
    note = f"Срок исправлен кодом: «{phrase}» = {expected.isoformat()}, аналитик указал {text[:10]}"
    return fixed, note
