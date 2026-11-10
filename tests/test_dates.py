"""Даты, которые код вычисляет из запроса сам: дни недели и «завтра/через N дней»."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from ai_task_manager.agents.dates import corrected_deadline, mentioned_dates

# среда, 14 октября 2026
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=ZoneInfo("Europe/Moscow"))


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("приготовить курицу к воскресенью", {"к воскресенью": "2026-10-18"}),
        ("отчёт к пятнице", {"к пятнице": "2026-10-16"}),
        ("ко вторнику", {"ко вторнику": "2026-10-20"}),
        ("к субботе", {"к субботе": "2026-10-17"}),
        ("к четвергу", {"к четвергу": "2026-10-15"}),
        ("к среде", {"к среде": "2026-10-14"}),
        ("до воскресенья", {"до воскресенья": "2026-10-18"}),
        ("в эту субботу", {"в эту субботу": "2026-10-17"}),
        ("запустить тесты в среде Linux", {}),
        ("настроить среду разработки", {}),
    ],
)
def test_weekday_forms(request_text: str, expected: dict[str, str]) -> None:
    found = {word: day.isoformat() for word, day in mentioned_dates(request_text, NOW).items()}
    assert found == expected


def test_wrong_day_from_analyst_is_corrected() -> None:
    # живой случай: «к воскресенью», а аналитик поставил пятницу
    fix = corrected_deadline("2026-10-16", "приготовить курицу к воскресенью", NOW)
    assert fix is not None
    assert fix[0] == "2026-10-18"
    assert "к воскресенью" in fix[1]


def test_time_is_kept_when_day_is_corrected() -> None:
    fix = corrected_deadline("2026-10-16T18:00:00+03:00", "созвон в воскресенье в 18:00", NOW)
    assert fix is not None
    assert fix[0] == "2026-10-18T18:00:00+03:00"


@pytest.mark.parametrize(
    ("deadline", "request_text"),
    [
        ("2026-10-18", "к воскресенью"),  # уже верно
        ("2026-10-21", "в следующую среду"),  # «следующая» - допустимы обе среды
        ("2026-10-16", "завтра или в пятницу"),  # две даты в запросе - не угадываем
        ("2026-10-16", "к 16 числу"),  # код такую дату не вычисляет
        (None, "к воскресенью"),  # срока нет
    ],
)
def test_deadline_is_not_touched(deadline: str | None, request_text: str) -> None:
    assert corrected_deadline(deadline, request_text, NOW) is None
