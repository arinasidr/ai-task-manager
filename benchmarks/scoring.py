"""Разбор ответа модели и сравнение с эталоном из cases.jsonl.

Правила сравнения:
- проверяются только поля, которые есть в `expected`; лишние поля ответа не штрафуются,
  кроме лишних изменений в `updates` и выдуманных задач или ID;
- названия задач сравниваются нечётко: регистр, «ё/е», кавычки и знаки препинания не важны,
  а нужна похожесть строк не ниже TITLE_THRESHOLD;
- сроки сравниваются как значения: дата с датой, момент времени с моментом
  (разные записи одного момента считаются равными, время без смещения — ошибка);
- `warnings` и `missing_fields`: эталонные коды должны присутствовать, лишние допустимы;
- `task_ids`, `matching_tasks`, `add_dependency_ids` сравниваются как множества,
  `ordered_task_ids` — как список с учётом порядка.

Часть проверок помечена как критичная: это ошибки, опасные для агента (выполнение действия
без подтверждения, выдуманные задачи или ID). Они считаются отдельно от точности.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any

TITLE_THRESHOLD = 0.8

# Версия правил оценки. Меняется при любом изменении проверок, пишется в результаты,
# чтобы не сравнивать прогоны, оценённые по-разному (пересчёт: python -m benchmarks.rescore).
# 1 — первая версия; 2 — для create_task поля с ID существующих задач не считаются выдумкой.
SCORING_VERSION = 2

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[«»\"'“”„.,!?:;()\[\]]")
_TASK_REF_RE = re.compile(r"#(\d+)")

# Поля ответа, в которых модель ссылается на существующие задачи
_ID_FIELDS = ("task_id", "task_ids", "ordered_task_ids", "matching_tasks", "candidate_task_ids")
_UPDATE_ID_FIELDS = ("add_dependency_ids", "remove_dependency_ids")
_SET_FIELDS = {"task_ids", "matching_tasks", "candidate_task_ids"}
_SUBSET_FIELDS = {"warnings", "missing_fields"}
_BOOL_FIELDS = {
    "requires_confirmation",
    "should_execute",
    "should_change_deadlines",
    "replaces_existing",
}


@dataclass
class Check:
    field: str
    ok: bool
    expected: Any = None
    actual: Any = None
    note: str = ""
    critical: bool = False


@dataclass
class ParsedOutput:
    data: dict[str, Any] | None
    strict_json: bool
    error: str | None = None


@dataclass
class CaseScore:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.ok for check in self.checks)

    @property
    def score(self) -> float:
        if not self.checks:
            return 0.0
        return sum(check.ok for check in self.checks) / len(self.checks)

    @property
    def critical_failures(self) -> list[str]:
        return [check.field for check in self.checks if check.critical and not check.ok]

    def failed(self) -> list[dict[str, Any]]:
        return [asdict(check) for check in self.checks if not check.ok]


# ---------------------------------------------------------------- разбор ответа


def parse_model_output(text: str) -> ParsedOutput:
    """Достаёт JSON-объект из ответа модели.

    strict_json=True означает, что весь ответ сразу оказался корректным JSON-объектом.
    Если модель обернула JSON в Markdown или добавила текст, объект извлекается,
    но strict_json=False: это ухудшает оценку следования формату, но не точность.
    """
    cleaned = _THINK_RE.sub("", text or "").strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return ParsedOutput(value, strict_json=True)
        return ParsedOutput(None, False, f"ответ является {type(value).__name__}, а не объектом")
    except json.JSONDecodeError:
        pass

    candidate = _FENCE_RE.sub("", cleaned).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(candidate[start : end + 1])
            if isinstance(value, dict):
                return ParsedOutput(value, strict_json=False)
        except json.JSONDecodeError as exc:
            return ParsedOutput(None, False, f"некорректный JSON: {exc}")
    return ParsedOutput(None, False, "в ответе нет JSON-объекта")


# ---------------------------------------------------------------- нормализация


def normalize_text(value: Any) -> str:
    text = str(value).lower().replace("ё", "е")
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def title_similarity(a: Any, b: Any) -> float:
    left, right = normalize_text(a), normalize_text(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("#").isdigit():
        return int(value.strip().lstrip("#"))
    if isinstance(value, dict):
        return as_int(value.get("id"))
    return None


def _id_set(values: Any) -> set[int] | None:
    if values is None:
        return set()
    if not isinstance(values, list):
        return None
    ids = {as_int(item) for item in values}
    return None if None in ids else ids


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _parse_moment(value: Any) -> date | datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    try:
        if len(value) == 10:
            return date.fromisoformat(value)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def compare_deadline(expected: Any, actual: Any) -> tuple[bool, str]:
    if expected is None:
        if _is_empty(actual):
            return True, ""
        return False, "срок придуман: в запросе его не было"
    if _is_empty(actual):
        return False, "срок не указан"

    want, got = _parse_moment(expected), _parse_moment(actual)
    if got is None:
        return False, "срок не удалось разобрать как дату ISO 8601"
    if isinstance(want, datetime):
        if not isinstance(got, datetime):
            return False, "потеряно время срока"
        if got.tzinfo is None:
            return False, "время без смещения часового пояса"
        return (want == got, "" if want == got else "другой момент времени")
    if isinstance(got, datetime):
        if got.date() == want:
            return False, "добавлено время, которого не было в запросе"
        return False, "другая дата"
    return (want == got, "" if want == got else "другая дата")


# ---------------------------------------------------------------- проверки полей


def _pair_by_title(expected: list[dict], actual: list[Any]) -> dict[int, int]:
    """Сопоставляет эталонные и полученные задачи по похожести названий (жадно)."""
    scored = []
    for i, want in enumerate(expected):
        for j, got in enumerate(actual):
            title = got.get("title") if isinstance(got, dict) else got
            scored.append((title_similarity(want.get("title"), title), i, j))
    pairs: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in sorted(scored, key=lambda item: -item[0]):
        if i not in pairs and j not in used:
            pairs[i] = j
            used.add(j)
    return pairs


def _check_tasks(expected: list[dict], output: dict[str, Any]) -> list[Check]:
    actual = output.get("tasks")
    if actual is None:
        actual = []
    if not isinstance(actual, list):
        return [Check("tasks", False, expected, actual, "поле tasks должно быть списком")]

    if not expected:
        ok = not actual
        note = "" if ok else f"выдуманы задачи: {len(actual)}"
        return [Check("tasks", ok, [], actual, note, critical=True)]

    checks = [
        Check(
            "tasks.count",
            len(actual) == len(expected),
            len(expected),
            len(actual),
            "" if len(actual) <= len(expected) else "лишние задачи",
            critical=len(actual) > len(expected),
        )
    ]
    pairs = _pair_by_title(expected, actual)
    for i, want in enumerate(expected):
        label = f"tasks[{want.get('title')}]"
        got = actual[pairs[i]] if i in pairs else None
        if not isinstance(got, dict):
            for part in ("title", "priority", "deadline"):
                checks.append(Check(f"{label}.{part}", False, want.get(part), None, "нет задачи"))
            continue
        similarity = title_similarity(want.get("title"), got.get("title"))
        checks.append(
            Check(
                f"{label}.title",
                similarity >= TITLE_THRESHOLD,
                want.get("title"),
                got.get("title"),
                f"похожесть {similarity:.2f}",
            )
        )
        want_priority = want.get("priority")
        got_priority = got.get("priority")
        priority_ok = (want_priority is None and _is_empty(got_priority)) or (
            want_priority is not None
            and normalize_text(want_priority) == normalize_text(got_priority)
        )
        checks.append(Check(f"{label}.priority", priority_ok, want_priority, got_priority))
        deadline_ok, note = compare_deadline(want.get("deadline"), got.get("deadline"))
        checks.append(
            Check(f"{label}.deadline", deadline_ok, want.get("deadline"), got.get("deadline"), note)
        )
    return checks


def _check_dependencies(expected: list[dict], output: dict[str, Any]) -> list[Check]:
    actual = output.get("dependencies") or []
    if not isinstance(actual, list):
        return [Check("dependencies", False, expected, actual, "должен быть список")]
    checks = [Check("dependencies.count", len(actual) == len(expected), len(expected), len(actual))]
    pairs = [item for item in actual if isinstance(item, dict)]
    for want in expected:
        label = f"dependencies[{want.get('task')} <- {want.get('depends_on')}]"
        found, reversed_found = False, False
        for got in pairs:
            direct = min(
                title_similarity(want.get("task"), got.get("task")),
                title_similarity(want.get("depends_on"), got.get("depends_on")),
            )
            reverse = min(
                title_similarity(want.get("task"), got.get("depends_on")),
                title_similarity(want.get("depends_on"), got.get("task")),
            )
            found = found or direct >= TITLE_THRESHOLD
            reversed_found = reversed_found or reverse >= TITLE_THRESHOLD
        note = "" if found else ("направление перепутано" if reversed_found else "связь не найдена")
        checks.append(Check(label, found, want, actual, note))
    return checks


def _check_updates(expected: dict[str, Any], output: dict[str, Any]) -> list[Check]:
    actual = output.get("updates")
    if actual is None:
        actual = {}
    if not isinstance(actual, dict):
        return [Check("updates", False, expected, actual, "поле updates должно быть объектом")]

    checks = []
    for key, want in expected.items():
        got = actual.get(key)
        label = f"updates.{key}"
        if key == "deadline":
            ok, note = compare_deadline(want, got)
            checks.append(Check(label, ok, want, got, note))
        elif isinstance(want, list):
            want_ids, got_ids = _id_set(want), _id_set(got)
            if want_ids is not None and got_ids is not None:
                checks.append(Check(label, want_ids == got_ids, want, got))
            else:
                ok = sorted(map(normalize_text, want)) == sorted(map(normalize_text, got or []))
                checks.append(Check(label, ok, want, got))
        else:
            checks.append(Check(label, normalize_text(want) == normalize_text(got), want, got))

    extra = sorted(
        key for key, value in actual.items() if key not in expected and not _is_empty(value)
    )
    checks.append(
        Check(
            "updates.extra",
            not extra,
            [],
            extra,
            "" if not extra else "лишние изменения, о которых не просили",
        )
    )
    return checks


def _check_bool(key: str, expected: bool, output: dict[str, Any]) -> Check:
    if key not in output:
        actual, ok, note = None, False, "поле отсутствует"
    else:
        actual = output[key]
        ok = actual is expected
        note = "" if ok else "неверное значение"
    critical = (key == "requires_confirmation" and expected is True) or (
        key == "should_execute" and expected is False
    )
    return Check(key, ok, expected, actual, note, critical=critical and not ok)


def _check_field(key: str, expected: Any, output: dict[str, Any]) -> list[Check]:
    actual = output.get(key)
    if key == "tasks":
        return _check_tasks(expected, output)
    if key == "dependencies":
        return _check_dependencies(expected, output)
    if key == "updates":
        return _check_updates(expected, output)
    if key in _BOOL_FIELDS:
        return [_check_bool(key, expected, output)]
    if key == "task_id":
        return [Check(key, as_int(actual) == expected, expected, actual)]
    if key in _SET_FIELDS:
        got_ids = _id_set(actual)
        ok = got_ids is not None and got_ids == set(expected)
        critical = key == "matching_tasks" and not expected and not ok
        note = "выдуманы задачи" if critical else ""
        return [Check(key, ok, expected, actual, note, critical=critical)]
    if key == "ordered_task_ids":
        got = [as_int(item) for item in actual] if isinstance(actual, list) else None
        return [Check(key, got == expected, expected, actual)]
    if key in _SUBSET_FIELDS:
        got = {normalize_text(item) for item in actual} if isinstance(actual, list) else set()
        missing = [item for item in expected if normalize_text(item) not in got]
        note = f"не хватает: {', '.join(missing)}" if missing else ""
        return [Check(key, not missing, expected, actual, note)]
    if isinstance(expected, str):
        return [Check(key, normalize_text(expected) == normalize_text(actual), expected, actual)]
    return [Check(key, expected == actual, expected, actual)]


# ---------------------------------------------------------------- выдуманные ID


def _collect_ids(value: Any, key: str = "") -> set[int]:
    ids: set[int] = set()
    if isinstance(value, dict):
        for name, item in value.items():
            ids |= _collect_ids(item, name)
    elif isinstance(value, list):
        for item in value:
            ids |= _collect_ids(item, key)
    elif key in _ID_FIELDS or key in _UPDATE_ID_FIELDS or key in {"id", "dependency_ids"}:
        number = as_int(value)
        if number is not None:
            ids.add(number)
    return ids


def referenced_ids(output: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for key in _ID_FIELDS:
        ids |= _collect_ids(output.get(key), key)
    updates = output.get("updates")
    if isinstance(updates, dict):
        for key in _UPDATE_ID_FIELDS:
            ids |= _collect_ids(updates.get(key), key)
    return ids


def _check_invented_ids(case: dict[str, Any], output: dict[str, Any]) -> Check:
    """ID из ответа должны быть в контексте, в запросе пользователя (#123) или в эталоне.

    Для create_task поля с ID существующих задач не проверяются: исполнитель при создании их
    не читает, а модели часто нумеруют в них сами новые задачи (ordered_task_ids: [1, 2]).
    Это не ссылка на несуществующую задачу и последствий не имеет.
    """
    if output.get("intent") == "create_task":
        return Check("hallucination.ids", True, [], [], "для create_task не проверяется")
    context = case.get("context") or {}
    known = _collect_ids(context.get("tasks") or [])
    known |= {int(number) for number in _TASK_REF_RE.findall(case.get("input", ""))}
    known |= _collect_ids(case.get("expected") or {})
    invented = sorted(referenced_ids(output) - known)
    note = f"ссылки на несуществующие задачи: {invented}" if invented else ""
    return Check("hallucination.ids", not invented, [], invented, note, critical=bool(invented))


# ---------------------------------------------------------------- итог по кейсу


def score_case(case: dict[str, Any], output: dict[str, Any] | None) -> CaseScore:
    """Сравнивает разобранный ответ модели с полем expected тест-кейса."""
    expected = case["expected"]
    if output is None:
        return CaseScore([Check("json", False, "JSON-объект", None, "ответ не разобран")])
    checks: list[Check] = []
    for key, value in expected.items():
        checks.extend(_check_field(key, value, output))
    checks.append(_check_invented_ids(case, output))
    return CaseScore(checks)
