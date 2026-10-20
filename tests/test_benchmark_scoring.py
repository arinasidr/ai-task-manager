"""Проверки оценщика бенчмарка: без LLM и без сети."""

import copy
import json
from pathlib import Path

import pytest

from benchmarks.schema import OUTPUT_SCHEMA, validate
from benchmarks.scoring import compare_deadline, parse_model_output, score_case

ROOT = Path(__file__).resolve().parent.parent
CASES = [
    json.loads(line)
    for line in (ROOT / "benchmarks" / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
CASES_BY_ID = {case["id"]: case for case in CASES}
REQUIRED_DEFAULTS = {"warnings": [], "requires_confirmation": False, "should_execute": True}


def complete(expected: dict) -> dict:
    """Эталон, дополненный обязательными полями схемы (как ответила бы идеальная модель)."""
    answer = copy.deepcopy(expected)
    for key, value in REQUIRED_DEFAULTS.items():
        answer.setdefault(key, value)
    if answer["requires_confirmation"]:
        answer["should_execute"] = False
    return answer


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_reference_answer_passes_its_own_case(case: dict) -> None:
    result = score_case(case, complete(case["expected"]))
    assert result.passed, result.failed()
    assert result.critical_failures == []


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_reference_answer_fits_output_schema(case: dict) -> None:
    # если эталон не укладывается в схему, модель в режиме schema физически не сможет его выдать
    assert validate(complete(case["expected"]), OUTPUT_SCHEMA) == []


def test_system_prompt_does_not_leak_test_inputs() -> None:
    prompt = (ROOT / "benchmarks" / "prompts" / "system.md").read_text(encoding="utf-8")
    leaked = [case["id"] for case in CASES if case["input"] in prompt]
    assert leaked == []


# ---------------------------------------------------------------- разбор ответа


def test_parse_plain_json_is_strict() -> None:
    parsed = parse_model_output('{"intent": "get_tasks"}')
    assert parsed.data == {"intent": "get_tasks"}
    assert parsed.strict_json


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"intent": "get_tasks"}\n```',
        'Вот ответ: {"intent": "get_tasks"} Готово.',
        '<think>сначала подумаю</think>\n{"intent": "get_tasks"}',
    ],
)
def test_parse_recovers_wrapped_json(text: str) -> None:
    parsed = parse_model_output(text)
    assert parsed.data == {"intent": "get_tasks"}


def test_parse_think_block_then_clean_json_is_strict() -> None:
    assert parse_model_output('<think>...</think>{"intent": "x"}').strict_json


@pytest.mark.parametrize("text", ["", "не знаю", "[1, 2]", '{"intent": '])
def test_parse_rejects_non_objects(text: str) -> None:
    parsed = parse_model_output(text)
    assert parsed.data is None
    assert parsed.error


def test_unparsed_answer_scores_zero() -> None:
    result = score_case(CASES_BY_ID["TEST_01"], None)
    assert not result.passed
    assert result.score == 0


# ---------------------------------------------------------------- сроки


@pytest.mark.parametrize(
    ("expected", "actual", "ok"),
    [
        (None, None, True),
        (None, "", True),
        (None, "2026-10-15", False),
        ("2026-10-15", "2026-10-15", True),
        ("2026-10-15", "2026-10-16", False),
        ("2026-10-15", "2026-10-15T00:00:00+03:00", False),
        ("2026-10-20T18:30:00+03:00", "2026-10-20T18:30+03:00", True),
        ("2026-10-20T18:30:00+03:00", "2026-10-20T15:30:00Z", True),
        ("2026-10-20T18:30:00+03:00", "2026-10-20T18:30:00", False),
        ("2026-10-20T18:30:00+03:00", "2026-10-20", False),
        ("2026-10-20T18:30:00+03:00", "20 октября", False),
    ],
)
def test_compare_deadline(expected, actual, ok: bool) -> None:
    assert compare_deadline(expected, actual)[0] is ok


# ---------------------------------------------------------------- типичные ошибки моделей


def test_titles_ignore_case_yo_and_quotes() -> None:
    answer = complete(CASES_BY_ID["TEST_01"]["expected"])
    answer["tasks"][0]["title"] = "«подготовить ОТЧЕТ»"
    assert score_case(CASES_BY_ID["TEST_01"], answer).passed


def test_task_order_does_not_matter() -> None:
    answer = complete(CASES_BY_ID["TEST_02"]["expected"])
    answer["tasks"].reverse()
    assert score_case(CASES_BY_ID["TEST_02"], answer).passed


def test_extra_task_is_critical() -> None:
    answer = complete(CASES_BY_ID["TEST_02"]["expected"])
    answer["tasks"].append({"title": "Позвонить маме", "priority": None, "deadline": None})
    result = score_case(CASES_BY_ID["TEST_02"], answer)
    assert not result.passed
    assert "tasks.count" in result.critical_failures


def test_invented_default_priority_fails() -> None:
    answer = complete(CASES_BY_ID["TEST_01"]["expected"])
    answer["tasks"][0]["priority"] = "medium"
    assert not score_case(CASES_BY_ID["TEST_01"], answer).passed


def test_hallucinated_tasks_on_empty_list_are_critical() -> None:
    answer = complete(CASES_BY_ID["TEST_19"]["expected"])
    answer["tasks"] = [{"title": "Купить молоко", "priority": None, "deadline": None}]
    assert "tasks" in score_case(CASES_BY_ID["TEST_19"], answer).critical_failures


def test_missing_list_counts_as_empty_when_nothing_expected() -> None:
    answer = complete(CASES_BY_ID["TEST_19"]["expected"])
    del answer["tasks"]
    assert score_case(CASES_BY_ID["TEST_19"], answer).passed


def test_reversed_dependency_fails() -> None:
    case = CASES_BY_ID["TEST_09"]
    answer = complete(case["expected"])
    dep = answer["dependencies"][0]
    dep["task"], dep["depends_on"] = dep["depends_on"], dep["task"]
    result = score_case(case, answer)
    assert not result.passed
    assert any("направление" in check["note"] for check in result.failed())


def test_executing_dangerous_action_is_critical() -> None:
    case = CASES_BY_ID["TEST_30"]
    answer = complete(case["expected"])
    answer.update({"requires_confirmation": False, "should_execute": True})
    result = score_case(case, answer)
    assert set(result.critical_failures) == {"requires_confirmation", "should_execute"}


def test_unrequested_update_fields_fail() -> None:
    case = CASES_BY_ID["TEST_12"]
    answer = complete(case["expected"])
    answer["updates"]["status"] = "done"
    result = score_case(case, answer)
    assert [check["field"] for check in result.failed()] == ["updates.extra"]


def test_unknown_task_id_is_hallucination() -> None:
    case = CASES_BY_ID["TEST_17"]
    answer = complete(case["expected"])
    answer["ordered_task_ids"] = [1, 2, 3, 77]
    assert "hallucination.ids" in score_case(case, answer).critical_failures


def test_task_id_mentioned_by_user_is_not_hallucination() -> None:
    case = CASES_BY_ID["TEST_23"]
    answer = complete(case["expected"])
    assert "hallucination.ids" not in score_case(case, answer).critical_failures


def test_extra_warnings_are_allowed_missing_are_not() -> None:
    case = CASES_BY_ID["TEST_11"]
    answer = complete(case["expected"])
    answer["warnings"] = ["dependency_cycle", "ambiguous_request"]
    assert score_case(case, answer).passed
    answer["warnings"] = []
    assert not score_case(case, answer).passed


def test_numeric_string_ids_are_rejected() -> None:
    case = CASES_BY_ID["TEST_13"]
    answer = complete(case["expected"])
    answer["task_id"] = "14"
    assert "action.task_id" in score_case(case, answer).critical_failures


def test_update_task_requires_id_and_updates() -> None:
    case = CASES_BY_ID["TEST_13"]
    answer = complete(case["expected"])
    del answer["task_id"], answer["updates"]
    assert {"action.task_id", "action.updates"} <= set(score_case(case, answer).critical_failures)


def test_schema_rejects_bad_answer() -> None:
    errors = validate({"intent": "fly_to_moon", "warnings": "нет", "extra": 1})
    joined = " ".join(errors)
    assert "fly_to_moon" in joined
    assert "requires_confirmation" in joined
    assert "extra" in joined


def test_ids_in_create_task_are_not_hallucination() -> None:
    # модели нумеруют новые задачи в ordered_task_ids — при создании это поле не читается
    case = CASES_BY_ID["TEST_02"]
    answer = complete(case["expected"])
    answer["ordered_task_ids"] = [1, 2]
    result = score_case(case, answer)
    assert result.passed
    assert result.critical_failures == []
