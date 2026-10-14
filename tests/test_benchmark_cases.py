"""Страховка для общего набора эталонных запросов: файл валиден и id не повторяются."""

import json
from pathlib import Path

CASES_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "cases.jsonl"


def _load_cases() -> list[dict]:
    lines = CASES_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_cases_are_valid_and_unique() -> None:
    cases = _load_cases()
    assert cases, "benchmarks/cases.jsonl пуст"
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "id тестов должны быть уникальными"
    for case in cases:
        assert {"id", "category", "input", "expected"} <= case.keys(), case["id"]
