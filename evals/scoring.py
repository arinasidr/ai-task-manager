from collections import Counter
from datetime import UTC, datetime


def score_plan(case: dict, run: dict) -> dict:
    expected = case["expected"]["plan"]
    plan = run["result"].get("plan") or {}

    titles_by_id = {task["id"]: task["title"] for task in run["before"]["tasks"]}
    actual_order = [titles_by_id.get(task_id) for task_id in plan.get("ordered_task_ids", [])]

    expected_titles = expected["task_titles"]
    positions = {title: index for index, title in enumerate(actual_order)}

    checks = {
        "all_tasks": (
            len(actual_order) == len(expected_titles) and set(actual_order) == set(expected_titles)
        ),
        "dependencies": all(
            edge["before"] in positions
            and edge["after"] in positions
            and positions[edge["before"]] < positions[edge["after"]]
            for edge in expected["precedence"]
        ),
        "warnings": set(plan.get("warnings") or []) == set(expected.get("warnings", [])),
        "db_unchanged": run["before"] == run["after"],
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "actual_order": actual_order,
    }


def _title(value: str) -> str:
    return " ".join(value.casefold().replace("ё", "е").split())


def _deadline(value: str | None) -> str | None:
    if value is None or "T" not in value:
        return value
    return datetime.fromisoformat(value).astimezone(UTC).isoformat()


def _tasks(items: list[dict]) -> Counter:
    return Counter(
        (_title(item["title"]), item.get("priority"), _deadline(item.get("deadline")))
        for item in items
    )


def _dependencies(items: list[dict]) -> Counter:
    return Counter((_title(item["task"]), _title(item["depends_on"])) for item in items)


def _updates(value: dict | None) -> dict:
    result = {
        key: item
        for key, item in (value or {}).items()
        if item is not None and item != [] and item != ""
    }
    if "deadline" in result:
        result["deadline"] = _deadline(result["deadline"])
    return result


def score_analysis(case: dict, run: dict) -> dict:
    expected = case["expected"]["analysis"]
    actual = run["result"].get("analysis") or {}

    checks = {"intent": actual.get("intent") == expected["intent"]}

    if "tasks" in expected:
        checks["tasks"] = _tasks(actual.get("tasks") or []) == _tasks(expected["tasks"])
    if "dependencies" in expected:
        checks["dependencies"] = _dependencies(actual.get("dependencies") or []) == _dependencies(
            expected["dependencies"]
        )
    if "task_id" in expected:
        checks["task_id"] = actual.get("task_id") == expected["task_id"]
    if "task_ids" in expected:
        checks["task_ids"] = sorted(actual.get("task_ids") or []) == sorted(expected["task_ids"])
    if "updates" in expected:
        checks["updates"] = _updates(actual.get("updates")) == _updates(expected["updates"])

    return {"passed": all(checks.values()), "checks": checks}


def score_db(case: dict, run: dict) -> dict:
    expected = case["expected"]
    before = run["before"]
    after = run["after"]

    checks = {
        "confirmation_correct": bool(run["result"].get("action_id"))
        == expected.get("pending_action", False),
    }
    if case.get("follow_up") is not None:
        follow_up_result = run["follow_up_result"] or {}
        checks["unchanged_before_confirmation"] = before == run["after_request"]
        checks["follow_up_status"] = follow_up_result.get("status") == expected["follow_up_status"]
        checks["follow_up_applied"] = (
            follow_up_result.get("applied") is expected["follow_up_applied"]
        )

    if expected.get("db_unchanged"):
        checks["db_unchanged"] = before == after
        return {"passed": all(checks.values()), "checks": checks}

    wanted = expected["final_db"]
    wanted_tasks = wanted["tasks"]
    actual_tasks = after["tasks"]

    checks["task_titles"] = Counter(_title(task["title"]) for task in actual_tasks) == Counter(
        _title(task["title"]) for task in wanted_tasks
    )

    actual_by_title = {_title(task["title"]): task for task in actual_tasks}
    fields = {field for task in wanted_tasks for field in task if field != "title"}

    for field in sorted(fields):
        checks[field] = all(
            _title(task["title"]) in actual_by_title
            and actual_by_title[_title(task["title"])].get(field) == task[field]
            for task in wanted_tasks
            if field in task
        )

    titles_by_id = {task["id"]: task["title"] for task in actual_tasks}
    actual_dependencies = [
        {
            "task": titles_by_id[link["successor_task_id"]],
            "depends_on": titles_by_id[link["predecessor_task_id"]],
        }
        for link in after["dependencies"]
    ]
    checks["dependencies"] = _dependencies(actual_dependencies) == _dependencies(
        wanted["dependencies"]
    )
    checks["preferences_unchanged"] = before["preferences"] == after["preferences"]

    return {"passed": all(checks.values()), "checks": checks}


def score_case(case: dict, run: dict) -> dict:
    """Собрать проверки разбора, БД и, если есть, плана."""
    sections = {
        "analysis": score_analysis(case, run),
        "database": score_db(case, run),
    }
    if "plan" in case["expected"]:
        sections["plan"] = score_plan(case, run)

    return {
        "passed": all(section["passed"] for section in sections.values()),
        "sections": sections,
    }
