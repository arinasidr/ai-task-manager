"""Негативные диалоги с настоящими локальными моделями."""

import argparse
import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_task_manager.config import Settings
from ai_task_manager.db.database import connect
from ai_task_manager.memory.checkpoint import open_checkpointer
from ai_task_manager.memory.inference import same_value
from ai_task_manager.workflow.service import create_workflow, run_chat
from evals.run import seed_case, snapshot_db

CASES_PATH = Path(__file__).with_name("negative_cases.json")
RESULTS_DIR = Path(__file__).parent / "results"


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def run_case(case: dict) -> dict:
    """Каждый диалог выполняется на собственной временной базе."""
    with TemporaryDirectory() as folder:
        db_path = Path(folder) / "tasks.db"
        seed_case(db_path, case)
        before = snapshot_db(db_path)

        settings = Settings(
            database_path=str(db_path),
            timezone=case["timezone"],
            log_dir=None,
            phoenix_endpoint=None,
        )
        now = datetime.fromisoformat(case["now"])
        turns = []
        conversation_id = None

        with open_checkpointer(Path(folder) / "history.db") as checkpointer:
            graph = create_workflow(settings, checkpointer=checkpointer)
            for message in case["messages"]:
                result = run_chat(
                    graph,
                    settings,
                    message,
                    conversation_id,
                    now=now,
                )
                conversation_id = result.conversation_id
                turns.append(
                    {
                        "message": message,
                        "result": result.model_dump(mode="json"),
                        "db": snapshot_db(db_path),
                    }
                )

        with closing(connect(db_path)) as db:
            operations = [
                dict(row)
                for row in db.execute(
                    "SELECT action_id, status FROM operations ORDER BY created_at, action_id"
                )
            ]

        after = snapshot_db(db_path)

    return {
        "before": before,
        "turns": turns,
        "after": after,
        "operations": operations,
    }


def _tasks_and_links(snapshot: dict) -> tuple[list[dict], list[dict]]:
    return snapshot["tasks"], snapshot["dependencies"]


def _confirmed_preferences(snapshot: dict) -> dict:
    return {
        row["kind"]: json.loads(row["value_json"])
        for row in snapshot["preferences"]
        if row["status"] == "confirmed"
    }


def score_case(case: dict, run: dict) -> dict:
    expected = case["expected"]
    turns = run["turns"]
    checks: dict[str, bool] = {}
    allowed_errors = expected.get("error_prefixes", [None] * len(turns))
    checks["error_count"] = len(allowed_errors) == len(turns)

    for index, turn in enumerate(turns):
        error = turn["result"].get("error")
        allowed = allowed_errors[index] if index < len(allowed_errors) else None
        checks[f"error_{index + 1}"] = (
            error is None
            if allowed is None
            else isinstance(error, str) and error.startswith(allowed)
        )
        checks[f"decision_error_{index + 1}"] = turn["result"].get("decision_error") is None
    checks["trajectory_recorded"] = all(bool(turn["result"].get("trajectory")) for turn in turns)

    for field, key, transform in (
        ("intents", "analysis", lambda value: (value or {}).get("intent")),
        ("action_flags", "action_id", bool),
        ("preference_flags", "preference_id", bool),
    ):
        if field not in expected:
            continue
        wanted = expected[field]
        checks[f"{field}_count"] = len(wanted) == len(turns)
        for index, value in enumerate(wanted):
            if value is None or index >= len(turns):
                continue
            actual = transform(turns[index]["result"].get(key))
            checks[f"{field}_{index + 1}"] = actual == value

    if "reply_contains" in expected:
        wanted = expected["reply_contains"]
        checks["reply_count"] = len(wanted) == len(turns)
        for index, fragment in enumerate(wanted):
            if fragment is None or index >= len(turns):
                continue
            reply = turns[index]["result"].get("reply") or ""
            checks[f"reply_{index + 1}"] = fragment.casefold() in reply.casefold()
    if "reply_equals" in expected:
        wanted = expected["reply_equals"]
        checks["reply_equals_count"] = len(wanted) == len(turns)
        for index, exact in enumerate(wanted):
            if exact is None or index >= len(turns):
                continue
            reply = turns[index]["result"].get("reply") or ""
            checks[f"reply_equals_{index + 1}"] = reply.strip() == exact

    if expected.get("tasks_unchanged_each_turn"):
        original = _tasks_and_links(run["before"])
        checks["tasks_unchanged_each_turn"] = all(
            _tasks_and_links(turn["db"]) == original for turn in turns
        )

    if "task_titles" in expected:
        actual = [task["title"].strip().casefold() for task in run["after"]["tasks"]]
        wanted = [title.strip().casefold() for title in expected["task_titles"]]
        checks["task_titles"] = sorted(actual) == sorted(wanted)

    if "operation_statuses" in expected:
        actual = [operation["status"] for operation in run["operations"]]
        checks["operation_statuses"] = actual == expected["operation_statuses"]

    if "preference_statuses" in expected:
        actual = [row["status"] for row in run["after"]["preferences"]]
        checks["preference_statuses"] = actual == expected["preference_statuses"]

    if "confirmed_preferences" in expected:
        actual = _confirmed_preferences(run["after"])
        wanted = expected["confirmed_preferences"]
        checks["confirmed_preference_kinds"] = set(actual) == set(wanted)
        for kind, value in wanted.items():
            checks[f"preference_{kind}"] = kind in actual and same_value(kind, actual[kind], value)

    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Негативные Evals")
    parser.add_argument("--only", nargs="+", help="ID сценариев для выборочного запуска")
    parser.add_argument("--output", type=Path, help="Файл результатов JSONL")
    args = parser.parse_args()

    cases = load_cases()
    if args.only:
        selected = set(args.only)
        unknown = selected - {case["id"] for case in cases}
        if unknown:
            parser.error(f"Неизвестные сценарии: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["id"] in selected]

    output = args.output or RESULTS_DIR / (
        "a17-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f") + ".jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as file:
        for case in cases:
            try:
                run = run_case(case)
                score = score_case(case, run)
                record = {"case": case, "run": run, "score": score}
            except Exception as exc:
                record = {
                    "case": case,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "score": {"passed": False, "checks": {}},
                }

            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()

            failed = [name for name, passed in record["score"]["checks"].items() if not passed]
            status = "PASS" if record["score"]["passed"] else "FAIL"
            print(f"{case['id']}: {status}", flush=True)
            if failed:
                print(f"  Не прошли проверки: {', '.join(failed)}", flush=True)
            if "error" in record:
                print(f"  Ошибка: {record['error']}", flush=True)

    print(f"Результаты: {output}")


if __name__ == "__main__":
    main()
