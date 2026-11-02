import csv
import json
from collections import Counter
from math import ceil
from pathlib import Path
from statistics import median

from evals.run import load_cases
from evals.scoring import score_case

RESULTS = Path(__file__).parent / "results"
SOURCES = (
    "20261102-190655-rescored.jsonl",
    "20261102-192436.jsonl",
    "20261102-195654.jsonl",
    "20261102-200327.jsonl",
    "20261102-201228.jsonl",
)


def rate(values: list[bool]) -> dict:
    passed = sum(values)
    total = len(values)
    return {
        "passed": passed,
        "total": total,
        "percent": round(100 * passed / total, 1) if total else None,
    }


CRITERIA = ("coverage", "order", "constraints", "explanation", "restraint")


def judge_agreement(filename: str) -> dict:
    with (RESULTS / "human_scores.csv").open(encoding="utf-8", newline="") as file:
        human_rows = list(csv.DictReader(file))
    human = {row["case_id"]: row for row in human_rows}

    judge_rows = [
        json.loads(line) for line in (RESULTS / filename).read_text(encoding="utf-8").splitlines()
    ]
    judge = {row["case_id"]: row for row in judge_rows}

    if len(human_rows) != 10 or len(judge_rows) != 10 or set(human) != set(judge):
        raise ValueError(f"Ожидались одинаковые 10 оценок: {filename}")

    human_totals = {
        case_id: sum(int(row[key]) for key in CRITERIA) for case_id, row in human.items()
    }
    differences = {
        case_id: abs(human_totals[case_id] - judge[case_id]["total"]) for case_id in human
    }

    return {
        "cases": len(human),
        "human_mean": round(sum(human_totals.values()) / 10, 2),
        "judge_mean": round(sum(row["total"] for row in judge.values()) / 10, 2),
        "mean_absolute_difference": round(sum(differences.values()) / 10, 2),
        "exact_matches": sum(diff == 0 for diff in differences.values()),
        "within_one_point": sum(diff <= 1 for diff in differences.values()),
        "criterion_mean_absolute_difference": {
            key: round(
                sum(
                    abs(int(human[case_id][key]) - judge[case_id]["scores"][key])
                    for case_id in human
                )
                / 10,
                2,
            )
            for key in CRITERIA
        },
        "disagreement_at_least_three_points": [
            case_id for case_id, diff in differences.items() if diff >= 3
        ],
    }


def main() -> None:
    records = {}
    for name in SOURCES:
        for line in (RESULTS / name).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            records[record["case"]["id"]] = record

    cases = {case["id"]: case for case in load_cases()}
    if set(records) != set(cases):
        raise ValueError(
            f"Нет результатов: {sorted(set(cases) - set(records))}; "
            f"лишние: {sorted(set(records) - set(cases))}"
        )

    runs = {case_id: records[case_id]["run"] for case_id in cases}
    scores = {case_id: score_case(case, runs[case_id]) for case_id, case in cases.items()}
    durations = sorted(run["duration_seconds"] for run in runs.values())

    intent_checks = [score["sections"]["analysis"]["checks"]["intent"] for score in scores.values()]
    extraction_checks = [
        checks["tasks"]
        for score in scores.values()
        if (checks := score["sections"]["analysis"]["checks"]) and "tasks" in checks
    ]
    plan_checks = [
        score["sections"]["plan"]["passed"]
        for score in scores.values()
        if "plan" in score["sections"]
    ]
    confirmation_checks = [
        scores[case_id]["sections"]["database"]["passed"]
        for case_id, case in cases.items()
        if case["expected"].get("pending_action")
    ]

    agent_errors = Counter()
    retries = Counter()
    for run in runs.values():
        result = run["result"]
        agent_errors.update(result.get("agent_error_counts") or {})
        retries.update(result.get("model_retry_counts") or {})

    summary = {
        "task_success": rate([score["passed"] for score in scores.values()]),
        "failed_cases": [case_id for case_id, score in scores.items() if not score["passed"]],
        "intent_accuracy": rate(intent_checks),
        "task_extraction_accuracy": rate(extraction_checks),
        "plan_validity": rate(plan_checks),
        "confirmation_safety": rate(confirmation_checks),
        "chat_latency_seconds": {
            "p50": round(median(durations), 2),
            "p95": round(durations[ceil(0.95 * len(durations)) - 1], 2),
            "max": round(durations[-1], 2),
        },
        "workflow_errors": sum(bool(run["result"].get("error")) for run in runs.values()),
        "agent_errors": dict(agent_errors),
        "model_retries": dict(retries),
        "llm_judge": {
            "phi4_mini_pilot": judge_agreement("judge_phi4.jsonl"),
            "phi4_mini_same_rubric": judge_agreement("judge_phi4_neutral.jsonl"),
            "ministral_3": judge_agreement("judge_ministral.jsonl"),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
