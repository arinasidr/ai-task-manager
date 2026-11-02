import csv
import json
from pathlib import Path

from evals.run import load_cases

SOURCES = (
    Path("evals/results/20261102-190655-rescored.jsonl"),
    Path("evals/results/20261102-192436.jsonl"),
)
PACKET = Path("evals/results/human_review.md")
SCORES = Path("evals/results/human_scores.csv")
CRITERIA = ("coverage", "order", "constraints", "explanation", "restraint")


def main() -> None:
    records = {}
    for source in SOURCES:
        for line in source.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            records[record["case"]["id"]] = record

    cases = [case for case in load_cases() if case.get("judge")]
    if len(cases) != 10:
        raise ValueError(f"Ожидалось 10 сценариев, найдено {len(cases)}")
    if PACKET.exists() or SCORES.exists():
        raise FileExistsError("Файлы оценки уже существуют")

    pages = [
        "# Ручная оценка планов A16",
        "",
        "Оцени каждый критерий от 0 до 2 по judge_rubric.md. Запиши баллы в human_scores.csv.",
        "",
    ]

    with SCORES.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("case_id", *CRITERIA, "comment"))

        for case in cases:
            case_id = case["id"]
            record = records[case_id]
            run = record["run"]
            result = run["result"]
            tasks = run["before"]["tasks"]
            titles_by_id = {task["id"]: task["title"] for task in tasks}
            plan = result.get("plan") or {}

            view = {
                "case_id": case_id,
                "request": case["message"],
                "now": case["now"],
                "tasks": [
                    {key: task[key] for key in ("id", "title", "status", "priority", "due_date")}
                    for task in tasks
                ],
                "dependencies": case["initial_db"]["dependencies"],
                "preferences": case["initial_db"].get("preferences", []),
                "proposed_order": [
                    titles_by_id.get(task_id, f"Неизвестный ID {task_id}")
                    for task_id in plan.get("ordered_task_ids", [])
                ],
                "recommendations": plan.get("recommendations", []),
                "warnings": plan.get("warnings", []),
                "reply": result["reply"],
            }
            pages.extend(
                [
                    f"## {case_id}",
                    "",
                    "```json",
                    json.dumps(view, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
            writer.writerow((case_id, "", "", "", "", "", ""))

    PACKET.write_text("\n".join(pages), encoding="utf-8")
    print("Материалы:", PACKET)
    print("Таблица оценок:", SCORES)


if __name__ == "__main__":
    main()
