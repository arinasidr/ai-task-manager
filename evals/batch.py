import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from evals.run import load_cases, run_case
from evals.scoring import score_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск сценариев A16")
    parser.add_argument("--only", nargs="+", help="ID сценариев для выборочного запуска")
    parser.add_argument("--output", type=Path, help="Путь к файлу результатов")
    args = parser.parse_args()

    cases = load_cases()
    if args.only:
        selected = set(args.only)
        unknown = selected - {case["id"] for case in cases}
        if unknown:
            parser.error(f"Неизвестные сценарии: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["id"] in selected]

    output = args.output or Path("evals/results") / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + ".jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as file:
        for case in cases:
            try:
                run = run_case(case)
                record = {
                    "case": case,
                    "run": run,
                    "score": score_case(case, run),
                }
            except Exception as exc:
                record = {
                    "case": case,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "score": {"passed": False},
                }

            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            status = "PASS" if record["score"]["passed"] else "FAIL"
            print(f"{case['id']}: {status}", flush=True)

    print(f"Результаты: {output}")


if __name__ == "__main__":
    main()
