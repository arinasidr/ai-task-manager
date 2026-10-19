"""Пересчёт оценок по уже сохранённым ответам, без повторного запуска моделей.

Нужен, когда меняются правила проверки (SCORING_VERSION в scoring.py) или эталоны в
cases.jsonl. Сырые ответы моделей, задержки и скорость не меняются, пересчитываются
только проверки и сводные метрики.

  python -m benchmarks.rescore                       # все папки в benchmarks/results
  python -m benchmarks.rescore benchmarks/results/20261019-210650_qwen3-1.7b
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.run import DEFAULT_CASES, DEFAULT_RESULTS, evaluate, load_cases, sha256  # noqa: E402
from benchmarks.scoring import SCORING_VERSION  # noqa: E402
from benchmarks.summary import summarize  # noqa: E402

# Поля записи, которые получены от движка и при пересчёте сохраняются как есть
_KEEP = (
    "error",
    "thinking",
    "latency_s",
    "prompt_tokens",
    "output_tokens",
    "gen_tps",
    "prompt_tps",
    "tps_estimated",
    "finish_reason",
)
# Поля сводки, которые описывают прогон, а не качество ответов
_SUMMARY_META = (
    "model",
    "status",
    "engine",
    "machine",
    "settings",
    "system_prompt_sha256",
    "warmup",
    "memory",
    "finished_at",
    "results_dir",
)


def rescore_dir(run_dir: Path, cases: dict[str, dict[str, Any]], cases_hash: str) -> dict[str, Any]:
    responses = run_dir / "responses.jsonl"
    old_records = [
        json.loads(line) for line in responses.read_text(encoding="utf-8").splitlines() if line
    ]
    records = []
    for old in old_records:
        case = cases.get(old["case_id"])
        if case is None:
            raise SystemExit(f"{run_dir}: теста {old['case_id']} больше нет в cases.jsonl")
        record = evaluate(case, old.get("raw_text") or "", old.get("repeat", 0))
        record.update({key: old.get(key) for key in _KEEP})
        if record["error"]:
            record.update({"json_parsed": False, "schema_valid": False})
        records.append(record)
    responses.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )

    summary_path = run_dir / "summary.json"
    old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = summarize(records)
    summary.update({key: old_summary.get(key) for key in _SUMMARY_META})
    summary["cases_sha256"] = cases_hash
    summary["scoring_version"] = SCORING_VERSION
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"old": old_summary, "new": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.rescore")
    parser.add_argument("dirs", nargs="*", type=Path, help="папки прогонов (по умолчанию все)")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = parser.parse_args(argv)

    cases_text = args.cases.read_text(encoding="utf-8")
    cases = {case["id"]: case for case in load_cases(args.cases)}
    dirs = args.dirs or sorted(p.parent for p in DEFAULT_RESULTS.glob("*/responses.jsonl"))
    if not dirs:
        print("Нет сохранённых прогонов")
        return 1
    for run_dir in dirs:
        result = rescore_dir(run_dir, cases, sha256(cases_text))
        old, new = result["old"], result["new"]
        print(
            f"{run_dir.name}: точность {old.get('accuracy', 0):.0%} -> {new['accuracy']:.0%}, "
            f"критичных ошибок {old.get('critical_failures', 0)} -> {new['critical_failures']} "
            f"(оценка v{old.get('scoring_version', 1)} -> v{SCORING_VERSION})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
