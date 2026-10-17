"""Сводные метрики по прогону одной модели."""

import statistics
from collections import defaultdict
from typing import Any


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(share * (len(ordered) - 1))))
    return ordered[index]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Считает метрики по записям из responses.jsonl.

    Записи с ошибкой движка (таймаут, сервер недоступен) входят в точность как непройденные,
    но не входят в метрики скорости.
    """
    total = len(records)
    if not total:
        return {"runs": 0}
    ok_runs = [r for r in records if not r.get("error")]
    latencies = [r["latency_s"] for r in ok_runs if r.get("latency_s") is not None]
    gen_tps = [r["gen_tps"] for r in ok_runs if r.get("gen_tps")]
    prompt_tps = [r["prompt_tps"] for r in ok_runs if r.get("prompt_tps")]
    out_tokens = [r["output_tokens"] for r in ok_runs if r.get("output_tokens")]

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
        by_case[record["case_id"]].append(record)

    critical = [
        {"case_id": r["case_id"], "repeat": r.get("repeat", 0), "fields": r["critical"]}
        for r in records
        if r.get("critical")
    ]

    summary: dict[str, Any] = {
        "runs": total,
        "cases": len(by_case),
        "accuracy": _round(sum(r["passed"] for r in records) / total),
        "mean_score": _round(statistics.fmean(r["score"] for r in records)),
        "json_parsed_rate": _round(sum(r["json_parsed"] for r in records) / total),
        "json_strict_rate": _round(sum(r["json_strict"] for r in records) / total),
        "schema_valid_rate": _round(sum(r["schema_valid"] for r in records) / total),
        "errors": sum(bool(r.get("error")) for r in records),
        "truncated": sum(r.get("finish_reason") == "length" for r in records),
        "critical_failures": len(critical),
        "critical_details": critical,
        "by_category": {
            name: {
                "runs": len(items),
                "accuracy": _round(sum(r["passed"] for r in items) / len(items)),
                "mean_score": _round(statistics.fmean(r["score"] for r in items)),
            }
            for name, items in sorted(by_category.items())
        },
        "latency_s": {
            "mean": _round(_mean(latencies), 2),
            "median": _round(statistics.median(latencies), 2) if latencies else None,
            "p90": _round(_percentile(latencies, 0.9), 2),
            "max": _round(max(latencies), 2) if latencies else None,
        },
        "gen_tokens_per_s": _round(_mean(gen_tps), 1),
        "prompt_tokens_per_s": _round(_mean(prompt_tps), 1),
        "mean_output_tokens": _round(_mean(out_tokens), 1),
        "failed_cases": sorted(
            case for case, items in by_case.items() if not all(r["passed"] for r in items)
        ),
    }

    repeats = {len(items) for items in by_case.values()}
    if max(repeats) > 1:
        same_verdict = sum(len({r["passed"] for r in items}) == 1 for items in by_case.values())
        same_answer = sum(len({r["answer_key"] for r in items}) == 1 for items in by_case.values())
        summary["consistency"] = {
            "same_verdict_rate": _round(same_verdict / len(by_case)),
            "same_answer_rate": _round(same_answer / len(by_case)),
        }
    return summary
