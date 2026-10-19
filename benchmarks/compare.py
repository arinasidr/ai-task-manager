"""Сравнительные таблицы по результатам прогонов (benchmarks/results/*/summary.json).

Качество и скорость выводятся отдельно: скорость зависит от компьютера, поэтому её сравнивают
только внутри одной машины, а качество — между всеми прогонами на одинаковых промпте и тестах.

  python -m benchmarks.compare                 # последний прогон каждой модели
  python -m benchmarks.compare --all           # все прогоны
  python -m benchmarks.compare --output benchmarks/results/comparison.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent


def load_summaries(results_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(results_dir.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("runs"):
            summaries.append(data)
    return summaries


def _run_key(summary: dict[str, Any]) -> tuple:
    settings = summary.get("settings") or {}
    machine = (summary.get("machine") or {}).get("label")
    return (
        summary.get("model"),
        machine,
        settings.get("format_mode"),
        settings.get("think"),
        summary.get("system_prompt_sha256"),
    )


def latest_only(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple, dict[str, Any]] = {}
    for summary in sorted(summaries, key=lambda s: s.get("finished_at") or ""):
        latest[_run_key(summary)] = summary
    return list(latest.values())


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def _num(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _gpu_share(memory: dict[str, Any]) -> float | None:
    if memory.get("gpu_share") is not None:
        return memory["gpu_share"]
    if memory.get("size_gb"):
        return round((memory.get("size_vram_gb") or 0) / memory["size_gb"], 2)
    return None


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return "Нет результатов. Сначала запустите python -m benchmarks.run --model <модель>."

    ordered = sorted(summaries, key=lambda s: (-(s.get("accuracy") or 0), s.get("model") or ""))
    parts = ["# Сравнение моделей на cases.jsonl", ""]

    versions = {s.get("scoring_version", 1) for s in ordered}
    if len(versions) > 1:
        parts += [
            "> Внимание: прогоны оценены разными версиями правил "
            f"({sorted(versions)}). Выполните python -m benchmarks.rescore.",
            "",
        ]

    prompts = {s.get("system_prompt_sha256", "")[:12] for s in ordered}
    cases = {s.get("cases_sha256", "")[:12] for s in ordered}
    if len(prompts) > 1 or len(cases) > 1:
        parts += [
            "> Внимание: прогоны сделаны с разными промптами или наборами тестов "
            f"(промпты {sorted(prompts)}, тесты {sorted(cases)}). Сравнивать качество нельзя.",
            "",
        ]

    quality_rows = []
    for s in ordered:
        engine = s.get("engine") or {}
        settings = s.get("settings") or {}
        status = s.get("status", "completed")
        quality_rows.append(
            [
                s["model"] if status == "completed" else f"{s['model']} ({status}, неполный)",
                engine.get("parameter_size") or "-",
                engine.get("quantization") or "-",
                f"{settings.get('format_mode')}/{settings.get('think')}",
                _pct(s.get("accuracy")),
                _num(s.get("mean_score"), 2),
                _pct(s.get("json_strict_rate")),
                _pct(s.get("schema_valid_rate")),
                str(s.get("critical_failures", 0)),
                _pct((s.get("consistency") or {}).get("same_answer_rate")),
                str(s.get("runs")),
            ]
        )
    parts += [
        "## Качество",
        "",
        _table(
            [
                "Модель",
                "Параметры",
                "Квант.",
                "Формат/think",
                "Точность",
                "Ср. балл",
                "Чистый JSON",
                "Схема",
                "Крит. ошибки",
                "Стабильность",
                "Прогонов",
            ],
            quality_rows,
        ),
        "",
    ]

    categories = sorted({name for s in ordered for name in (s.get("by_category") or {})})
    category_rows = [
        [s["model"]] + [_pct((s["by_category"].get(c) or {}).get("accuracy")) for c in categories]
        for s in ordered
    ]
    parts += ["## Точность по категориям", "", _table(["Модель"] + categories, category_rows), ""]

    speed_rows = []
    for s in sorted(
        ordered, key=lambda s: ((s.get("machine") or {}).get("label") or "", s["model"])
    ):
        latency = s.get("latency_s") or {}
        memory = s.get("memory") or {}
        warmup = s.get("warmup") or {}
        speed_rows.append(
            [
                (s.get("machine") or {}).get("label") or "-",
                s["model"],
                _num(latency.get("median")),
                _num(latency.get("p90")),
                _num(s.get("gen_tokens_per_s")),
                _num(warmup.get("load_s")),
                _num(memory.get("size_gb")),
                _pct(_gpu_share(memory)),
                str(s.get("errors", 0)),
            ]
        )
    parts += [
        "## Скорость (сравнивать в пределах одной машины)",
        "",
        _table(
            [
                "Машина",
                "Модель",
                "Задержка, медиана, с",
                "p90, с",
                "Генерация, ток/с",
                "Загрузка, с",
                "Память, ГБ",
                "На GPU",
                "Ошибки",
            ],
            speed_rows,
        ),
        "",
    ]

    failed = [
        f"- **{s['model']}**: {', '.join(s['failed_cases'])}"
        for s in ordered
        if s.get("failed_cases")
    ]
    if failed:
        parts += ["## Непройденные тесты", "", *failed, ""]
    critical = []
    for s in ordered:
        counts: dict[tuple[str, tuple[str, ...]], int] = {}
        for item in s.get("critical_details") or []:
            key = (item["case_id"], tuple(item["fields"]))
            counts[key] = counts.get(key, 0) + 1
        if counts:
            entries = [
                f"{case} ({', '.join(fields)})" + (f" ×{n}" if n > 1 else "")
                for (case, fields), n in sorted(counts.items())
            ]
            critical.append(f"- **{s['model']}**: " + "; ".join(entries))
    if critical:
        parts += ["## Критичные ошибки", "", *critical, ""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.compare")
    parser.add_argument("--results", type=Path, default=BENCH_DIR / "results")
    parser.add_argument("--all", action="store_true", help="все прогоны, а не только последние")
    parser.add_argument("--output", type=Path, help="сохранить таблицы в Markdown-файл")
    args = parser.parse_args(argv)

    summaries = load_summaries(args.results)
    if not args.all:
        summaries = latest_only(summaries)
    report = render(summaries)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"Сохранено: {args.output}")
    else:
        sys.stdout.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
