"""Evals производительности AI Task Manager (P16).

    python -m perf.run                       # настоящие модели (нужна Ollama), 3 повтора
    python -m perf.run --compare-routing     # роутинг против «всё на одну модель»
    python -m perf.run --mode mock --check   # без моделей: CI, проверка стабильности

Что меряется на одном и том же диалоге (`TURNS`), каждый повтор — с чистой БД:
- **задержки:** ход целиком и каждый узел графа (траектория P14), p50 / p95 / максимум;
- **скорость LLM:** для каждой модели — число вызовов, время, токены/с генерации и чтения
  промпта, время загрузки модели (переключение моделей в Ollama);
- **ресурсы:** CPU и память системы, память процесса приложения и процессов Ollama,
  память моделей в видеокарте (`/api/ps`);
- **ошибки и стабильность:** исход каждого хода против ожидаемого, ошибки агентов, повторы,
  запасные планы, совпадение исходов между повторами.

Результат — `perf/results/<время>-<режим>.json` и сводка в Markdown рядом. Если в `.env` задан
`ATM_PHOENIX_ENDPOINT`, каждый ход прогона попадает в Phoenix в проект `perf-<режим>-<роутинг>`.
"""

import argparse
import json
import statistics
import sys
import tempfile
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_task_manager.agents.llm import JsonLLM, OllamaJsonLLM
from ai_task_manager.agents.planner import PLANNER_OUTPUT_SCHEMA
from ai_task_manager.agents.validator import VALIDATOR_OUTPUT_SCHEMA
from ai_task_manager.config import Settings, get_settings
from ai_task_manager.db.database import connect, init_db
from ai_task_manager.memory.checkpoint import open_checkpointer
from ai_task_manager.workflow.service import ChatResult, create_workflow, run_chat

RESULTS_DIR = Path(__file__).parent / "results"

# Диалог одного повтора: сообщение и ожидаемый исход. Три запроса чтения — на них работает
# роутинг (быстрая модель), остальное — полный разбор.
TURNS: list[tuple[str, str]] = [
    ("Добавь задачу подготовить презентацию к пятнице", "created"),
    ("Добавь срочную задачу исправить баг в авторизации на завтра", "created"),
    ("Покажи задачи", "listed"),
    ("Добавь задачу созвон с командой, мне вообще удобнее работать вечером", "created"),
    ("да", "preference_saved"),
    ("С чего мне лучше начать?", "ordered"),
    ("Покажи мои задачи", "listed"),
    ("Удали задачу #1", "pending"),
    ("Удали отчёт", "clarification"),
    ("Какие у меня задачи", "listed"),
]

# Пороги для --check (режим mock, CI): обработка без моделей должна быть быстрой и без ошибок.
CHECK_MAX_P95_MS = 1500.0


# ---------------------------------------------------------------- модели


class Recorder:
    """Все вызовы моделей прогона: модель, время, токены."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.turn: int | None = None

    def wrap(self, model: str, inner: JsonLLM) -> JsonLLM:
        recorder = self

        class Measured:
            def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict:
                started = time.perf_counter()
                ok = True
                try:
                    return inner.complete_json(system, user, schema)
                except Exception:
                    ok = False
                    raise
                finally:
                    meta = getattr(inner, "last_metadata", None) or {}
                    recorder.calls.append(
                        {
                            "turn": recorder.turn,
                            "model": model,
                            "role": _role(schema),
                            "ok": ok,
                            "seconds": time.perf_counter() - started,
                            "eval_count": meta.get("eval_count"),
                            "eval_s": _ns(meta.get("eval_duration")),
                            "prompt_count": meta.get("prompt_eval_count"),
                            "prompt_s": _ns(meta.get("prompt_eval_duration")),
                            "load_s": _ns(meta.get("load_duration")),
                        }
                    )

        return Measured()


def _ns(value: Any) -> float | None:
    return value / 1e9 if isinstance(value, int | float) else None


def _role(schema: dict[str, Any]) -> str:
    if schema is PLANNER_OUTPUT_SCHEMA:
        return "planner"
    if schema is VALIDATOR_OUTPUT_SCHEMA:
        return "validator"
    return "analyst"


class MockLLM:
    """Модель без Ollama: отвечает по схеме и тексту запроса, с заданной задержкой."""

    def __init__(self, model: str, latency: dict[str, float]) -> None:
        self.model = model
        self.delay = latency.get(model, latency.get("*", 0.0))
        self.last_metadata: dict[str, Any] = {}

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict:
        if self.delay:
            time.sleep(self.delay)
        data = json.loads(user)
        self.last_metadata = {"eval_count": 40, "prompt_eval_count": len(system + user) // 3}
        if schema is VALIDATOR_OUTPUT_SCHEMA:
            return {"decision": "accept", "issues": []}
        if schema is PLANNER_OUTPUT_SCHEMA:
            ids = [task["id"] for task in data["tasks"]]
            return {"ordered_task_ids": ids, "recommendations": [], "message": "Начните с первой."}
        request = data["request"].casefold()
        if "начать" in request:
            return {"intent": "plan_tasks"}
        if "задачи" in request and not request.startswith("добавь"):
            return {"intent": "get_tasks"}
        if request.startswith("удали задачу #"):
            return {"intent": "delete_task", "task_id": int(request.rsplit("#", 1)[1])}
        if request.startswith("удали"):
            return {
                "intent": "clarification_required",
                "missing_fields": ["task_id"],
                "clarification_question": "Укажи номер задачи.",
            }
        title = request.removeprefix("добавь срочную задачу").removeprefix("добавь задачу")
        title = title.split(",")[0].replace(" на завтра", "").replace(" к пятнице", "").strip()
        return {
            "intent": "create_task",
            "tasks": [
                {"title": title.capitalize(), "priority": "high" if "срочн" in request else None}
            ],
        }


def _no_decision(_: dict) -> dict:
    return {"decision": None, "decision_error": None}


# ---------------------------------------------------------------- ресурсы


class ResourceSampler(threading.Thread):
    """Каждые 0,5 с: CPU и память системы, память приложения и Ollama."""

    def __init__(self, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop_event = threading.Event()
        try:
            import psutil
        except ImportError:
            self.psutil = None
        else:
            self.psutil = psutil
            self.me = psutil.Process()
            psutil.cpu_percent()  # первый вызов всегда 0 — «прогреваем»

    def _ollama_rss(self) -> float:
        total = 0
        for proc in self.psutil.process_iter(["name", "memory_info"]):
            name = (proc.info.get("name") or "").casefold()
            if "ollama" in name and proc.info.get("memory_info"):
                total += proc.info["memory_info"].rss
        return total / 2**20

    def run(self) -> None:
        if self.psutil is None:
            return
        while not self._stop_event.wait(self.interval):
            memory = self.psutil.virtual_memory()
            self.samples.append(
                {
                    "cpu_percent": self.psutil.cpu_percent(),
                    "ram_used_mb": (memory.total - memory.available) / 2**20,
                    "app_rss_mb": self.me.memory_info().rss / 2**20,
                    "ollama_rss_mb": self._ollama_rss(),
                }
            )

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        self.join(timeout=2)
        if not self.samples:
            return {"available": self.psutil is not None}
        summary: dict[str, Any] = {"available": True, "samples": len(self.samples)}
        for key in self.samples[0]:
            values = [s[key] for s in self.samples]
            summary[key] = {"mean": statistics.fmean(values), "max": max(values)}
        return summary


def loaded_models(base_url: str) -> list[dict[str, Any]]:
    """Модели в памяти Ollama сейчас: сколько занимают и сколько из этого — видеокарта."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/ps", timeout=5) as response:
            models = json.load(response).get("models") or []
    except Exception:  # noqa: BLE001 — Ollama не запущена или старая версия
        return []
    return [
        {
            "model": m.get("name"),
            "size_mb": round((m.get("size") or 0) / 2**20),
            "vram_mb": round((m.get("size_vram") or 0) / 2**20),
        }
        for m in models
    ]


# ---------------------------------------------------------------- прогон


def outcome(result: ChatResult) -> str:
    if result.error and not (result.analysis or {}).get("intent") == "clarification_required":
        return "error"
    if result.action_id:
        return "pending"
    reply = result.reply
    if reply.startswith("Запомнил"):
        return "preference_saved"
    if reply.startswith("Готово: создан"):
        return "created"
    if reply.startswith(("Ваши задачи", "Задач пока нет")):
        return "listed"
    if (result.plan or {}).get("intent") == "plan_tasks" and (result.plan or {}).get(
        "ordered_task_ids"
    ):
        return "ordered"
    if (result.analysis or {}).get("intent") == "clarification_required":
        return "clarification"
    return "other"


def run_config(
    settings: Settings,
    llm_for: Callable[[str], JsonLLM],
    recorder: Recorder,
    repeats: int,
    label: str,
    decision_node: Callable[[dict], dict] | None = None,
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for repeat in range(repeats):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "perf.db"
            init_db(path)
            run_settings = settings.model_copy(update={"database_path": str(path)})
            with open_checkpointer(path) as checkpointer:
                graph = create_workflow(
                    run_settings,
                    connect=lambda path=path: connect(path),
                    llm_for=llm_for,
                    checkpointer=checkpointer,
                    decision_node=decision_node,
                )
                conversation_id = None
                for index, (message, expected) in enumerate(TURNS):
                    recorder.turn = len(turns)
                    started = time.perf_counter()
                    try:
                        result = run_chat(graph, run_settings, message, conversation_id)
                    except Exception as exc:  # noqa: BLE001 — сбой графа тоже результат
                        turns.append(
                            {
                                "config": label,
                                "repeat": repeat,
                                "turn": index,
                                "message": message,
                                "expected": expected,
                                "outcome": "crash",
                                "error": f"{type(exc).__name__}: {exc}",
                                "ms": (time.perf_counter() - started) * 1000,
                            }
                        )
                        continue
                    conversation_id = result.conversation_id
                    got = outcome(result)
                    turns.append(
                        {
                            "config": label,
                            "repeat": repeat,
                            "turn": index,
                            "message": message,
                            "expected": expected,
                            "outcome": got,
                            "ms": (time.perf_counter() - started) * 1000,
                            "route": result.route,
                            "models": result.models,
                            "error": result.error,
                            "agent_errors": result.agent_error_counts,
                            "model_retries": result.model_retry_counts,
                            "validation_attempts": result.validation_attempts,
                            "plan_source": (result.plan or {}).get("source"),
                            "trace_id": result.trace_id,
                            "nodes": {s["node"]: s["ms"] for s in result.trajectory},
                        }
                    )
                    print(
                        f"  [{label} {repeat + 1}/{repeats}] {message[:45]:45} "
                        f"{turns[-1]['ms'] / 1000:6.1f} с  {got}"
                        + ("" if got == expected else f"  (ожидалось {expected})"),
                        flush=True,
                    )
    return turns


# ---------------------------------------------------------------- сводка


def percentile(values: list[float], q: float) -> float:
    """Метод ближайшего ранга — как в Evals A16."""
    ordered = sorted(values)
    rank = max(1, -(-int(q * len(ordered) * 100) // 100))
    return ordered[min(rank, len(ordered)) - 1]


def latency_stats(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
    }


def summarize(turns: list[dict], calls: list[dict]) -> dict[str, Any]:
    ok = [t for t in turns if t["outcome"] == t["expected"]]
    by_kind: dict[str, list[float]] = defaultdict(list)
    for t in turns:
        by_kind[t["expected"]].append(t["ms"])
    nodes: dict[str, list[float]] = defaultdict(list)
    for t in turns:
        for node, ms in (t.get("nodes") or {}).items():
            nodes[node].append(ms)

    models: dict[str, dict[str, Any]] = {}
    for model in sorted({c["model"] for c in calls}):
        mine = [c for c in calls if c["model"] == model]
        gen = [c for c in mine if c["eval_count"] and c["eval_s"]]
        read = [c for c in mine if c["prompt_count"] and c["prompt_s"]]
        models[model] = {
            "calls": len(mine),
            "failed": sum(not c["ok"] for c in mine),
            "mean_s": statistics.fmean(c["seconds"] for c in mine),
            "max_s": max(c["seconds"] for c in mine),
            "gen_tokens_per_s": (
                sum(c["eval_count"] for c in gen) / sum(c["eval_s"] for c in gen) if gen else None
            ),
            "prompt_tokens_per_s": (
                sum(c["prompt_count"] for c in read) / sum(c["prompt_s"] for c in read)
                if read
                else None
            ),
            "load_s_total": sum(c["load_s"] or 0 for c in mine),
            "loads_over_1s": sum((c["load_s"] or 0) > 1 for c in mine),
        }

    # стабильность: один и тот же ход в разных повторах даёт один исход
    per_turn: dict[tuple, Counter] = defaultdict(Counter)
    for t in turns:
        per_turn[(t["config"], t["turn"])][t["outcome"]] += 1
    unstable = [
        {"config": cfg, "turn": turn, "message": TURNS[turn][0], "outcomes": dict(counts)}
        for (cfg, turn), counts in per_turn.items()
        if len(counts) > 1
    ]
    return {
        "turns": len(turns),
        "expected_outcomes": len(ok),
        "success_rate": len(ok) / len(turns) if turns else 0.0,
        "crashes": sum(t["outcome"] == "crash" for t in turns),
        "errors": sum(t["outcome"] == "error" for t in turns),
        "agent_errors": dict(sum((Counter(t.get("agent_errors") or {}) for t in turns), Counter())),
        "model_retries": dict(
            sum((Counter(t.get("model_retries") or {}) for t in turns), Counter())
        ),
        "validator_retries": sum(max(0, (t.get("validation_attempts") or 0) - 1) for t in turns),
        "fallback_plans": sum(t.get("plan_source") == "fallback" for t in turns),
        "unstable_turns": unstable,
        "latency_all": latency_stats([t["ms"] for t in turns]),
        "latency_by_kind": {k: latency_stats(v) for k, v in by_kind.items()},
        "latency_by_node": {k: latency_stats(v) for k, v in nodes.items()},
        "models": models,
    }


def best_of_repeats(turns: list[dict], config: str, kind: str | None = None) -> float:
    """Сумма лучших времён по каждому ходу: устойчиво к единичным зависаниям."""
    best: dict[int, float] = {}
    for t in turns:
        if t["config"] == config and (kind is None or t["expected"] == kind):
            best[t["turn"]] = min(best.get(t["turn"], float("inf")), t["ms"])
    return sum(best.values())


def outliers(turns: list[dict], factor: float = 3.0) -> list[dict]:
    """Ходы, которые в разы медленнее того же хода в других повторах."""
    by_turn: dict[tuple, list[dict]] = defaultdict(list)
    for t in turns:
        by_turn[(t["config"], t["turn"])].append(t)
    found = []
    for group in by_turn.values():
        if len(group) < 2:
            continue
        for t in group:
            typical = statistics.median([o["ms"] for o in group if o is not t])
            if typical > 1000 and t["ms"] > factor * typical:
                nodes = t.get("nodes") or {"—": 0}
                found.append(
                    {
                        "config": t["config"],
                        "repeat": t["repeat"],
                        "message": t["message"],
                        "ms": t["ms"],
                        "typical_ms": typical,
                        "slowest_node": max(nodes.items(), key=lambda item: item[1])[0],
                    }
                )
    return found


def routing_savings(
    routed: dict[str, Any], single: dict[str, Any], turns: list[dict] | None = None
) -> dict[str, Any]:
    def total(summary: dict, kind: str | None = None) -> float:
        stats = summary["latency_by_kind"][kind] if kind else summary["latency_all"]
        return stats["mean_ms"] * stats["n"]

    reads_routed, reads_single = total(routed, "listed"), total(single, "listed")
    all_routed, all_single = total(routed), total(single)
    best: dict[str, float] = {}
    if turns:
        b = {
            (cfg, kind): best_of_repeats(turns, cfg, kind)
            for cfg in ("routing", "single")
            for kind in (None, "listed")
        }
        best = {
            "best_read_ms_routed": b[("routing", "listed")],
            "best_read_ms_single": b[("single", "listed")],
            "best_total_ms_routed": b[("routing", None)],
            "best_total_ms_single": b[("single", None)],
            "best_read_saving_percent": 100
            * (1 - b[("routing", "listed")] / b[("single", "listed")]),
            "best_total_saving_percent": 100 * (1 - b[("routing", None)] / b[("single", None)]),
        }
    return {
        **best,
        "read_turns_ms_routed": reads_routed,
        "read_turns_ms_single": reads_single,
        "read_saving_percent": 100 * (1 - reads_routed / reads_single) if reads_single else None,
        "total_ms_routed": all_routed,
        "total_ms_single": all_single,
        "total_saving_percent": 100 * (1 - all_routed / all_single) if all_single else None,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Производительность: {report['mode']}, {report['started_at']}",
        "",
        f"Повторов: {report['repeats']}; ходов в повторе: {len(TURNS)}.",
        "",
    ]
    for label, summary in report["configs"].items():
        all_ = summary["latency_all"]
        lines += [
            f"## Конфигурация `{label}`",
            "",
            f"- ожидаемый исход: {summary['expected_outcomes']} из {summary['turns']} "
            f"({summary['success_rate']:.0%}); сбоев {summary['crashes']}, ошибок "
            f"{summary['errors']}, повторов валидатора {summary['validator_retries']}, "
            f"запасных планов {summary['fallback_plans']}",
            f"- ошибки агентов: {summary['agent_errors'] or 'нет'}; повторы моделей: "
            f"{summary['model_retries'] or 'нет'}",
            f"- нестабильных ходов (разный исход в повторах): {len(summary['unstable_turns'])}",
            f"- ход: p50 {all_['p50_ms'] / 1000:.1f} с, p95 {all_['p95_ms'] / 1000:.1f} с, "
            f"максимум {all_['max_ms'] / 1000:.1f} с",
            "",
            "| Тип хода | n | p50, с | p95, с | максимум, с |",
            "|---|---|---|---|---|",
        ]
        for kind, stats in summary["latency_by_kind"].items():
            lines.append(
                f"| {kind} | {stats['n']} | {stats['p50_ms'] / 1000:.1f} | "
                f"{stats['p95_ms'] / 1000:.1f} | {stats['max_ms'] / 1000:.1f} |"
            )
        lines += [
            "",
            "| Модель | вызовов | среднее, с | генерация, ток/с | промпт, ток/с | загрузки > 1 с |",
            "|---|---|---|---|---|---|",
        ]
        for model, m in summary["models"].items():
            gen = f"{m['gen_tokens_per_s']:.1f}" if m["gen_tokens_per_s"] else "—"
            read = f"{m['prompt_tokens_per_s']:.0f}" if m["prompt_tokens_per_s"] else "—"
            lines.append(
                f"| {model} | {m['calls']} | {m['mean_s']:.1f} | {gen} | {read} | "
                f"{m['loads_over_1s']} ({m['load_s_total']:.1f} с) |"
            )
        lines += [
            "",
            "| Узел | p50, мс | p95, мс |",
            "|---|---|---|",
        ]
        for node, stats in sorted(
            summary["latency_by_node"].items(), key=lambda item: -item[1]["p95_ms"]
        ):
            lines.append(f"| {node} | {stats['p50_ms']:.0f} | {stats['p95_ms']:.0f} |")
        resources = report["resources"].get(label) or {}
        if resources.get("samples"):
            lines += [
                "",
                f"Ресурсы: CPU системы в среднем {resources['cpu_percent']['mean']:.0f} % "
                f"(пик {resources['cpu_percent']['max']:.0f} %); память системы до "
                f"{resources['ram_used_mb']['max'] / 1024:.1f} ГБ; приложение до "
                f"{resources['app_rss_mb']['max']:.0f} МБ; процессы Ollama до "
                f"{resources['ollama_rss_mb']['max'] / 1024:.1f} ГБ.",
            ]
        loaded = report["loaded_models"].get(label) or []
        if loaded:
            shown = "; ".join(
                f"{m['model']} — {m['size_mb']} МБ, в видеокарте {m['vram_mb']} МБ" for m in loaded
            )
            lines.append(f"Модели в памяти Ollama после прогона: {shown}.")
        lines.append("")
    if report.get("routing"):
        r = report["routing"]
        lines += [
            "## Роутинг против «всё на одну модель»",
            "",
            f"- запросы чтения: {r['read_turns_ms_routed'] / 1000:.1f} с против "
            f"{r['read_turns_ms_single'] / 1000:.1f} с — экономия {r['read_saving_percent']:.0f} %",
            f"- весь диалог: {r['total_ms_routed'] / 1000:.1f} с против "
            f"{r['total_ms_single'] / 1000:.1f} с — экономия {r['total_saving_percent']:.0f} %",
        ]
        if "best_total_ms_routed" in r:
            lines.append(
                "- по лучшему времени каждого хода (без случайных зависаний): чтение "
                f"{r['best_read_ms_routed'] / 1000:.1f} с против "
                f"{r['best_read_ms_single'] / 1000:.1f} с "
                f"(экономия {r['best_read_saving_percent']:.0f} %), диалог "
                f"{r['best_total_ms_routed'] / 1000:.1f} с против "
                f"{r['best_total_ms_single'] / 1000:.1f} с "
                f"(экономия {r['best_total_saving_percent']:.0f} %)"
            )
        lines.append("")
    if report.get("outliers"):
        lines += ["## Выбросы: ход в 3+ раза медленнее, чем в других повторах", ""]
        for o in report["outliers"]:
            lines.append(
                f"- `{o['config']}`, повтор {o['repeat'] + 1}: «{o['message']}» — "
                f"{o['ms'] / 1000:.1f} с при обычных {o['typical_ms'] / 1000:.1f} с; "
                f"дольше всего узел `{o['slowest_node']}`"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- запуск


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evals производительности (P16)")
    parser.add_argument("--mode", choices=["live", "mock"], default="live")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--compare-routing",
        action="store_true",
        help="дополнительно прогнать «всё на одну модель» (запросы чтения — на основной модели)",
    )
    parser.add_argument(
        "--mock-latency",
        type=json.loads,
        default={},
        help='mock: задержка модели в секундах, например {"gemma4:e2b": 0.05, "*": 0.01}',
    )
    parser.add_argument("--check", action="store_true", help="код возврата 1 при нарушении порогов")
    parser.add_argument(
        "--output", type=Path, help="куда записать JSON (по умолчанию perf/results)"
    )
    parser.add_argument(
        "--from",
        dest="source",
        type=Path,
        help="не запускать модели, а пересчитать сводку по сохранённому JSON прогона",
    )
    args = parser.parse_args(argv)

    if args.source:
        report = json.loads(args.source.read_text(encoding="utf-8"))
        if "single" in report["configs"]:
            report["routing"] = routing_savings(
                report["configs"]["routing"], report["configs"]["single"], report["turns"]
            )
        report["outliers"] = outliers(report["turns"])
        text = markdown(report)
        args.source.with_suffix(".md").write_text(text, encoding="utf-8")
        print(text)
        return 0

    base = get_settings() if args.mode == "live" else Settings(_env_file=None)
    configs = {"routing": base}
    if args.compare_routing:
        # «всё на одну модель»: быстрая модель не используется, чтение идёт на основной
        configs["single"] = base.model_copy(update={"fast_model": base.analyst_model})

    report: dict[str, Any] = {
        "mode": args.mode,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "repeats": args.repeats,
        "turns_per_repeat": len(TURNS),
        "settings": {
            "analyst_model": base.analyst_model,
            "planner_model": base.planner_model,
            "fast_model": base.fast_model,
            "validator_model": base.validator_model,
            "decision_model": base.decision_model,
        },
        "configs": {},
        "resources": {},
        "loaded_models": {},
        "turns": [],
        "llm_calls": [],
    }
    for label, settings in configs.items():
        settings = settings.model_copy(
            update={"phoenix_project": f"perf-{args.mode}-{label}"}
            if settings.phoenix_endpoint
            else {}
        )
        recorder = Recorder()
        if args.mode == "live":

            def llm_for(model: str, settings=settings, recorder=recorder) -> JsonLLM:
                return recorder.wrap(model, OllamaJsonLLM(model, settings.ollama_base_url))
        else:

            def llm_for(model: str, recorder=recorder) -> JsonLLM:
                return recorder.wrap(model, MockLLM(model, args.mock_latency))

        print(f"\n=== {label}: {args.repeats} × {len(TURNS)} ходов ({args.mode})", flush=True)
        sampler = ResourceSampler()
        sampler.start()
        # mock: Tev1 тоже подменяем — в CI нет ни Ollama, ни сети
        decision = None if args.mode == "live" else _no_decision
        turns = run_config(settings, llm_for, recorder, args.repeats, label, decision)
        report["resources"][label] = sampler.stop()
        if args.mode == "live":
            report["loaded_models"][label] = loaded_models(settings.ollama_base_url)
        report["configs"][label] = summarize(turns, recorder.calls)
        report["turns"] += turns
        report["llm_calls"] += [{**c, "config": label} for c in recorder.calls]

    if args.compare_routing:
        report["routing"] = routing_savings(
            report["configs"]["routing"], report["configs"]["single"], report["turns"]
        )
    report["outliers"] = outliers(report["turns"])

    output = args.output or RESULTS_DIR / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + f"-{args.mode}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    text = markdown(report)
    output.with_suffix(".md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Результаты: {output}")

    if args.check:
        problems = []
        for label, summary in report["configs"].items():
            if summary["success_rate"] < 1:
                problems.append(f"{label}: исходы не совпали с ожидаемыми")
            if summary["crashes"] or summary["errors"]:
                problems.append(f"{label}: сбои {summary['crashes']}, ошибки {summary['errors']}")
            if summary["unstable_turns"]:
                problems.append(f"{label}: исход меняется между повторами")
            if summary["latency_all"]["p95_ms"] > CHECK_MAX_P95_MS:
                problems.append(
                    f"{label}: p95 {summary['latency_all']['p95_ms']:.0f} мс > {CHECK_MAX_P95_MS}"
                )
        for problem in problems:
            print(f"FAIL {problem}")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
