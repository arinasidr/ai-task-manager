"""Общий бенчмарк LLM на эталонных запросах.

Модель получает системный промпт и JSON с `context` и `input` тест-кейса (без `expected`),
возвращает структурированный JSON, который сравнивается с эталоном. Для каждой модели
создаётся папка benchmarks/results/<время>_<модель>/ с файлами:
  meta.json        — модель, движок, настройки, хэши промпта и тестов, сведения о машине;
  responses.jsonl  — ответ, проверки, ошибки, задержка и скорость по каждому кейсу;
  summary.json     — сводные метрики.

Примеры:
  python -m benchmarks.run --model qwen3:4b --machine ASUS
  python -m benchmarks.run --model qwen3:4b --model gemma3:4b --repeats 3
  python -m benchmarks.run --model qwen3:4b --only TEST_05,TEST_29
  python -m benchmarks.run --backend openai --base-url http://localhost:8080/v1 --model local
"""

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # запуск как `python benchmarks/run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.backends import (  # noqa: E402
    BackendError,
    GenerationSettings,
    OllamaBackend,
    OpenAIBackend,
)
from benchmarks.schema import validate  # noqa: E402
from benchmarks.scoring import parse_model_output, score_case  # noqa: E402
from benchmarks.summary import summarize  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = BENCH_DIR / "cases.jsonl"
DEFAULT_PROMPT = BENCH_DIR / "prompts" / "system.md"
DEFAULT_RESULTS = BENCH_DIR / "results"
MAX_CONSECUTIVE_ERRORS = 3
WARMUP_REQUEST = {"tasks": [], "request": "Покажи мои задачи."}


# ---------------------------------------------------------------- входные данные


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number}: некорректный JSON ({exc})") from exc
        missing = {"id", "category", "input", "expected"} - case.keys()
        if missing:
            raise SystemExit(f"{path}:{number}: нет полей {sorted(missing)}")
        cases.append(case)
    return cases


def select_cases(
    cases: list[dict[str, Any]], only: str | None, categories: str | None
) -> list[dict[str, Any]]:
    if only:
        wanted = {item.strip() for item in only.split(",") if item.strip()}
        unknown = wanted - {case["id"] for case in cases}
        if unknown:
            raise SystemExit(f"Нет таких тестов: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in wanted]
    if categories:
        wanted = {item.strip() for item in categories.split(",") if item.strip()}
        cases = [case for case in cases if case["category"] in wanted]
    if not cases:
        raise SystemExit("После фильтрации не осталось тестов")
    return cases


def build_user_message(case: dict[str, Any]) -> str:
    """Контекст и запрос пользователя. Поле expected модели не передаётся."""
    payload = dict(case.get("context") or {})
    payload["request"] = case["input"]
    return json.dumps(payload, ensure_ascii=False)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "model"


def _total_memory_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / 1024**3, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (AttributeError, OSError, ValueError):
        return None


def machine_info(label: str | None) -> dict[str, Any]:
    return {
        "label": label,
        "system": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "ram_gb": _total_memory_gb(),
        "python": platform.python_version(),
    }


def default_base_url(backend: str) -> str:
    if backend == "openai":
        return os.environ.get("ATM_OPENAI_BASE_URL", "http://localhost:8080/v1")
    try:
        from ai_task_manager.config import get_settings

        return get_settings().ollama_base_url
    except ImportError:
        return os.environ.get("ATM_OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------- прогон


def evaluate(case: dict[str, Any], text: str, repeat: int) -> dict[str, Any]:
    """Разбирает ответ модели и сравнивает с эталоном."""
    parsed = parse_model_output(text)
    schema_errors = validate(parsed.data) if parsed.data is not None else ["нет JSON"]
    result = score_case(case, parsed.data)
    answer_key = json.dumps(parsed.data, ensure_ascii=False, sort_keys=True) if parsed.data else ""
    return {
        "case_id": case["id"],
        "category": case["category"],
        "repeat": repeat,
        "input": case["input"],
        "expected": case["expected"],
        "raw_text": text,
        "parsed": parsed.data,
        "json_parsed": parsed.data is not None,
        "json_strict": parsed.strict_json,
        "parse_error": parsed.error,
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "passed": result.passed,
        "score": round(result.score, 4),
        "critical": result.critical_failures,
        "failed_checks": result.failed(),
        "answer_key": answer_key,
    }


def failed_record(case: dict[str, Any], repeat: int, error: str) -> dict[str, Any]:
    """Запись для кейса, на котором движок не ответил (таймаут, обрыв связи)."""
    record = evaluate(case, "", repeat)
    record.update(
        {
            "error": error,
            "thinking": None,
            "latency_s": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "gen_tps": None,
            "prompt_tps": None,
            "tps_estimated": False,
            "finish_reason": None,
        }
    )
    return record


def make_backend(args: argparse.Namespace, model: str, settings: GenerationSettings):
    base_url = args.base_url or default_base_url(args.backend)
    if args.backend == "openai":
        api_key = os.environ.get("ATM_OPENAI_API_KEY")
        return OpenAIBackend(base_url, model, settings, api_key=api_key)
    return OllamaBackend(base_url, model, settings)


def _fmt(value: float | None, pattern: str) -> str:
    return "-" if value is None else format(value, pattern)


def run_model(
    model: str,
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    system_prompt: str,
    cases_text: str,
    backend=None,
) -> dict[str, Any] | None:
    settings = GenerationSettings(
        temperature=args.temperature,
        seed=args.seed,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        format_mode=args.format,
        think=args.think,
        timeout=args.timeout,
    )
    backend = backend or make_backend(args, model, settings)
    print(f"\n=== {model} ({backend.name})")
    try:
        engine_info = backend.info()
    except BackendError as exc:
        print(f"  пропуск: {exc}")
        backend.close()
        return None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"_{safe_name(args.tag)}" if args.tag else ""
    run_dir = Path(args.out_dir) / f"{stamp}_{safe_name(model)}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "model": model,
        "backend": backend.name,
        "base_url": str(backend.client.base_url),
        "engine": engine_info,
        "settings": asdict(settings),
        "system_prompt_file": str(args.prompt),
        "system_prompt_sha256": sha256(system_prompt),
        "cases_file": str(args.cases),
        "cases_sha256": sha256(cases_text),
        "case_ids": [case["id"] for case in cases],
        "repeats": args.repeats,
        "machine": machine_info(args.machine),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    if not args.no_warmup:
        try:
            warm = backend.generate(system_prompt, json.dumps(WARMUP_REQUEST, ensure_ascii=False))
            meta["warmup"] = {"latency_s": round(warm.latency_s, 2), "load_s": warm.load_s}
            print(
                f"  прогрев: {warm.latency_s:.1f} с (загрузка модели {_fmt(warm.load_s, '.1f')} с)"
            )
        except BackendError as exc:
            meta["warmup"] = {"error": str(exc)}
            print(f"  прогрев не удался: {exc}")
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    consecutive_errors = 0
    status = "completed"
    try:
        with (run_dir / "responses.jsonl").open("w", encoding="utf-8") as out:
            for repeat in range(args.repeats):
                for case in cases:
                    try:
                        gen = backend.generate(system_prompt, build_user_message(case))
                        record = evaluate(case, gen.text, repeat)
                        record.update(
                            {
                                "error": None,
                                "thinking": gen.thinking,
                                "latency_s": round(gen.latency_s, 3),
                                "prompt_tokens": gen.prompt_tokens,
                                "output_tokens": gen.output_tokens,
                                "gen_tps": _round_opt(gen.gen_tps),
                                "prompt_tps": _round_opt(gen.prompt_tps),
                                "tps_estimated": gen.tps_estimated,
                                "finish_reason": gen.finish_reason,
                            }
                        )
                        consecutive_errors = 0
                    except BackendError as exc:
                        record = failed_record(case, repeat, str(exc))
                        consecutive_errors += 1
                    records.append(record)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    _print_record(record, args.repeats)
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        status = "aborted"
                        print(f"  {MAX_CONSECUTIVE_ERRORS} ошибки подряд, прогон модели остановлен")
                        break
                if status == "aborted":
                    break
    except KeyboardInterrupt:
        status = "interrupted"
        print("\n  прервано пользователем, сохраняю то, что успели")

    summary = summarize(records)
    summary.update(
        {
            "model": model,
            "status": status,
            "engine": engine_info,
            "machine": meta["machine"],
            "settings": meta["settings"],
            "system_prompt_sha256": meta["system_prompt_sha256"],
            "cases_sha256": meta["cases_sha256"],
            "warmup": meta.get("warmup"),
            "memory": backend.memory(),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "results_dir": str(run_dir),
        }
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    backend.close()
    _print_summary(summary)
    if status == "interrupted":
        raise KeyboardInterrupt
    return summary


def _round_opt(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def _print_record(record: dict[str, Any], repeats: int) -> None:
    verdict = "ok  " if record["passed"] else "FAIL"
    if record.get("error"):
        verdict = "ERR "
    marks = " !крит" if record["critical"] else ""
    repeat = f" #{record['repeat'] + 1}" if repeats > 1 else ""
    print(
        f"  {record['case_id']}{repeat:<4} {record['category']:<24} {verdict} "
        f"{record['score']:.2f}  {_fmt(record.get('latency_s'), '6.1f')} с  "
        f"{_fmt(record.get('gen_tps'), '5.1f')} ток/с{marks}"
    )
    if record.get("error"):
        print(f"      ошибка: {record['error']}")


def _print_summary(summary: dict[str, Any]) -> None:
    if not summary.get("runs"):
        print("  нет результатов")
        return
    latency = summary["latency_s"]
    print(
        f"  итог: точность {summary['accuracy']:.0%}, средний балл {summary['mean_score']:.2f}, "
        f"JSON {summary['json_parsed_rate']:.0%}, схема {summary['schema_valid_rate']:.0%}, "
        f"критичных ошибок {summary['critical_failures']}"
    )
    print(
        f"  скорость: медиана {_fmt(latency['median'], '.1f')} с, "
        f"генерация {_fmt(summary['gen_tokens_per_s'], '.1f')} ток/с, "
        f"ошибок движка {summary['errors']}"
    )
    if summary["failed_cases"]:
        print(f"  не пройдены: {', '.join(summary['failed_cases'])}")
    print(f"  результаты: {summary['results_dir']}")


# ---------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="Прогон LLM на эталонных запросах benchmarks/cases.jsonl",
    )
    parser.add_argument(
        "--model", action="append", required=True, help="модель; можно несколько раз"
    )
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--base-url", help="адрес движка (по умолчанию из ATM_OLLAMA_BASE_URL)")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT, help="системный промпт")
    parser.add_argument("--only", help="только эти тесты: TEST_01,TEST_05")
    parser.add_argument("--category", help="только эти категории: planning,ambiguity")
    parser.add_argument(
        "--format",
        choices=["schema", "json", "none"],
        default="schema",
        help="ограничение формата ответа: JSON-схема, любой JSON или без ограничений",
    )
    parser.add_argument(
        "--think",
        choices=["off", "on", "default"],
        default="off",
        help="режим рассуждений для моделей, которые его поддерживают",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-ctx", type=int, default=4096, help="размер контекстного окна")
    parser.add_argument("--num-predict", type=int, default=1024, help="максимум токенов ответа")
    parser.add_argument("--repeats", type=int, default=1, help="повторов каждого теста")
    parser.add_argument("--timeout", type=float, default=300.0, help="таймаут запроса, с")
    parser.add_argument("--no-warmup", action="store_true", help="не делать прогревочный запрос")
    parser.add_argument("--machine", help="метка компьютера, например ASUS или MacBook-M2")
    parser.add_argument("--tag", help="суффикс к имени папки с результатами")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats должен быть не меньше 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases_text = args.cases.read_text(encoding="utf-8")
    cases = select_cases(load_cases(args.cases), args.only, args.category)
    system_prompt = args.prompt.read_text(encoding="utf-8")
    print(
        f"Тестов: {len(cases)} x {args.repeats}, формат: {args.format}, think: {args.think}, "
        f"промпт {sha256(system_prompt)[:12]}"
    )

    started = time.perf_counter()
    summaries = []
    try:
        for model in args.model:
            summaries.append(run_model(model, args, cases, system_prompt, cases_text))
    except KeyboardInterrupt:
        return 130
    print(f"\nГотово за {time.perf_counter() - started:.0f} с")
    if len(args.model) > 1:
        print("Сравнительная таблица: python -m benchmarks.compare")
    completed = all(summary and summary["status"] == "completed" for summary in summaries)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
