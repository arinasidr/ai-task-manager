"""Сквозной прогон бенчмарка против поддельного Ollama (httpx.MockTransport), без сети."""

import json
from pathlib import Path

import httpx
import pytest

from benchmarks import compare, run
from benchmarks.backends import GenerationSettings, OllamaBackend, OpenAIBackend

ROOT = Path(__file__).resolve().parent.parent
CASES = run.load_cases(ROOT / "benchmarks" / "cases.jsonl")
BY_INPUT = {case["input"]: case for case in CASES}


def ideal_answer(expected: dict) -> dict:
    answer = dict(expected)
    answer.setdefault("warnings", [])
    answer.setdefault("requires_confirmation", False)
    answer.setdefault("should_execute", not answer["requires_confirmation"])
    return answer


class FakeOllama:
    """Отвечает эталоном, кроме тестов из `broken`, где возвращает мусор."""

    def __init__(self, broken: set[str] = frozenset()) -> None:
        self.broken = broken
        self.chat_payloads: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.0"})
        if request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "details": {
                        "family": "qwen3",
                        "parameter_size": "4.0B",
                        "quantization_level": "Q4_K_M",
                    },
                    "capabilities": ["completion", "thinking"],
                },
            )
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "fake:1b", "size": 3 * 1024**3}]})
        if request.url.path == "/api/chat":
            payload = json.loads(request.content)
            self.chat_payloads.append(payload)
            user = json.loads(payload["messages"][1]["content"])
            case = BY_INPUT.get(user["request"])
            if case is None:  # прогревочный запрос
                content = '{"intent": "get_tasks"}'
            elif case["id"] in self.broken:
                content = "Извините, не могу ответить"
            else:
                content = json.dumps(ideal_answer(case["expected"]), ensure_ascii=False)
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": content},
                    "done_reason": "stop",
                    "prompt_eval_count": 800,
                    "prompt_eval_duration": 400_000_000,
                    "eval_count": 50,
                    "eval_duration": 2_000_000_000,
                    "load_duration": 1_500_000_000,
                },
            )
        return httpx.Response(404, json={"error": "not found"})


def _run(tmp_path: Path, fake: FakeOllama, *extra: str) -> dict:
    args = run.parse_args(
        ["--model", "fake:1b", "--out-dir", str(tmp_path), "--machine", "CI", *extra]
    )
    backend = OllamaBackend(
        "http://ollama.test",
        "fake:1b",
        GenerationSettings(think=args.think, format_mode=args.format),
        transport=httpx.MockTransport(fake),
    )
    prompt = (ROOT / "benchmarks" / "prompts" / "system.md").read_text(encoding="utf-8")
    cases_text = (ROOT / "benchmarks" / "cases.jsonl").read_text(encoding="utf-8")
    cases = run.select_cases(CASES, args.only, args.category)
    return run.run_model("fake:1b", args, cases, prompt, cases_text, backend=backend)


def test_full_run_with_reference_answers(tmp_path: Path) -> None:
    fake = FakeOllama()
    summary = _run(tmp_path, fake)

    assert summary["status"] == "completed"
    assert summary["runs"] == len(CASES)
    assert summary["accuracy"] == 1.0
    assert summary["critical_failures"] == 0
    assert summary["schema_valid_rate"] == 1.0
    assert summary["gen_tokens_per_s"] == 25.0
    assert summary["memory"]["size_gb"] == 3.0

    run_dir = Path(summary["results_dir"])
    records = [
        json.loads(line) for line in (run_dir / "responses.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(records) == len(CASES)
    meta = json.loads((run_dir / "meta.json").read_text("utf-8"))
    assert meta["engine"]["quantization"] == "Q4_K_M"
    assert meta["warmup"]["load_s"] == 1.5


def test_expected_is_never_sent_to_model(tmp_path: Path) -> None:
    fake = FakeOllama()
    _run(tmp_path, fake, "--only", "TEST_10,TEST_17")
    for payload in fake.chat_payloads:
        assert "expected" not in payload["messages"][1]["content"]


def test_same_settings_and_think_off_for_thinking_models(tmp_path: Path) -> None:
    fake = FakeOllama()
    _run(tmp_path, fake, "--only", "TEST_01,TEST_02")
    payloads = fake.chat_payloads
    assert all(p["think"] is False for p in payloads)
    assert all(p["format"]["required"][0] == "intent" for p in payloads)
    assert len({json.dumps(p["options"], sort_keys=True) for p in payloads}) == 1
    assert len({p["messages"][0]["content"] for p in payloads}) == 1


def test_broken_answers_are_recorded_as_failures(tmp_path: Path) -> None:
    summary = _run(tmp_path, FakeOllama(broken={"TEST_05"}), "--only", "TEST_01,TEST_05")
    assert summary["accuracy"] == 0.5
    assert summary["json_parsed_rate"] == 0.5
    assert summary["failed_cases"] == ["TEST_05"]


def test_repeats_report_consistency(tmp_path: Path) -> None:
    summary = _run(tmp_path, FakeOllama(), "--only", "TEST_01", "--repeats", "2")
    assert summary["runs"] == 2
    assert summary["consistency"] == {"same_verdict_rate": 1.0, "same_answer_rate": 1.0}


def test_missing_model_is_skipped(tmp_path: Path) -> None:
    def no_model(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.0"})
        return httpx.Response(404, json={"error": "model not found"})

    args = run.parse_args(["--model", "nope", "--out-dir", str(tmp_path)])
    backend = OllamaBackend(
        "http://ollama.test", "nope", GenerationSettings(), transport=httpx.MockTransport(no_model)
    )
    assert run.run_model("nope", args, CASES[:1], "prompt", "", backend=backend) is None


def test_openai_backend_reads_llama_cpp_timings() -> None:
    def server(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"intent": "get_tasks"}'}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 700, "completion_tokens": 40},
                "timings": {"predicted_per_second": 18.5, "prompt_per_second": 210.0},
            },
        )

    backend = OpenAIBackend(
        "http://llama.test/v1", "local", GenerationSettings(), transport=httpx.MockTransport(server)
    )
    gen = backend.generate("system", "user")
    assert gen.gen_tps == 18.5
    assert not gen.tps_estimated
    assert gen.output_tokens == 40


def test_compare_renders_tables(tmp_path: Path) -> None:
    _run(tmp_path, FakeOllama())
    report = compare.render(compare.latest_only(compare.load_summaries(tmp_path)))
    assert "| fake:1b | 4.0B | Q4_K_M | schema/off | 100% |" in report
    assert "## Скорость" in report


@pytest.mark.parametrize("value", ["0", "-1"])
def test_repeats_must_be_positive(value: str) -> None:
    with pytest.raises(SystemExit):
        run.parse_args(["--model", "m", "--repeats", value])
