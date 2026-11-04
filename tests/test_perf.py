"""P16: бенчмарк производительности в режиме mock — то же, что запускает CI."""

import json
from pathlib import Path

from perf.run import TURNS, main, outliers, percentile


def test_mock_benchmark_passes_checks(tmp_path: Path) -> None:
    output = tmp_path / "perf.json"

    assert main(["--mode", "mock", "--repeats", "2", "--check", "--output", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    summary = report["configs"]["routing"]
    assert summary["turns"] == 2 * len(TURNS)
    assert summary["success_rate"] == 1
    assert summary["unstable_turns"] == []
    assert {"analyst", "planner", "validator", "act"} <= set(summary["latency_by_node"])
    assert output.with_suffix(".md").exists()


def test_routing_saves_time_on_read_requests(tmp_path: Path) -> None:
    output = tmp_path / "perf.json"
    latency = json.dumps({"gemma4:e2b": 0.03, "qwen3:1.7b": 0.003, "*": 0.01})

    main(
        [
            "--mode",
            "mock",
            "--repeats",
            "1",
            "--compare-routing",
            "--mock-latency",
            latency,
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    # запросы чтения при роутинге идут на быструю модель, без него — на основную
    read_models = {t["models"]["analyst"] for t in report["turns"] if t["expected"] == "listed"}
    assert read_models == {"qwen3:1.7b", "gemma4:e2b"}
    assert report["routing"]["read_saving_percent"] > 30


def test_percentile_nearest_rank() -> None:
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 0.95) == 19.0
    assert percentile([5.0], 0.95) == 5.0


def test_single_stall_is_reported_as_outlier() -> None:
    turns = [
        {
            "config": "single",
            "turn": 1,
            "repeat": r,
            "message": "Добавь",
            "ms": ms,
            "nodes": {"analyst": ms},
        }
        for r, ms in enumerate([368_000.0, 34_600.0, 35_000.0])
    ]
    [found] = outliers(turns)
    assert found["repeat"] == 0
    assert found["slowest_node"] == "analyst"
