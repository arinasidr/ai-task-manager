"""Траектория агентов: какие узлы графа прошёл запрос, сколько длился каждый и что решил."""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from ai_task_manager.config import Settings
from ai_task_manager.observability.setup import TRAJECTORY_FILE

Node = Callable[[dict[str, Any]], dict[str, Any]]

_write_lock = Lock()


def _analysis(update: dict[str, Any]) -> dict[str, Any]:
    analysis = update.get("analysis") or {}
    return {"intent": analysis.get("intent"), "error": update.get("error")}


def _plan(update: dict[str, Any]) -> dict[str, Any]:
    plan = update.get("plan") or {}
    return {
        "source": plan.get("source"),
        "tools": [op.get("tool") for op in plan.get("operations") or []],
        "needs_clarification": plan.get("needs_clarification") or None,
    }


def _verdict(update: dict[str, Any]) -> dict[str, Any]:
    verdict = update.get("verdict") or {}
    return {
        "decision": verdict.get("decision"),
        "source": verdict.get("source"),
        "issues": [issue.get("code") for issue in verdict.get("issues") or []] or None,
    }


def _decision(update: dict[str, Any]) -> dict[str, Any]:
    answers = update.get("decision") or {}
    request_type = answers.get("request_type") if isinstance(answers, dict) else None
    choice = request_type.get("choice") if isinstance(request_type, dict) else None
    return {"request_type": choice, "error": update.get("decision_error")}


_SUMMARY: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "preference_answer": lambda u: {"answered": bool(u.get("reply"))},
    "decision": _decision,
    "route": lambda u: {"route": u.get("route")},
    "analyst": _analysis,
    "planner": _plan,
    "validator": _verdict,
    "act": lambda u: {"pending_action": bool(u.get("pending_action_id"))},
    "propose_preference": lambda u: {"proposed": u.get("preference_id")},
}


def trajectory_node(name: str, node: Node) -> Node:
    """Обёртка узла: время и краткий итог шага дописываются в траекторию."""

    def run(state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        update = node(state) or {}
        step: dict[str, Any] = {
            "node": name,
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }
        summary = _SUMMARY.get(name)
        if summary is not None:
            step.update({k: v for k, v in summary(update).items() if v is not None})
        return {**update, "trajectory": [*(state.get("trajectory") or []), step]}

    return run


def write_trajectory(settings: Settings, record: dict[str, Any]) -> Path | None:
    if not settings.log_dir:
        return None
    path = Path(settings.log_dir) / TRAJECTORY_FILE
    line = json.dumps(
        {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **record},
        ensure_ascii=False,
        default=str,
    )
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    return path
