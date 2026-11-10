"""Ручные спаны и записи: ход диалога, вызовы Tools, траектория агентов."""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("ai_task_manager")
tools_logger = logging.getLogger("ai_task_manager.tools")

# Ключи OpenInference (openinference-semantic-conventions), чтобы не тянуть пакет в рантайм.
SPAN_KIND = "openinference.span.kind"
INPUT = "input.value"
OUTPUT = "output.value"
SESSION = "session.id"


def _json(value: Any, limit: int = 4000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def _parent_context() -> Any:
    """Контекст текущего узла LangGraph, чтобы спан Tool лёг внутрь узла `act` в Phoenix."""
    try:
        from openinference.instrumentation.langchain import get_current_span
    except ImportError:
        return None
    span = get_current_span()
    return trace.set_span_in_context(span) if span is not None else None


def current_trace_id() -> str | None:
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


@contextmanager
def chat_turn_span(message: str, conversation_id: str | None) -> Iterator[trace.Span]:
    """Корневой спан хода: внутри него - узлы графа, вызовы моделей и Tools."""
    with tracer.start_as_current_span("chat_turn") as span:
        span.set_attribute(SPAN_KIND, "AGENT")
        span.set_attribute(INPUT, message)
        if conversation_id:
            span.set_attribute(SESSION, conversation_id)
        yield span


@contextmanager
def tool_call(tool: str, args: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Вызов одного Tool: спан TOOL + JSON-строка в atm.log."""
    out: dict[str, Any] = {}
    started = time.perf_counter()
    with tracer.start_as_current_span(f"tool.{tool}", context=_parent_context()) as span:
        span.set_attribute(SPAN_KIND, "TOOL")
        span.set_attribute("tool.name", tool)
        span.set_attribute(INPUT, _json(args))
        try:
            yield out
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            tools_logger.warning(
                "tool_call",
                extra={
                    "tool": tool,
                    "tool_args": args,
                    "status": "error",
                    "error": str(exc),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    "trace_id": current_trace_id(),
                },
            )
            raise
        span.set_attribute(OUTPUT, _json(out.get("result")))
        tools_logger.info(
            "tool_call",
            extra={
                "tool": tool,
                "tool_args": args,
                "status": "ok",
                "result": out.get("result"),
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "trace_id": current_trace_id(),
            },
        )


def log_tools_event(event: str, **fields: Any) -> None:
    """События исполнителя вне одного Tool: откат плана, блокировка без подтверждения."""
    span = trace.get_current_span()
    span.add_event(event, {k: _json(v) for k, v in fields.items()})
    # ключи extra не должны совпадать с полями LogRecord (args, msg, name…) - префикс
    extra = {f"event_{key}": value for key, value in fields.items()}
    tools_logger.warning(event, extra={**extra, "trace_id": current_trace_id()})
