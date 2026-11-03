"""Подключение логов и трейсов (P14). Вызывается при сборке графа и при старте API.

- **Логи** — `ATM_LOG_DIR/atm.log`: JSON-строки от логгеров `ai_task_manager.*` (вызовы Tools,
  алерты A15, ошибки) и `uvicorn.error` (длительность `/chat`). Ротация: 5 МБ × 3 файла.
- **Трейсы** — Phoenix по OTLP/HTTP (`ATM_PHOENIX_ENDPOINT`). Вызовы моделей и узлы LangGraph
  перехватывает OpenInference-инструментация LangChain; ход диалога и Tools добавляются
  вручную (`observability.tracing`).

Оба канала необязательные: без настроек ничего не пишется и не отправляется. Если Phoenix не
запущен, спаны теряются, а приложение работает как обычно.
"""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

from ai_task_manager.config import Settings

LOG_FILE = "atm.log"
TRAJECTORY_FILE = "trajectories.jsonl"

_lock = Lock()
_log_paths: set[Path] = set()
_tracing_endpoint: str | None = None

# Поля LogRecord, которые не нужно дублировать в JSON: остальные пришли через extra=...
_STANDARD_FIELDS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Одна строка — один JSON-объект: такие логи легко фильтровать и загружать в Loki."""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        data.update({k: v for k, v in vars(record).items() if k not in _STANDARD_FIELDS})
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def setup_logging(settings: Settings) -> Path | None:
    if not settings.log_dir:
        return None
    path = Path(settings.log_dir) / LOG_FILE
    with _lock:
        if path in _log_paths:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(JsonFormatter())
        handler.setLevel(logging.INFO)
        for name in ("ai_task_manager", "uvicorn.error"):
            logger = logging.getLogger(name)
            logger.addHandler(handler)
            if logger.level == logging.NOTSET or logger.level > logging.INFO:
                logger.setLevel(logging.INFO)
        _log_paths.add(path)
    return path


def setup_tracing(settings: Settings) -> bool:
    """Регистрирует экспорт в Phoenix один раз на процесс."""
    global _tracing_endpoint
    if not settings.phoenix_endpoint:
        return False
    with _lock:
        if _tracing_endpoint is not None:
            return True
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            from phoenix.otel import register
        except ImportError:  # пакеты наблюдаемости не установлены — работаем без трейсов
            logging.getLogger(__name__).warning(
                "Трейсы выключены: не установлены arize-phoenix-otel и "
                "openinference-instrumentation-langchain"
            )
            return False
        provider = register(
            endpoint=settings.phoenix_endpoint,
            project_name=settings.phoenix_project,
            batch=True,
            verbose=False,
        )
        LangChainInstrumentor().instrument(tracer_provider=provider)
        _tracing_endpoint = settings.phoenix_endpoint
    logging.getLogger(__name__).info(
        "Трейсы отправляются в Phoenix", extra={"endpoint": settings.phoenix_endpoint}
    )
    return True


def setup_observability(settings: Settings) -> None:
    setup_logging(settings)
    setup_tracing(settings)
