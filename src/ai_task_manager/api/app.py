"""Точка входа FastAPI-приложения."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request

from ai_task_manager import __version__
from ai_task_manager.api.actions import router as actions_router
from ai_task_manager.api.chat import router as chat_router
from ai_task_manager.api.preferences import router as preferences_router
from ai_task_manager.api.tasks import router as tasks_router
from ai_task_manager.config import get_settings
from ai_task_manager.db.database import init_db
from ai_task_manager.observability.metrics import MetricsRegistry
from ai_task_manager.observability.setup import setup_observability

logger = logging.getLogger("uvicorn.error")


# выполняется при запуске и остановке приложения
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()  # при запуске: создать таблицы если их нет
    # логи в ATM_LOG_DIR и трейсы в Phoenix - до первого запроса, в том числе /actions
    setup_observability(get_settings())
    yield  # после этого FastAPI начинает обслуживать запросы


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Task Manager",
        version=__version__,
        description="Мультиагентная система управления личными задачами.",
        lifespan=lifespan,
    )
    app.state.metrics = MetricsRegistry()

    @app.middleware("http")
    async def measure_chat(request: Request, call_next):
        if request.url.path != "/api/v1/chat":
            return await call_next(request)

        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = perf_counter() - started
            request.app.state.metrics.record_request(duration)
            logger.info(
                "chat_request status=%s duration_seconds=%.3f",
                status_code,
                duration,
            )

    app.include_router(tasks_router)
    app.include_router(actions_router)
    app.include_router(chat_router)
    app.include_router(preferences_router)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/v1/metrics", tags=["observability"])
    def metrics() -> dict:
        return app.state.metrics.snapshot()

    return app


app = create_app()
