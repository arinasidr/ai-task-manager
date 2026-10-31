"""Точка входа FastAPI-приложения."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_task_manager import __version__
from ai_task_manager.api.actions import router as actions_router
from ai_task_manager.api.chat import router as chat_router
from ai_task_manager.api.preferences import router as preferences_router
from ai_task_manager.api.tasks import router as tasks_router
from ai_task_manager.db.database import init_db


# выполняется при запуске и остановке приложения
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()  # при запуске: создать таблицы если их нет
    yield  # после этого FastAPI начинает обслуживать запросы


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Task Manager",
        version=__version__,
        description="Мультиагентная система управления личными задачами.",
        lifespan=lifespan,
    )
    app.include_router(tasks_router)
    app.include_router(actions_router)
    app.include_router(chat_router)
    app.include_router(preferences_router)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
