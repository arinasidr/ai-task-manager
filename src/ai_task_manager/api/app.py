"""Точка входа FastAPI-приложения."""

from fastapi import FastAPI

from ai_task_manager import __version__


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Task Manager",
        version=__version__,
        description="Мультиагентная система управления личными задачами.",
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
