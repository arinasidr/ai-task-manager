"""Конфигурация приложения. Значения берутся из переменных окружения ATM_* и файла .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATM_", env_file=".env", extra="ignore")

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"
    database_path: str = "data/tasks.db"
    ollama_base_url: str = "http://localhost:11434"

    analyst_model: str = "gemma4:e2b"
    # модель планировщика (предварительный выбор A05); позже её будет выбирать роутер
    planner_model: str = "gemma4:e2b"
    # часовой пояс пользователя по умолчанию: сроки «на дату», сравнение сроков
    timezone: str = "Europe/Moscow"


@lru_cache
def get_settings() -> Settings:
    return Settings()
