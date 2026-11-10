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
    # модель планировщика
    planner_model: str = "gemma4:e2b"
    # часовой пояс пользователя по умолчанию: сроки «на дату», сравнение сроков
    timezone: str = "Europe/Moscow"

    fast_model: str = "qwen3:1.7b"
    decision_model: str = "tev1:0.8b-q8_0"
    validator_model: str = "ministral-3:3b"

    # Наблюдаемость. Пусто - выключено: тесты и запуск без Phoenix ничего не пишут.
    # Каталог логов: atm.log (JSON-строки: Tools, алерты, ошибки) и trajectories.jsonl.
    log_dir: str | None = None
    # OTLP/HTTP-адрес Phoenix, например http://localhost:6006/v1/traces
    phoenix_endpoint: str | None = None
    phoenix_project: str = "ai-task-manager"


@lru_cache
def get_settings() -> Settings:
    return Settings()
