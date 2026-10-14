import pytest

from ai_task_manager.config import Settings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATM_API_PORT", raising=False)
    settings = Settings(_env_file=None)
    assert settings.api_port == 8000
    assert settings.ollama_base_url == "http://localhost:11434"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATM_API_PORT", "9000")
    assert Settings(_env_file=None).api_port == 9000
