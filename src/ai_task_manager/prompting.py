"""Общие инструкции для агентных промптов."""

from importlib.resources import files


def load_soul() -> str:
    """Читает общие правила из установленного пакета."""
    return files("ai_task_manager").joinpath("prompts", "SOUL.md").read_text(encoding="utf-8")


def with_soul(role_prompt: str) -> str:
    """Добавляет общие правила без примеров ответов перед инструкциями роли."""
    rules = load_soul().split("\n## Примеры ответов пользователю", maxsplit=1)[0].rstrip()
    return f"{rules}\n\n{role_prompt.lstrip()}"
