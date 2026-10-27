"""Общие инструкции для агентных промптов."""

from collections.abc import Sequence
from importlib.resources import files


def load_soul() -> str:
    """Читает общие правила из установленного пакета."""
    return files("ai_task_manager").joinpath("prompts", "SOUL.md").read_text(encoding="utf-8")


def with_soul(role_prompt: str) -> str:
    """Добавляет общие правила без примеров ответов перед инструкциями роли."""
    rules = load_soul().split("\n## Примеры ответов пользователю", maxsplit=1)[0].rstrip()
    return f"{rules}\n\n{role_prompt.lstrip()}"


def _strip_frontmatter(text: str) -> str:
    """Убирает метаданные скилла (--- name, description ---): модели нужен только текст."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :].lstrip()
    return text


def load_skill(name: str) -> str:
    """Читает skills/<name>/SKILL.md из установленного пакета."""
    resource = files("ai_task_manager").joinpath("skills", name, "SKILL.md")
    return _strip_frontmatter(resource.read_text(encoding="utf-8"))


def skill_metadata(name: str) -> dict[str, str]:
    """name и description скилла из его заголовка (для документации и проверок)."""
    text = files("ai_task_manager").joinpath("skills", name, "SKILL.md").read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    header = text[3 : text.find("\n---", 3)]
    meta: dict[str, str] = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def build_prompt(role_prompt: str, skills: Sequence[str] = ()) -> str:
    """Системный промпт агента: SOUL.md (общие правила) + роль + скиллы."""
    parts = [role_prompt.strip(), *(load_skill(name) for name in skills)]
    return with_soul("\n\n".join(parts))
