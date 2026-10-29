"""Вызов модели решений через Ollama /v1/systemone."""

from collections.abc import Callable
from typing import Any

import httpx


class DecisionModelError(RuntimeError):
    """Модель решений недоступна или вернула некорректный ответ."""


class SystemOneClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        post: Callable[..., httpx.Response] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._post = post or httpx.post

    def ask(
        self,
        state: str | dict[str, Any],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            response = self._post(
                f"{self.base_url}/v1/systemone",
                json={
                    "model": self.model,
                    "state": state,
                    "questions": questions,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DecisionModelError(f"Ошибка вызова {self.model}: {exc}") from exc

        if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
            raise DecisionModelError("Модель решений вернула ответ без answers")

        return body["answers"]
