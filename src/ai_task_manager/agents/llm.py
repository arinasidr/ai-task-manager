"""Вызов LLM со структурированным ответом.

Агенты зависят только от протокола `JsonLLM`, поэтому в тестах подставляется поддельная модель,
а роутер (A10) может выдавать агенту нужную модель.
"""

import json
import logging
import re
from typing import Any, Protocol

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Модель не ответила или ответ не удалось разобрать как JSON-объект."""


class JsonLLM(Protocol):
    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Возвращает JSON-объект, ограниченный схемой `schema`."""
        ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _FENCE.sub("", text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Ответ модели не является JSON: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise LLMError("Ответ модели не является JSON-объектом")
    return data


class OllamaJsonLLM:
    """Модель из Ollama с теми же настройками, что в бенчмарке: temperature 0, seed 42."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        num_ctx: int = 4096,
        num_predict: int = 1024,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        # Режим рассуждений выключаем: он сильно увеличивает задержку. У моделей без этого режима
        # Ollama отвечает ошибкой на сам параметр, тогда повторяем запрос без него.
        self._reasoning: bool | None = False

    def _chat(self, schema: dict[str, Any]) -> ChatOllama:
        return ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0,
            seed=42,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
            format=schema,
            reasoning=self._reasoning,
        )

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        messages = [("system", system), ("human", user)]
        try:
            try:
                reply = self._chat(schema).invoke(messages)
            except Exception as exc:
                if self._reasoning is None or "think" not in str(exc).lower():
                    raise
                logger.info("Модель %s не поддерживает параметр think, отключаем его", self.model)
                self._reasoning = None
                reply = self._chat(schema).invoke(messages)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ошибка вызова модели {self.model}: {exc}") from exc
        return parse_json_object(str(reply.content))
