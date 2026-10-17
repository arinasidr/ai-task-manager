"""Клиенты движков для запуска LLM.

- OllamaBackend — родной API Ollama (/api/chat). Отдаёт точные счётчики токенов и время
  генерации, поэтому скорость считается без учёта сети и загрузки модели.
- OpenAIBackend — любой OpenAI-совместимый сервер (llama.cpp `llama-server`, vLLM, LM Studio).
  Нужен для сравнения движков тем же набором тестов (задача P04).
"""

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from benchmarks.schema import OUTPUT_SCHEMA

_NS = 1e9


class BackendError(RuntimeError):
    """Ошибка обращения к движку: сервер недоступен, модель не найдена, неверный ответ."""


@dataclass
class GenerationSettings:
    """Одинаковые для всех сравниваемых моделей параметры генерации."""

    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 4096
    num_predict: int = 1024
    format_mode: str = "schema"  # schema | json | none
    think: str = "off"  # off | on | default
    timeout: float = 300.0


@dataclass
class Generation:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    gen_tps: float | None = None
    prompt_tps: float | None = None
    load_s: float | None = None
    finish_reason: str | None = None
    thinking: str | None = None
    tps_estimated: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _rate(tokens: int | None, seconds: float | None) -> float | None:
    if not tokens or not seconds:
        return None
    return tokens / seconds


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_error:
        raise BackendError(f"HTTP {response.status_code}: {response.text[:300]}")


class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        settings: GenerationSettings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=settings.timeout, transport=transport
        )
        self.capabilities: list[str] = []

    def info(self) -> dict[str, Any]:
        """Версия Ollama и сведения о модели (семейство, размер, квантование)."""
        try:
            version = self.client.get("/api/version")
        except httpx.HTTPError as exc:
            raise BackendError(
                f"Ollama недоступна по адресу {self.client.base_url}. Запустите `ollama serve`. "
                f"({exc})"
            ) from exc
        _raise_for_status(version)
        show = self.client.post("/api/show", json={"model": self.model})
        if show.status_code == 404:
            raise BackendError(
                f"Модель {self.model} не скачана. Выполните: ollama pull {self.model}"
            )
        _raise_for_status(show)
        data = show.json()
        self.capabilities = list(data.get("capabilities") or [])
        details = data.get("details") or {}
        return {
            "engine": "ollama",
            "engine_version": version.json().get("version"),
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "format": details.get("format"),
            "capabilities": self.capabilities,
        }

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        s = self.settings
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": "15m",
            "options": {
                "temperature": s.temperature,
                "seed": s.seed,
                "num_ctx": s.num_ctx,
                "num_predict": s.num_predict,
            },
        }
        if s.format_mode == "schema":
            payload["format"] = OUTPUT_SCHEMA
        elif s.format_mode == "json":
            payload["format"] = "json"
        # Параметр think принимают только модели с режимом рассуждений
        if s.think != "default" and "thinking" in self.capabilities:
            payload["think"] = s.think == "on"
        return payload

    def generate(self, system: str, user: str) -> Generation:
        started = time.perf_counter()
        try:
            response = self.client.post("/api/chat", json=self._payload(system, user))
        except httpx.HTTPError as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc
        latency = time.perf_counter() - started
        _raise_for_status(response)
        data = response.json()
        message = data.get("message") or {}
        eval_count = data.get("eval_count")
        prompt_count = data.get("prompt_eval_count")
        eval_s = (data.get("eval_duration") or 0) / _NS
        prompt_s = (data.get("prompt_eval_duration") or 0) / _NS
        return Generation(
            text=message.get("content") or "",
            thinking=message.get("thinking") or None,
            latency_s=latency,
            prompt_tokens=prompt_count,
            output_tokens=eval_count,
            gen_tps=_rate(eval_count, eval_s),
            prompt_tps=_rate(prompt_count, prompt_s),
            load_s=(data.get("load_duration") or 0) / _NS,
            finish_reason=data.get("done_reason"),
        )

    def memory(self) -> dict[str, Any] | None:
        """Сколько памяти занимает загруженная модель (по данным /api/ps)."""
        try:
            response = self.client.get("/api/ps")
            _raise_for_status(response)
        except (httpx.HTTPError, BackendError):
            return None
        for item in response.json().get("models") or []:
            if item.get("name") == self.model or item.get("model") == self.model:
                return {
                    "size_gb": round((item.get("size") or 0) / 1024**3, 2),
                    "size_vram_gb": round((item.get("size_vram") or 0) / 1024**3, 2),
                    "context_length": item.get("context_length"),
                }
        return None

    def close(self) -> None:
        self.client.close()


class OpenAIBackend:
    name = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        settings: GenerationSettings,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=settings.timeout,
            headers=headers,
            transport=transport,
        )

    def info(self) -> dict[str, Any]:
        try:
            response = self.client.get("/models")
        except httpx.HTTPError as exc:
            raise BackendError(
                f"Сервер недоступен по адресу {self.client.base_url} ({exc})"
            ) from exc
        _raise_for_status(response)
        names = [item.get("id") for item in response.json().get("data") or []]
        return {"engine": "openai-compatible", "served_models": names}

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        s = self.settings
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": s.temperature,
            "seed": s.seed,
            "max_tokens": s.num_predict,
        }
        if s.format_mode == "schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "task_manager_answer", "schema": OUTPUT_SCHEMA},
            }
        elif s.format_mode == "json":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def generate(self, system: str, user: str) -> Generation:
        started = time.perf_counter()
        try:
            response = self.client.post("/chat/completions", json=self._payload(system, user))
        except httpx.HTTPError as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc
        latency = time.perf_counter() - started
        _raise_for_status(response)
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}  # есть у llama.cpp llama-server
        output_tokens = usage.get("completion_tokens")
        gen_tps = timings.get("predicted_per_second")
        estimated = gen_tps is None
        if estimated:
            # Без timings скорость приблизительная: в неё входит и обработка промпта
            gen_tps = _rate(output_tokens, latency)
        return Generation(
            text=message.get("content") or "",
            thinking=message.get("reasoning_content") or None,
            latency_s=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=output_tokens,
            gen_tps=gen_tps,
            prompt_tps=timings.get("prompt_per_second"),
            finish_reason=choice.get("finish_reason"),
            tps_estimated=estimated,
        )

    def memory(self) -> dict[str, Any] | None:
        return None

    def close(self) -> None:
        self.client.close()
