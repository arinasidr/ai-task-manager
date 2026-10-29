"""Запросы CLI к FastAPI."""

from typing import Any
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from ai_task_manager.api.schemas import (
    CancelActionResponse,
    ChatResponse,
    ConfirmActionResponse,
)


class APIClientError(RuntimeError):
    """API недоступен или вернул некорректный ответ."""


class APIClient:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = self._http.post(path, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            try:
                error = exc.response.json()
            except ValueError:
                error = None
            detail = error.get("detail") if isinstance(error, dict) else None
            raise APIClientError(
                f"API вернул HTTP {exc.response.status_code}: "
                f"{detail or exc.response.reason_phrase}"
            ) from exc
        except httpx.HTTPError as exc:
            raise APIClientError(f"Нет связи с API: {exc}") from exc
        except ValueError as exc:
            raise APIClientError("API вернул некорректный JSON") from exc

        if not isinstance(data, dict):
            raise APIClientError("API вернул ответ не в виде JSON-объекта")
        return data

    def chat(
        self,
        message: str,
        conversation_id: str | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        payload = {"message": message, "conversation_id": conversation_id}
        if request_id is not None:
            payload["request_id"] = request_id
        data = self._post(
            "/api/v1/chat",
            payload,
        )
        try:
            return ChatResponse.model_validate(data)
        except ValidationError as exc:
            raise APIClientError("API вернул некорректный ответ чата") from exc

    def confirm(self, action_id: str) -> ConfirmActionResponse:
        safe_id = quote(action_id, safe="")
        data = self._post(f"/api/v1/actions/{safe_id}/confirm")
        try:
            return ConfirmActionResponse.model_validate(data)
        except ValidationError as exc:
            raise APIClientError("API вернул некорректный ответ подтверждения") from exc

    def cancel(self, action_id: str) -> CancelActionResponse:
        safe_id = quote(action_id, safe="")
        data = self._post(f"/api/v1/actions/{safe_id}/cancel")
        try:
            return CancelActionResponse.model_validate(data)
        except ValidationError as exc:
            raise APIClientError("API вернул некорректный ответ отмены") from exc
