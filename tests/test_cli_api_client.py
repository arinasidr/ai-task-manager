import json

import httpx
import pytest

from ai_task_manager.cli.api_client import APIClient, APIClientError


def test_chat_and_action_requests() -> None:
    seen: list[tuple[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.url.path, body))

        if request.url.path == "/api/v1/chat":
            return httpx.Response(
                200,
                json={
                    "conversation_id": "dialog-1",
                    "reply": "Ответ агента",
                    "action_id": "action-7",
                },
            )
        if request.url.path.endswith("/confirm"):
            return httpx.Response(
                200,
                json={
                    "action_id": "action-7",
                    "status": "confirmed",
                    "applied": True,
                    "message": "Задача удалена",
                },
            )
        return httpx.Response(
            200,
            json={"action_id": "action-8", "status": "cancelled", "applied": True},
        )

    with httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)) as http:
        api = APIClient(http)
        reply = api.chat("Удали задачу #7", "dialog-1")
        confirmed = api.confirm(reply.action_id)
        cancelled = api.cancel("action-8")

    assert reply.reply == "Ответ агента"
    assert confirmed.message == "Задача удалена"
    assert cancelled.status == "cancelled"
    assert seen == [
        (
            "/api/v1/chat",
            {"message": "Удали задачу #7", "conversation_id": "dialog-1"},
        ),
        ("/api/v1/actions/action-7/confirm", None),
        ("/api/v1/actions/action-8/cancel", None),
    ]


def test_api_error_shows_server_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "Модели недоступны"})

    with httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(APIClientError, match="503.*Модели недоступны"):
            APIClient(http).chat("Привет")
