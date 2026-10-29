import json

import httpx
import pytest
from typer.testing import CliRunner

import ai_task_manager.cli.main as cli_main


def test_chat_command_reuses_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "conversation_id": "dialog-1",
                "reply": "Ответ агента",
                "action_id": None,
            },
        )

    real_client = httpx.Client

    def mocked_client(*, base_url: str, timeout: float) -> httpx.Client:
        return real_client(
            base_url=base_url,
            timeout=timeout,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(cli_main.httpx, "Client", mocked_client)

    result = CliRunner().invoke(
        cli_main.app,
        ["chat"],
        input="Привет\nПродолжим\n/exit\n",
    )

    assert result.exit_code == 0
    assert [(item["message"], item["conversation_id"]) for item in payloads] == [
        ("Привет", None),
        ("Продолжим", "dialog-1"),
    ]
    assert payloads[0]["request_id"] != payloads[1]["request_id"]
    assert "ID диалога: dialog-1" in result.output


@pytest.mark.parametrize(
    ("answer", "expected_endpoint", "expected_text"),
    [
        ("y", "confirm", "Задача удалена"),
        ("", "cancel", "Действие отменено."),
    ],
)
def test_chat_action_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    expected_endpoint: str,
    expected_text: str,
) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        paths.append(path)

        if path == "/api/v1/chat":
            return httpx.Response(
                200,
                json={
                    "conversation_id": "dialog-1",
                    "reply": "Удалить задачу #7?",
                    "action_id": "action-7",
                },
            )
        if path.endswith("/confirm"):
            return httpx.Response(
                200,
                json={
                    "action_id": "action-7",
                    "status": "confirmed",
                    "applied": True,
                    "message": "Задача удалена",
                },
            )
        if path.endswith("/cancel"):
            return httpx.Response(
                200,
                json={
                    "action_id": "action-7",
                    "status": "cancelled",
                    "applied": True,
                },
            )
        raise AssertionError(f"Неожиданный маршрут: {path}")

    real_client = httpx.Client

    def mocked_client(*, base_url: str, timeout: float) -> httpx.Client:
        return real_client(
            base_url=base_url,
            timeout=timeout,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(cli_main.httpx, "Client", mocked_client)

    result = CliRunner().invoke(
        cli_main.app,
        ["chat"],
        input=f"Удали задачу #7\n{answer}\n/exit\n",
    )

    assert result.exit_code == 0
    assert paths == [
        "/api/v1/chat",
        f"/api/v1/actions/action-7/{expected_endpoint}",
    ]
    assert expected_text in result.output


def test_chat_retries_same_request_after_lost_response(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            raise httpx.ReadTimeout("Ответ потерян", request=request)
        return httpx.Response(
            200,
            json={"conversation_id": "dialog-1", "reply": "Готово", "action_id": None},
        )

    real_client = httpx.Client

    def mocked_client(*, base_url: str, timeout: float) -> httpx.Client:
        return real_client(
            base_url=base_url, timeout=timeout, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(cli_main.httpx, "Client", mocked_client)
    result = CliRunner().invoke(
        cli_main.app,
        ["chat"],
        input="Добавь задачу купить молоко\nДобавь другую задачу\n/retry\n/exit\n",
    )

    assert result.exit_code == 0
    assert len(payloads) == 2
    assert payloads[0] == payloads[1]
    assert "Исход предыдущего запроса неизвестен" in result.output
    assert "Готово" in result.output


def test_chat_retries_confirmation_after_lost_response(monkeypatch: pytest.MonkeyPatch) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/v1/chat":
            return httpx.Response(
                200,
                json={"conversation_id": "dialog-1", "reply": "Удалить #7?", "action_id": "a7"},
            )
        if len(paths) == 2:
            raise httpx.ReadTimeout("Ответ потерян", request=request)
        return httpx.Response(
            200,
            json={"action_id": "a7", "status": "confirmed", "applied": False},
        )

    real_client = httpx.Client

    def mocked_client(*, base_url: str, timeout: float) -> httpx.Client:
        return real_client(
            base_url=base_url, timeout=timeout, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(cli_main.httpx, "Client", mocked_client)
    result = CliRunner().invoke(
        cli_main.app,
        ["chat"],
        input="Удали задачу #7\ny\n/retry\n/exit\n",
    )

    assert result.exit_code == 0
    assert paths == [
        "/api/v1/chat",
        "/api/v1/actions/a7/confirm",
        "/api/v1/actions/a7/confirm",
    ]
    assert "уже выполнялось" in result.output
