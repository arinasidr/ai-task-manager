"""CLI: `uv run atm --help`."""

from uuid import uuid4

import httpx
import typer

from ai_task_manager import __version__
from ai_task_manager.config import Settings

from .api_client import APIClient, APIClientError

app = typer.Typer(help="CLI AI Task Manager (работает через HTTP API).", no_args_is_help=True)


@app.command()
def version() -> None:
    """Показать версию."""
    typer.echo(__version__)


@app.command()
def health() -> None:
    """Проверить, что API запущен."""
    settings = Settings()
    url = f"http://{settings.api_host}:{settings.api_port}/health"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        typer.echo(f"API недоступен ({url}): {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(response.text)


def run_chat_session(api: APIClient, conversation_id: str | None = None) -> None:
    typer.secho("✦  AI TASK MANAGER", fg=typer.colors.MAGENTA, bold=True)
    typer.secho("   Пиши задачу обычным языком · /exit для выхода", fg=typer.colors.CYAN)
    typer.echo()

    pending_chat: tuple[str, str] | None = None
    pending_action: tuple[str, bool] | None = None

    while True:
        waiting = pending_chat is not None or pending_action is not None
        prompt = "↻  /retry или /exit: " if waiting else "❯  Вы: "
        try:
            message = input(typer.style(prompt, fg=typer.colors.BRIGHT_CYAN, bold=True)).strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nДиалог завершён.")
            return

        if message.casefold() in {"/exit", "/quit", "выход"}:
            typer.secho("◇  До встречи!", fg=typer.colors.CYAN)
            return
        if waiting and message.casefold() != "/retry":
            typer.secho(
                "⚠  Исход предыдущего запроса неизвестен. Повторите его через /retry или выйдите.",
                fg=typer.colors.YELLOW,
            )
            continue
        if not message:
            continue

        if pending_action is None:
            if pending_chat is None:
                pending_chat = (message, str(uuid4()))
            request, request_id = pending_chat
            try:
                result = api.chat(request, conversation_id, request_id)
            except APIClientError as exc:
                typer.secho("⚠  Ошибка API", fg=typer.colors.RED, bold=True, err=True)
                typer.echo(f"   {exc}", err=True)
                typer.echo(f"   ID запроса: {request_id}. Для безопасного повтора введите /retry.")
                continue
            pending_chat = None

            if conversation_id is None:
                typer.secho(
                    f"◇  ID диалога: {result.conversation_id}",
                    fg=typer.colors.CYAN,
                )
            conversation_id = result.conversation_id

            typer.secho("✦  Агент", fg=typer.colors.MAGENTA, bold=True)
            for line in result.reply.splitlines() or [""]:
                typer.echo(f"   {line}")
            typer.echo()

            if result.action_id is None:
                continue

            typer.secho("⚠  Нужно подтверждение", fg=typer.colors.YELLOW, bold=True)
            try:
                approved = typer.confirm("   Выполнить действие?", default=False)
            except typer.Abort:
                approved = False
            pending_action = (result.action_id, approved)

        action_id, approved = pending_action
        try:
            if approved:
                action = api.confirm(action_id)
                if action.applied:
                    typer.secho("✓  Действие выполнено", fg=typer.colors.GREEN, bold=True)
                else:
                    typer.secho("◇  Действие уже выполнялось", fg=typer.colors.CYAN)
                if action.message:
                    typer.echo(f"   {action.message}")
            else:
                api.cancel(action_id)
                typer.secho("×  Действие отменено.", fg=typer.colors.YELLOW)
        except APIClientError as exc:
            typer.secho(
                f"⚠  Ошибка действия {action_id}",
                fg=typer.colors.RED,
                bold=True,
                err=True,
            )
            typer.echo(f"   {exc}", err=True)
            typer.echo("   Повторите тот же запрос через /retry.")
            continue
        pending_action = None
        typer.echo()


@app.command()
def chat(
    conversation_id: str | None = typer.Option(
        None, "--conversation-id", help="Продолжить диалог по его ID"
    ),
) -> None:
    """Общаться с агентами через FastAPI."""
    settings = Settings()
    base_url = f"http://{settings.api_host}:{settings.api_port}"

    with httpx.Client(base_url=base_url, timeout=120.0) as http:
        run_chat_session(APIClient(http), conversation_id)
