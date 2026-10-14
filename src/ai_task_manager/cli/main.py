"""CLI: `uv run atm --help`."""

import httpx
import typer

from ai_task_manager import __version__
from ai_task_manager.config import Settings

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
