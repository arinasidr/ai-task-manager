from typer.testing import CliRunner

from ai_task_manager import __version__
from ai_task_manager.cli.main import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
