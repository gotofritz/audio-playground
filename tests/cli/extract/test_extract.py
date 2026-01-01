from click.testing import CliRunner

from audio_playground.cli.__main__ import cli


def test_extract_help(cli_runner: CliRunner, cli_env: None) -> None:
    """Test help for extract"""
    result = cli_runner.invoke(cli, ["extract", "--help"])
    assert result.exit_code == 0
    assert "extract [OPTIONS] COMMAND [ARGS]" in result.output
