from unittest.mock import patch

from click.testing import CliRunner

from audio_playground.cli.__main__ import cli


def test_sam_audio_help(cli_runner: CliRunner, cli_env: None) -> None:
    """Test help for sam_audiocommand"""
    result = cli_runner.invoke(cli, ["extract", "sam-audio", "--help"])
    assert result.exit_code == 0
    assert "extract sam-audio [OPTIONS]" in result.output


def test_sam_audio_exception_handling(cli_runner: CliRunner, cli_env: None) -> None:
    """Test exception handling when app_context access fails"""
    with patch("audio_playground.cli.extract.sam_audio.click.echo") as mock_echo:
        mock_echo.side_effect = Exception("Mocked exception")

        result = cli_runner.invoke(cli, ["extract", "sam-audio"], obj=None)

        assert result.exit_code == 1
