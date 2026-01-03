"""Tests for the convert to-wav command."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from audio_playground.cli.__main__ import cli


def test_convert_to_wav_help(cli_runner: CliRunner, cli_env: None) -> None:
    """Test convert to-wav help command"""
    result = cli_runner.invoke(cli, ["convert", "to-wav", "--help"])
    assert result.exit_code == 0
    assert "Convert any audio format to WAV" in result.output
    assert "--src" in result.output
    assert "--target" in result.output


def test_convert_to_wav_missing_args(cli_runner: CliRunner, cli_env: None) -> None:
    """Test convert to-wav with missing arguments"""
    result = cli_runner.invoke(cli, ["convert", "to-wav"])
    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_convert_to_wav_success(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test convert to-wav successful conversion"""
    # Create a dummy source file
    src_file = tmp_path / "input.mp3"
    src_file.write_text("dummy audio data")

    target_file = tmp_path / "output.wav"

    # Mock the actual conversion function
    with patch("audio_playground.cli.convert.to_wav.convert_to_wav_fn") as mock_convert:
        result = cli_runner.invoke(
            cli,
            ["convert", "to-wav", "--src", str(src_file), "--target", str(target_file)],
        )

        assert result.exit_code == 0
        assert "Converting" in result.output
        assert "Conversion complete" in result.output

        # Verify the conversion function was called with correct args
        mock_convert.assert_called_once()
        args = mock_convert.call_args[0]
        assert args[0] == src_file
        assert args[1] == target_file


def test_convert_group_help(cli_runner: CliRunner, cli_env: None) -> None:
    """Test convert group help command"""
    result = cli_runner.invoke(cli, ["convert", "--help"])
    assert result.exit_code == 0
    assert "Commands to convert audio files between formats" in result.output
    assert "to-wav" in result.output
