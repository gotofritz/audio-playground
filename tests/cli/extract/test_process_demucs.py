"""Tests for the extract process-demucs command."""

from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from audio_playground.cli.__main__ import cli


def test_process_demucs_help(cli_runner: CliRunner, cli_env: None) -> None:
    """Test process-demucs help command"""
    result = cli_runner.invoke(cli, ["extract", "process-demucs", "--help"])
    assert result.exit_code == 0
    assert "Process audio file with Demucs model" in result.output
    assert "--src" in result.output
    assert "--output-dir" in result.output
    assert "--model" in result.output


def test_process_demucs_missing_required_args(cli_runner: CliRunner, cli_env: None) -> None:
    """Test process-demucs with missing required arguments"""
    # Missing all args
    result = cli_runner.invoke(cli, ["extract", "process-demucs"])
    assert result.exit_code != 0
    assert "Missing option" in result.output

    # Missing output-dir
    result = cli_runner.invoke(
        cli,
        ["extract", "process-demucs", "--src", "test.wav"],
    )
    assert result.exit_code != 0

    # Missing src
    result = cli_runner.invoke(
        cli,
        ["extract", "process-demucs", "--output-dir", "/tmp"],
    )
    assert result.exit_code != 0


def test_process_demucs_basic(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test process-demucs with basic arguments"""
    # Create a dummy audio file
    audio_file = tmp_path / "audio.wav"
    audio_file.write_text("dummy audio data")

    output_dir = tmp_path / "output"

    # Mock the processing function
    with patch("audio_playground.cli.extract.process_demucs.process_audio_with_demucs") as mock_process:
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-demucs",
                "--src",
                str(audio_file),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0

        # Verify the processing function was called
        mock_process.assert_called_once()
        kwargs = mock_process.call_args[1]
        assert kwargs["audio_path"] == audio_file
        assert kwargs["output_dir"] == output_dir
        assert kwargs["model_name"] == "htdemucs_ft"  # default
        assert kwargs["shifts"] == 6  # default
        assert kwargs["num_workers"] == 4  # default
        assert kwargs["show_progress"] is True  # default


def test_process_demucs_custom_model(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test process-demucs with custom model"""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_text("dummy audio data")

    output_dir = tmp_path / "output"

    with patch("audio_playground.cli.extract.process_demucs.process_audio_with_demucs") as mock_process:
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-demucs",
                "--src",
                str(audio_file),
                "--output-dir",
                str(output_dir),
                "--model",
                "htdemucs",
            ],
        )

        assert result.exit_code == 0
        kwargs = mock_process.call_args[1]
        assert kwargs["model_name"] == "htdemucs"


def test_process_demucs_custom_shifts(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test process-demucs with custom shifts parameter"""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_text("dummy audio data")

    output_dir = tmp_path / "output"

    with patch("audio_playground.cli.extract.process_demucs.process_audio_with_demucs") as mock_process:
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-demucs",
                "--src",
                str(audio_file),
                "--output-dir",
                str(output_dir),
                "--shifts",
                "10",
            ],
        )

        assert result.exit_code == 0
        kwargs = mock_process.call_args[1]
        assert kwargs["shifts"] == 10


def test_process_demucs_device_auto(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test device auto-detection"""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_text("dummy audio data")

    output_dir = tmp_path / "output"

    # Mock torch accelerator
    mock_accelerator = Mock()
    mock_accelerator.type = "cuda"

    with (
        patch("audio_playground.cli.extract.process_demucs.process_audio_with_demucs") as mock_process,
        patch("torch.accelerator.is_available", return_value=True),
        patch("torch.accelerator.current_accelerator", return_value=mock_accelerator),
    ):
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-demucs",
                "--src",
                str(audio_file),
                "--output-dir",
                str(output_dir),
                "--device",
                "auto",
            ],
        )

        assert result.exit_code == 0
        kwargs = mock_process.call_args[1]
        assert kwargs["device"] == "cuda"


def test_process_demucs_device_explicit(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test explicit device specification"""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_text("dummy audio data")

    output_dir = tmp_path / "output"

    with patch("audio_playground.cli.extract.process_demucs.process_audio_with_demucs") as mock_process:
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-demucs",
                "--src",
                str(audio_file),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ],
        )

        assert result.exit_code == 0
        kwargs = mock_process.call_args[1]
        assert kwargs["device"] == "cpu"


def test_process_demucs_nonexistent_file(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test process-demucs with non-existent file"""
    nonexistent_file = tmp_path / "nonexistent.wav"
    output_dir = tmp_path / "output"

    result = cli_runner.invoke(
        cli,
        [
            "extract",
            "process-demucs",
            "--src",
            str(nonexistent_file),
            "--output-dir",
            str(output_dir),
        ],
    )

    # Click validates path exists, so this should fail
    assert result.exit_code != 0


def test_process_demucs_progress_flag(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test process-demucs with progress flag"""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_text("dummy audio data")

    output_dir = tmp_path / "output"

    with patch("audio_playground.cli.extract.process_demucs.process_audio_with_demucs") as mock_process:
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-demucs",
                "--src",
                str(audio_file),
                "--output-dir",
                str(output_dir),
                "--progress",
            ],
        )

        assert result.exit_code == 0
        kwargs = mock_process.call_args[1]
        assert kwargs["show_progress"] is True


def test_process_demucs_no_progress_flag(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test process-demucs with --no-progress flag"""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_text("dummy audio data")

    output_dir = tmp_path / "output"

    with patch("audio_playground.cli.extract.process_demucs.process_audio_with_demucs") as mock_process:
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-demucs",
                "--src",
                str(audio_file),
                "--output-dir",
                str(output_dir),
                "--no-progress",
            ],
        )

        assert result.exit_code == 0
        kwargs = mock_process.call_args[1]
        assert kwargs["show_progress"] is False
