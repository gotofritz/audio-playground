"""Tests for the segment split command."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from audio_playground.cli.__main__ import cli


def test_segment_split_help(cli_runner: CliRunner, cli_env: None) -> None:
    """Test segment split help command"""
    result = cli_runner.invoke(cli, ["segment", "split", "--help"])
    assert result.exit_code == 0
    assert "Split audio file into fixed-size segments" in result.output
    assert "--src" in result.output
    assert "--output-dir" in result.output
    assert "--window-size" in result.output


def test_segment_split_missing_args(cli_runner: CliRunner, cli_env: None) -> None:
    """Test segment split with missing arguments"""
    result = cli_runner.invoke(cli, ["segment", "split"])
    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_segment_split_success(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test segment split successful execution"""
    # Create a dummy source file
    src_file = tmp_path / "input.wav"
    src_file.write_text("dummy audio data")

    output_dir = tmp_path / "segments"

    # Mock the functions
    with (
        patch("audio_playground.cli.segment.split.load_audio_duration") as mock_duration,
        patch("audio_playground.cli.segment.split.create_segments") as mock_create,
        patch("audio_playground.cli.segment.split.split_to_files") as mock_split,
    ):
        # Setup mocks
        mock_duration.return_value = 30.0
        mock_create.return_value = [10.0, 10.0, 10.0]

        segment_files = [
            output_dir / "segment-000.wav",
            output_dir / "segment-001.wav",
            output_dir / "segment-002.wav",
        ]
        segment_metadata = [(0.0, 10.0), (10.0, 10.0), (20.0, 10.0)]
        mock_split.return_value = (segment_files, segment_metadata)

        result = cli_runner.invoke(
            cli,
            [
                "segment",
                "split",
                "--src",
                str(src_file),
                "--output-dir",
                str(output_dir),
                "--window-size",
                "10.0",
            ],
        )

        assert result.exit_code == 0
        assert "Splitting" in result.output
        assert "Audio duration: 30.00s" in result.output
        assert "Creating 3 segments" in result.output
        assert "segment-000.wav" in result.output

        # Verify functions were called correctly
        mock_duration.assert_called_once_with(src_file)
        mock_create.assert_called_once_with(30.0, 10.0, None)
        mock_split.assert_called_once()


def test_segment_split_with_max_segments(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test segment split with max-segments option"""
    src_file = tmp_path / "input.wav"
    src_file.write_text("dummy audio data")
    output_dir = tmp_path / "segments"

    with (
        patch("audio_playground.cli.segment.split.load_audio_duration") as mock_duration,
        patch("audio_playground.cli.segment.split.create_segments") as mock_create,
        patch("audio_playground.cli.segment.split.split_to_files") as mock_split,
    ):
        mock_duration.return_value = 100.0
        mock_create.return_value = [10.0, 10.0]  # Only 2 segments due to max

        segment_files = [output_dir / "segment-000.wav", output_dir / "segment-001.wav"]
        segment_metadata = [(0.0, 10.0), (10.0, 10.0)]
        mock_split.return_value = (segment_files, segment_metadata)

        result = cli_runner.invoke(
            cli,
            [
                "segment",
                "split",
                "--src",
                str(src_file),
                "--output-dir",
                str(output_dir),
                "--max-segments",
                "2",
            ],
        )

        assert result.exit_code == 0
        # Verify max_segments was passed
        mock_create.assert_called_once_with(100.0, 10.0, 2)


def test_segment_group_help(cli_runner: CliRunner, cli_env: None) -> None:
    """Test segment group help command"""
    result = cli_runner.invoke(cli, ["segment", "--help"])
    assert result.exit_code == 0
    assert "Commands to split audio files into segments" in result.output
    assert "split" in result.output
