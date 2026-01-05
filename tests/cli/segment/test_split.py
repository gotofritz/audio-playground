"""Tests for the segment split command."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from audio_playground.cli.__main__ import cli


def test_segment_split_help(cli_runner: CliRunner, cli_env: None) -> None:
    """Test segment split help command"""
    result = cli_runner.invoke(cli, ["segment", "split", "--help"])
    assert result.exit_code == 0
    assert "Split audio file into overlapping chunks" in result.output
    assert "--src" in result.output
    assert "--output-dir" in result.output
    assert "--chunk-duration" in result.output
    assert "--overlap-duration" in result.output


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
    """Test segment split successful execution with overlapping chunks"""
    # Create a dummy source file
    src_file = tmp_path / "input.wav"
    src_file.write_text("dummy audio data")

    output_dir = tmp_path / "chunks"

    # Mock the functions
    with (
        patch("audio_playground.cli.segment.split.load_audio_duration") as mock_duration,
        patch("audio_playground.cli.segment.split.split_to_files") as mock_split,
    ):
        # Setup mocks
        mock_duration.return_value = 30.0

        # With 10s chunks and 2s overlap: chunk 0: 0-10s, chunk 1: 8-18s, chunk 2: 16-26s, chunk 3: 24-30s
        chunk_files = [
            output_dir / "chunk-000.wav",
            output_dir / "chunk-001.wav",
            output_dir / "chunk-002.wav",
            output_dir / "chunk-003.wav",
        ]
        chunk_metadata = [(0.0, 10.0), (8.0, 10.0), (16.0, 10.0), (24.0, 6.0)]
        mock_split.return_value = (chunk_files, chunk_metadata)

        result = cli_runner.invoke(
            cli,
            [
                "segment",
                "split",
                "--src",
                str(src_file),
                "--output-dir",
                str(output_dir),
                "--chunk-duration",
                "10.0",
                "--overlap-duration",
                "2.0",
            ],
        )

        assert result.exit_code == 0
        assert "Splitting" in result.output
        assert "Audio duration: 30.00s" in result.output
        assert "Created 4 chunks" in result.output
        assert "chunk-000.wav" in result.output

        # Verify functions were called correctly
        mock_duration.assert_called_once_with(src_file)
        mock_split.assert_called_once()
        # Check split_to_files was called with correct parameters
        call_args = mock_split.call_args
        assert call_args[0][0] == src_file
        assert call_args[0][1] == output_dir
        assert call_args[1]["chunk_duration"] == 10.0
        assert call_args[1]["overlap_duration"] == 2.0


def test_segment_split_with_defaults(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test segment split uses config defaults when options not specified"""
    src_file = tmp_path / "input.wav"
    src_file.write_text("dummy audio data")
    output_dir = tmp_path / "chunks"

    with (
        patch("audio_playground.cli.segment.split.load_audio_duration") as mock_duration,
        patch("audio_playground.cli.segment.split.split_to_files") as mock_split,
    ):
        mock_duration.return_value = 30.0

        chunk_files = [output_dir / "chunk-000.wav"]
        chunk_metadata = [(0.0, 10.0)]
        mock_split.return_value = (chunk_files, chunk_metadata)

        result = cli_runner.invoke(
            cli,
            [
                "segment",
                "split",
                "--src",
                str(src_file),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        # Verify split_to_files was called with config defaults (10.0 and 2.0)
        call_args = mock_split.call_args
        assert call_args[1]["chunk_duration"] == 10.0  # From config
        assert call_args[1]["overlap_duration"] == 2.0  # From config


def test_segment_group_help(cli_runner: CliRunner, cli_env: None) -> None:
    """Test segment group help command"""
    result = cli_runner.invoke(cli, ["segment", "--help"])
    assert result.exit_code == 0
    assert "Commands to split audio files" in result.output
    assert "split" in result.output
