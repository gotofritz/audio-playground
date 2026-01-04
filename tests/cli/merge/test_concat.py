"""Tests for the merge concat command."""

from pathlib import Path
from unittest.mock import patch

import torch
from click.testing import CliRunner

from audio_playground.cli.__main__ import cli


def test_merge_concat_help(cli_runner: CliRunner, cli_env: None) -> None:
    """Test merge concat help command"""
    result = cli_runner.invoke(cli, ["merge", "concat", "--help"])
    assert result.exit_code == 0
    assert "Concatenate audio segment files" in result.output
    assert "--input-dir" in result.output
    assert "--pattern" in result.output
    assert "--target" in result.output


def test_merge_concat_missing_args(cli_runner: CliRunner, cli_env: None) -> None:
    """Test merge concat with missing arguments"""
    result = cli_runner.invoke(cli, ["merge", "concat"])
    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_merge_concat_no_files_found(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test merge concat when no files match the pattern"""
    input_dir = tmp_path / "segments"
    input_dir.mkdir()

    result = cli_runner.invoke(
        cli,
        [
            "merge",
            "concat",
            "--input-dir",
            str(input_dir),
            "--pattern",
            "segment-*.wav",
            "--target",
            str(tmp_path / "output.wav"),
        ],
    )

    assert result.exit_code != 0
    assert "No files found" in result.output


def test_merge_concat_success(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test merge concat successful concatenation"""
    # Create input directory with dummy segment files
    input_dir = tmp_path / "segments"
    input_dir.mkdir()

    segment_files = [
        input_dir / "segment-000.wav",
        input_dir / "segment-001.wav",
        input_dir / "segment-002.wav",
    ]

    for seg_file in segment_files:
        seg_file.write_text("dummy audio segment")

    target_file = tmp_path / "output.wav"

    # Mock the concatenation and save functions
    mock_tensor = torch.zeros(2, 1000)  # Dummy audio tensor
    with (
        patch("audio_playground.cli.merge.concat.concatenate_segments") as mock_concat,
        patch("torchaudio.load") as mock_load,
        patch("torchaudio.save") as mock_save,
    ):
        mock_concat.return_value = mock_tensor
        mock_load.return_value = (mock_tensor, 44100)

        result = cli_runner.invoke(
            cli,
            [
                "merge",
                "concat",
                "--input-dir",
                str(input_dir),
                "--pattern",
                "segment-*.wav",
                "--target",
                str(target_file),
            ],
        )

        assert result.exit_code == 0
        assert "Found 3 files to concatenate" in result.output
        assert "Successfully concatenated" in result.output

        # Verify concatenate_segments was called with the segment files
        mock_concat.assert_called_once()
        called_files = mock_concat.call_args[0][0]
        assert len(called_files) == 3
        assert all(f.name.startswith("segment-") for f in called_files)

        # Verify save was called
        mock_save.assert_called_once()


def test_merge_concat_custom_pattern(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test merge concat with custom pattern"""
    # Create input directory with different file patterns
    input_dir = tmp_path / "segments"
    input_dir.mkdir()

    # Create both regular segments and target segments
    regular_files = [
        input_dir / "segment-000.wav",
        input_dir / "segment-001.wav",
    ]
    target_files = [
        input_dir / "segment-000-target-bass.wav",
        input_dir / "segment-001-target-bass.wav",
    ]

    for seg_file in regular_files + target_files:
        seg_file.write_text("dummy audio")

    target_file = tmp_path / "bass-output.wav"

    # Mock the concatenation and save functions
    mock_tensor = torch.zeros(2, 1000)
    with (
        patch("audio_playground.cli.merge.concat.concatenate_segments") as mock_concat,
        patch("torchaudio.load") as mock_load,
        patch("torchaudio.save"),  # Mock to prevent actual file I/O, but don't assert on it
    ):
        mock_concat.return_value = mock_tensor
        mock_load.return_value = (mock_tensor, 44100)

        result = cli_runner.invoke(
            cli,
            [
                "merge",
                "concat",
                "--input-dir",
                str(input_dir),
                "--pattern",
                "*target-bass.wav",
                "--target",
                str(target_file),
            ],
        )

        assert result.exit_code == 0
        assert "Found 2 files to concatenate" in result.output

        # Verify only target-bass files were concatenated
        called_files = mock_concat.call_args[0][0]
        assert len(called_files) == 2
        assert all("target-bass" in f.name for f in called_files)


def test_merge_group_help(cli_runner: CliRunner, cli_env: None) -> None:
    """Test merge group help command"""
    result = cli_runner.invoke(cli, ["merge", "--help"])
    assert result.exit_code == 0
    assert "Commands to merge audio segments" in result.output
    assert "concat" in result.output
