"""Tests for the extract process-sam-audio command."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from audio_playground.cli.__main__ import cli
from audio_playground.cli.extract.process_sam_audio import batch_items, expand_segment_paths


def test_process_sam_audio_help(cli_runner: CliRunner, cli_env: None) -> None:
    """Test process-sam-audio help command"""
    result = cli_runner.invoke(cli, ["extract", "process-sam-audio", "--help"])
    assert result.exit_code == 0
    assert "Process audio segment(s) with SAM-Audio model" in result.output
    assert "--segment" in result.output
    assert "--prompts" in result.output
    assert "--output-dir" in result.output
    assert "--suffix" in result.output


def test_process_sam_audio_missing_required_args(cli_runner: CliRunner, cli_env: None) -> None:
    """Test process-sam-audio with missing required arguments"""
    # Missing all args
    result = cli_runner.invoke(cli, ["extract", "process-sam-audio"])
    assert result.exit_code != 0
    assert "Missing option" in result.output

    # Missing prompts
    result = cli_runner.invoke(
        cli,
        ["extract", "process-sam-audio", "--segment", "test.wav", "--output-dir", "/tmp"],
    )
    assert result.exit_code != 0

    # Missing output-dir
    result = cli_runner.invoke(
        cli,
        ["extract", "process-sam-audio", "--segment", "test.wav", "--prompts", "bass"],
    )
    assert result.exit_code != 0


def test_process_sam_audio_single_segment(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test process-sam-audio with a single segment"""
    # Create a dummy segment file
    segment_file = tmp_path / "segment-000.wav"
    segment_file.write_text("dummy audio data")

    output_dir = tmp_path / "output"

    # Mock the processing function
    with patch(
        "audio_playground.cli.extract.process_sam_audio.process_segments_with_sam_audio"
    ) as mock_process:
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-sam-audio",
                "--segment",
                str(segment_file),
                "--prompts",
                "bass,vocals",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0

        # Verify the processing function was called
        mock_process.assert_called_once()
        kwargs = mock_process.call_args[1]
        assert len(kwargs["segment_files"]) == 1
        assert kwargs["segment_files"][0] == segment_file
        assert kwargs["prompts"] == ["bass", "vocals"]
        assert kwargs["output_dir"] == output_dir
        assert kwargs["suffix"] == "sam"  # Default suffix


def test_process_sam_audio_multiple_segments(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test process-sam-audio with multiple segments"""
    # Create dummy segment files
    segment1 = tmp_path / "segment-000.wav"
    segment2 = tmp_path / "segment-001.wav"
    segment1.write_text("dummy audio data 1")
    segment2.write_text("dummy audio data 2")

    output_dir = tmp_path / "output"

    # Mock the processing function
    with patch(
        "audio_playground.cli.extract.process_sam_audio.process_segments_with_sam_audio"
    ) as mock_process:
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-sam-audio",
                "--segment",
                str(segment1),
                "--segment",
                str(segment2),
                "--prompts",
                "bass",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0

        # Verify the processing function was called with both segments
        mock_process.assert_called_once()
        kwargs = mock_process.call_args[1]
        assert len(kwargs["segment_files"]) == 2
        # Segments should be sorted
        assert kwargs["segment_files"][0] == segment1
        assert kwargs["segment_files"][1] == segment2


def test_process_sam_audio_custom_suffix(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test process-sam-audio with custom suffix"""
    segment_file = tmp_path / "segment-000.wav"
    segment_file.write_text("dummy audio data")

    output_dir = tmp_path / "output"

    with patch(
        "audio_playground.cli.extract.process_sam_audio.process_segments_with_sam_audio"
    ) as mock_process:
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-sam-audio",
                "--segment",
                str(segment_file),
                "--prompts",
                "bass",
                "--output-dir",
                str(output_dir),
                "--suffix",
                "custom",
            ],
        )

        assert result.exit_code == 0
        kwargs = mock_process.call_args[1]
        assert kwargs["suffix"] == "custom"


def test_process_sam_audio_no_suffix(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test process-sam-audio with no suffix (empty string)"""
    segment_file = tmp_path / "segment-000.wav"
    segment_file.write_text("dummy audio data")

    output_dir = tmp_path / "output"

    with patch(
        "audio_playground.cli.extract.process_sam_audio.process_segments_with_sam_audio"
    ) as mock_process:
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-sam-audio",
                "--segment",
                str(segment_file),
                "--prompts",
                "bass",
                "--output-dir",
                str(output_dir),
                "--suffix",
                "",
            ],
        )

        assert result.exit_code == 0
        kwargs = mock_process.call_args[1]
        assert kwargs["suffix"] == ""


def test_expand_segment_paths_single(tmp_path: Path) -> None:
    """Test expanding a single segment path"""
    segment = tmp_path / "segment.wav"
    segment.write_text("dummy")

    result = expand_segment_paths((str(segment),))
    assert len(result) == 1
    assert result[0] == segment


def test_expand_segment_paths_multiple(tmp_path: Path) -> None:
    """Test expanding multiple segment paths"""
    segment1 = tmp_path / "segment-000.wav"
    segment2 = tmp_path / "segment-001.wav"
    segment1.write_text("dummy")
    segment2.write_text("dummy")

    result = expand_segment_paths((str(segment1), str(segment2)))
    assert len(result) == 2
    assert segment1 in result
    assert segment2 in result


def test_expand_segment_paths_glob(tmp_path: Path) -> None:
    """Test expanding glob patterns"""
    segment1 = tmp_path / "segment-000.wav"
    segment2 = tmp_path / "segment-001.wav"
    segment3 = tmp_path / "other.wav"
    segment1.write_text("dummy")
    segment2.write_text("dummy")
    segment3.write_text("dummy")

    # Use glob pattern
    pattern = str(tmp_path / "segment-*.wav")
    result = expand_segment_paths((pattern,))

    # Should match segment-000.wav and segment-001.wav, but not other.wav
    assert len(result) == 2
    assert segment1 in result
    assert segment2 in result
    assert segment3 not in result


def test_expand_segment_paths_glob_no_matches(tmp_path: Path) -> None:
    """Test glob pattern with no matches raises error"""
    pattern = str(tmp_path / "nonexistent-*.wav")

    with pytest.raises(FileNotFoundError, match="No files matched pattern"):
        expand_segment_paths((pattern,))


def test_expand_segment_paths_missing_file(tmp_path: Path) -> None:
    """Test non-glob path that doesn't exist raises error"""
    missing_file = str(tmp_path / "missing.wav")

    with pytest.raises(FileNotFoundError, match="Segment file not found"):
        expand_segment_paths((missing_file,))


def test_expand_segment_paths_mixed(tmp_path: Path) -> None:
    """Test mixing direct paths and glob patterns"""
    segment1 = tmp_path / "segment-000.wav"
    segment2 = tmp_path / "segment-001.wav"
    segment3 = tmp_path / "other.wav"
    segment1.write_text("dummy")
    segment2.write_text("dummy")
    segment3.write_text("dummy")

    # Mix direct path with glob
    pattern = str(tmp_path / "segment-*.wav")
    result = expand_segment_paths((pattern, str(segment3)))

    # Should get all three files
    assert len(result) == 3
    assert segment1 in result
    assert segment2 in result
    assert segment3 in result


def test_batch_items_even_split() -> None:
    """Test batching with even split"""
    items = ["a", "b", "c", "d"]
    result = batch_items(items, 2)
    assert result == [["a", "b"], ["c", "d"]]


def test_batch_items_uneven_split() -> None:
    """Test batching with uneven split"""
    items = ["a", "b", "c"]
    result = batch_items(items, 2)
    assert result == [["a", "b"], ["c"]]


def test_batch_items_single_batch() -> None:
    """Test batching where all items fit in one batch"""
    items = ["a", "b", "c"]
    result = batch_items(items, 10)
    assert result == [["a", "b", "c"]]


def test_batch_items_batch_size_one() -> None:
    """Test batching with batch size of 1"""
    items = ["a", "b", "c"]
    result = batch_items(items, 1)
    assert result == [["a"], ["b"], ["c"]]


def test_batch_items_invalid_batch_size() -> None:
    """Test batching with invalid batch size raises error"""
    items = ["a", "b", "c"]
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        batch_items(items, 0)


def test_process_sam_audio_device_auto(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test device auto-detection"""
    segment_file = tmp_path / "segment-000.wav"
    segment_file.write_text("dummy audio data")

    output_dir = tmp_path / "output"

    # Mock torch accelerator
    mock_accelerator = Mock()
    mock_accelerator.type = "cuda"

    with (
        patch(
            "audio_playground.cli.extract.process_sam_audio.process_segments_with_sam_audio"
        ) as mock_process,
        patch("torch.accelerator.is_available", return_value=True),
        patch("torch.accelerator.current_accelerator", return_value=mock_accelerator),
    ):
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-sam-audio",
                "--segment",
                str(segment_file),
                "--prompts",
                "bass",
                "--output-dir",
                str(output_dir),
                "--device",
                "auto",
            ],
        )

        assert result.exit_code == 0
        kwargs = mock_process.call_args[1]
        assert kwargs["device"] == "cuda"


def test_process_sam_audio_device_explicit(
    cli_runner: CliRunner,
    cli_env: None,
    tmp_path: Path,
) -> None:
    """Test explicit device specification"""
    segment_file = tmp_path / "segment-000.wav"
    segment_file.write_text("dummy audio data")

    output_dir = tmp_path / "output"

    with patch(
        "audio_playground.cli.extract.process_sam_audio.process_segments_with_sam_audio"
    ) as mock_process:
        result = cli_runner.invoke(
            cli,
            [
                "extract",
                "process-sam-audio",
                "--segment",
                str(segment_file),
                "--prompts",
                "bass",
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ],
        )

        assert result.exit_code == 0
        kwargs = mock_process.call_args[1]
        assert kwargs["device"] == "cpu"
