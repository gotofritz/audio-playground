"""Audio segmentation utilities."""

import json
from pathlib import Path

from pydub import AudioSegment


def create_segments(
    total_length: float,
    window_size: float = 14.0,
    max_segments: int | None = None,
) -> list[float]:
    """
    Create fixed-size segments from total duration.

    Args:
        total_length: Total audio duration in seconds
        window_size: Fixed segment length in seconds (default: 14.0)
        max_segments: Maximum number of segments to create (None = no limit)

    Returns:
        List of segment lengths in seconds that sum to total_length

    Note:
        All segments except the last are exactly window_size seconds.
        The last segment gets the remainder (may be shorter or longer).
        If max_segments is specified, it caps the number of segments (useful for testing).
    """
    if total_length <= window_size:
        # Audio is shorter than one window - single segment
        return [total_length]

    # Calculate how many full windows fit
    num_segments = int(total_length / window_size)

    # Cap number of segments if specified
    if max_segments is not None and max_segments > 0:
        num_segments = min(num_segments, max_segments)

    # Create fixed-size segments
    segments = [window_size] * num_segments

    # Last segment gets the remainder
    remainder = total_length - (num_segments * window_size)
    if remainder > 0:
        segments.append(remainder)

    return segments


def split_to_files(
    audio_path: Path,
    output_dir: Path,
    segment_lengths: list[float],
) -> tuple[list[Path], list[tuple[float, float]]]:
    """
    Split WAV file into segment files.

    Args:
        audio_path: Path to input WAV file
        output_dir: Directory to save segment files
        segment_lengths: List of segment lengths in seconds

    Returns:
        Tuple of (segment_files, segment_metadata) where:
            - segment_files: List of paths to created segment files
            - segment_metadata: List of (start_time, duration) tuples in seconds

    Note:
        Saves metadata to a JSON file in the output directory.
    """
    # Load audio
    audio = AudioSegment.from_file(audio_path.as_posix())

    segment_files: list[Path] = []
    segment_metadata: list[tuple[float, float]] = []
    current_time_ms = 0.0  # Keep as float to avoid rounding errors

    for i, seg_length in enumerate(segment_lengths):
        # Simple concatenation - segments end-to-end
        seg_start_ms = int(current_time_ms)
        seg_end_ms = int(current_time_ms + seg_length * 1000)

        segment = audio[seg_start_ms:seg_end_ms]
        segment_path = output_dir / f"segment-{i:03d}.wav"
        segment.export(segment_path.as_posix(), format="wav")
        segment_files.append(segment_path)

        # Load the saved segment to get actual duration (accounts for encoding)
        saved_audio = AudioSegment.from_file(segment_path.as_posix())
        actual_duration_s = saved_audio.duration_seconds
        start_time_s = seg_start_ms / 1000.0

        segment_metadata.append((start_time_s, actual_duration_s))

        # Increment as float to preserve precision
        current_time_ms += seg_length * 1000

    # Save segment metadata
    metadata_file = output_dir / "segment_metadata.json"
    metadata: dict[str, float | list[tuple[float, float]]] = {
        "overlap_duration": 0.0,
        "segments": segment_metadata,
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    return segment_files, segment_metadata
