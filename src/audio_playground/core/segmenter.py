"""Audio segmentation utilities."""

import json
import math
from pathlib import Path

from pydub import AudioSegment


def calculate_chunk_boundaries(
    total_frames: int,
    sample_rate: int,
    chunk_duration: float,
    overlap_duration: float,
) -> list[tuple[int, int, float, float]]:
    """
    Calculate chunk boundaries with overlap (same logic as process_long_audio).

    This is the CANONICAL chunking calculation used throughout the codebase.
    Both segment split and process_long_audio use this function.

    Args:
        total_frames: Total number of audio frames
        sample_rate: Sample rate in Hz
        chunk_duration: Duration of each chunk in seconds
        overlap_duration: Overlap between chunks in seconds

    Returns:
        List of (start_sample, end_sample, start_time, end_time) tuples

    Example:
        For 60s audio with 10s chunks and 2s overlap:
        - Chunk 0: 0-10s
        - Chunk 1: 8-18s (starts 2s before chunk 0 ends)
        - Chunk 2: 16-26s
        - etc.
    """
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    hop_samples = chunk_samples - overlap_samples

    # Calculate number of chunks needed
    num_chunks = math.ceil((total_frames - overlap_samples) / hop_samples)

    boundaries = []
    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * hop_samples
        end_sample = min(start_sample + chunk_samples, total_frames)
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        boundaries.append((start_sample, end_sample, start_time, end_time))

    return boundaries


def create_segments(
    total_length: float,
    window_size: float,
    max_segments: int | None = None,
) -> list[float]:
    """
    DEPRECATED: Use calculate_chunk_boundaries() instead for overlapping chunks.

    Create fixed-size segments from total duration (NO OVERLAP).

    Args:
        total_length: Total audio duration in seconds
        window_size: Fixed segment length in seconds
        max_segments: Maximum number of segments to create (None = no limit)

    Returns:
        List of segment lengths in seconds that sum to total_length

    Note:
        All segments except the last are exactly window_size seconds.
        The last segment gets the remainder (may be shorter or longer).
        If max_segments is specified, it caps the number of segments (useful for testing).

        This function creates NON-OVERLAPPING segments and is deprecated.
        For modern chunking with overlap, use calculate_chunk_boundaries().
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
    chunk_duration: float = 10.0,
    overlap_duration: float = 2.0,
) -> tuple[list[Path], list[tuple[float, float]]]:
    """
    Split WAV file into overlapping chunk files (same logic as process_long_audio).

    Args:
        audio_path: Path to input WAV file
        output_dir: Directory to save chunk files
        chunk_duration: Duration of each chunk in seconds (default: 10.0)
        overlap_duration: Overlap between chunks in seconds (default: 2.0)

    Returns:
        Tuple of (chunk_files, chunk_metadata) where:
            - chunk_files: List of paths to created chunk files
            - chunk_metadata: List of (start_time, duration) tuples in seconds

    Note:
        Saves metadata to a JSON file in the output directory.
        Uses the same chunking logic as process_long_audio for consistency.
    """
    # Load audio
    audio = AudioSegment.from_file(audio_path.as_posix())
    sample_rate = audio.frame_rate
    total_frames = len(audio.get_array_of_samples())

    # Use canonical chunking calculation
    boundaries = calculate_chunk_boundaries(
        total_frames=total_frames,
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
    )

    chunk_files: list[Path] = []
    chunk_metadata: list[tuple[float, float]] = []

    for i, (start_sample, end_sample, start_time, end_time) in enumerate(boundaries):
        # Extract chunk using sample boundaries
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        chunk = audio[start_ms:end_ms]
        chunk_path = output_dir / f"chunk-{i:03d}.wav"
        chunk.export(chunk_path.as_posix(), format="wav")
        chunk_files.append(chunk_path)

        # Calculate actual duration
        duration = end_time - start_time
        chunk_metadata.append((start_time, duration))

    # Save chunk metadata
    metadata_file = output_dir / "chunk_metadata.json"
    metadata: dict[str, float | list[tuple[float, float]]] = {
        "chunk_duration": chunk_duration,
        "overlap_duration": overlap_duration,
        "chunks": chunk_metadata,
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    return chunk_files, chunk_metadata
