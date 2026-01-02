"""Audio segmentation utilities."""

import json
from pathlib import Path


class Segmenter:
    """Handle audio segmentation with lazy imports."""

    @staticmethod
    def create_segments(
        total_length: float, min_length: float = 9.0, max_length: float = 17.0
    ) -> list[float]:
        """
        Create even-length segments from total duration.

        Args:
            total_length: Total audio duration in seconds
            min_length: Minimum segment length in seconds
            max_length: Maximum segment length in seconds

        Returns:
            List of segment lengths in seconds that sum to total_length

        Note:
            Segments are created to be as even as possible, with target length
            at the midpoint of min and max. The last segment gets the remainder
            to ensure exact total.
        """
        target_length = (min_length + max_length) / 2
        num_segments = max(1, round(total_length / target_length))

        # Create equal segments
        segment_length = total_length / num_segments
        segments = [segment_length] * (num_segments - 1)

        # Last segment gets the remainder to ensure exact total
        segments.append(total_length - sum(segments))

        return segments

    @staticmethod
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
            Uses lazy imports to avoid loading pydub at module level.
            Also saves metadata to a JSON file in the output directory.
        """
        from pydub import AudioSegment

        # Load audio
        audio = AudioSegment.from_file(audio_path.as_posix())

        segment_files: list[Path] = []
        segment_metadata: list[tuple[float, float]] = []
        current_time_ms = 0

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

            current_time_ms += int(seg_length * 1000)

        # Save segment metadata
        metadata_file = output_dir / "segment_metadata.json"
        metadata: dict[str, float | list[tuple[float, float]]] = {
            "overlap_duration": 0.0,
            "segments": segment_metadata,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        return segment_files, segment_metadata
