"""Audio format conversion utilities."""

from pathlib import Path


class WavConverter:
    """Handle audio format conversions with lazy imports."""

    @staticmethod
    def convert_to_wav(src_path: Path, dst_path: Path) -> None:
        """
        Convert any audio format to WAV.

        Args:
            src_path: Source audio file path
            dst_path: Destination WAV file path

        Note:
            Uses lazy imports to avoid loading heavy dependencies at module level.
        """
        import shutil
        import subprocess

        # For MP4 files, use ffmpeg
        if src_path.suffix.lower() == ".mp4":
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    src_path.as_posix(),
                    "-c:a",
                    "pcm_s16le",
                    dst_path.as_posix(),
                ],
                check=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
            )
        # If already WAV, just copy
        elif src_path.suffix.lower() == ".wav":
            shutil.copy(src_path, dst_path)
        else:
            # For other formats, use pydub
            from pydub import AudioSegment

            audio = AudioSegment.from_file(src_path)
            audio.export(dst_path.as_posix(), format="wav")

    @staticmethod
    def load_audio_duration(path: Path) -> float:
        """
        Return audio duration in seconds.

        Args:
            path: Path to audio file

        Returns:
            Duration in seconds

        Note:
            Uses lazy imports to avoid loading pydub at module level.
        """
        from typing import cast

        from pydub import AudioSegment

        audio = AudioSegment.from_file(path.as_posix())
        return cast(float, audio.duration_seconds)
