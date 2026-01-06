"""Process audio files with Demucs model to separate stems."""

import logging
from pathlib import Path

from audio_playground.core.separator.demucs import SeparatorDemucs


def process_audio_with_demucs(
    audio_path: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    shifts: int,
    num_workers: int,
    logger: logging.Logger,
    suffix: str,
    show_progress: bool = True,
) -> None:
    """
    Process audio file with Demucs model to separate stems.

    Args:
        audio_path: Path to the audio file to process
        output_dir: Output directory for separated stems
        model_name: Demucs model name (e.g., htdemucs_ft)
        device: Device to use (cpu, cuda, etc.)
        shifts: Number of random shifts for equivariant stabilization
        num_workers: Number of worker threads
        logger: Logger instance
        suffix: Suffix for output files (e.g., 'demucs' for 'drums-demucs.wav').
                Should come from config.demucs_suffix.
        show_progress: Show progress bar during processing
    """
    # Use the wrapper class to hide implementation details
    separator = SeparatorDemucs(
        model_name=model_name,
        device=device,
        shifts=shifts,
        num_workers=num_workers,
        logger=logger,
    )
    separator.separate(
        audio_path=audio_path,
        output_dir=output_dir,
        suffix=suffix,
        show_progress=show_progress,
    )
