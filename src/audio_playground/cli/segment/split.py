"""Split audio files into segments."""

from pathlib import Path

import click

from audio_playground.app_context import AppContext
from audio_playground.cli.common import (
    max_segments_option,
    output_dir_option,
    src_option,
    window_size_option,
)
from audio_playground.core.segmenter import create_segments, split_to_files
from audio_playground.core.wav_converter import load_audio_duration


@click.command()
@src_option(help_text="Source WAV file to split")
@output_dir_option(help_text="Output directory for segment files")
@window_size_option()
@max_segments_option()
@click.pass_obj
def split(
    app_context: AppContext,
    src: Path,
    output_dir: Path,
    window_size: float | None,
    max_segments: int | None,
) -> None:
    """
    Split audio file into fixed-size segments.

    Creates segment files and metadata in the output directory.
    All segments except the last are exactly window-size seconds.
    """
    logger = app_context.logger

    # Use default if not specified
    if window_size is None:
        window_size = 10.0  # Default: 10 second segments

    click.echo(f"Splitting {src} into segments...")
    logger.info(f"Splitting {src} with window size {window_size}s")

    # Get audio duration
    total_duration = load_audio_duration(src)
    click.echo(f"Audio duration: {total_duration:.2f}s")

    # Calculate segment lengths
    segment_lengths = create_segments(total_duration, window_size, max_segments)
    click.echo(f"Creating {len(segment_lengths)} segments...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split to files
    segment_files, segment_metadata = split_to_files(src, output_dir, segment_lengths)

    # Report results
    click.echo(f"\nCreated {len(segment_files)} segments:")
    for i, (seg_file, (start_time, duration)) in enumerate(zip(segment_files, segment_metadata)):
        click.echo(f"  {seg_file.name}: start={start_time:.2f}s, duration={duration:.2f}s")

    click.echo(f"\nMetadata saved to: {output_dir / 'segment_metadata.json'}")
    logger.info(f"Split complete: {len(segment_files)} segments in {output_dir}")
