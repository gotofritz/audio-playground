"""Split audio files into overlapping chunks."""

from pathlib import Path

import click

from audio_playground.app_context import AppContext
from audio_playground.cli.common import output_dir_option, src_option
from audio_playground.core.segmenter import split_to_files
from audio_playground.core.wav_converter import load_audio_duration


@click.command()
@src_option(help_text="Source WAV file to split")
@output_dir_option(help_text="Output directory for chunk files")
@click.option(
    "--chunk-duration",
    type=float,
    default=None,
    help="Duration of each chunk in seconds (default: 10.0)",
)
@click.option(
    "--overlap-duration",
    type=float,
    default=None,
    help="Overlap between chunks in seconds (default: 2.0)",
)
@click.pass_obj
def split(
    app_context: AppContext,
    src: Path,
    output_dir: Path,
    chunk_duration: float | None,
    overlap_duration: float | None,
) -> None:
    """
    Split audio file into overlapping chunks (same logic as sam-audio processing).

    Creates chunk files with overlap for seamless processing.
    Uses the SAME chunking calculation as process_long_audio for consistency.

    Example:
        For 60s audio with 10s chunks and 2s overlap:
        - Chunk 0: 0-10s
        - Chunk 1: 8-18s (overlaps 2s with chunk 0)
        - Chunk 2: 16-26s (overlaps 2s with chunk 1)
        - etc.
    """
    logger = app_context.logger
    config = app_context.config

    # Use defaults from config if not specified
    if chunk_duration is None:
        chunk_duration = config.chunk_duration
    if overlap_duration is None:
        overlap_duration = config.chunk_overlap

    click.echo(f"Splitting {src} into overlapping chunks...")
    logger.info(f"Splitting {src} with chunk_duration={chunk_duration}s, overlap={overlap_duration}s")

    # Get audio duration
    total_duration = load_audio_duration(src)
    click.echo(f"Audio duration: {total_duration:.2f}s")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split to files using canonical chunking logic
    chunk_files, chunk_metadata = split_to_files(
        src,
        output_dir,
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
    )

    # Report results
    click.echo(f"\nCreated {len(chunk_files)} chunks (with {overlap_duration}s overlap):")
    for i, (chunk_file, (start_time, duration)) in enumerate(zip(chunk_files, chunk_metadata)):
        click.echo(f"  {chunk_file.name}: start={start_time:.2f}s, duration={duration:.2f}s")

    click.echo(f"\nMetadata saved to: {output_dir / 'chunk_metadata.json'}")
    logger.info(f"Split complete: {len(chunk_files)} chunks in {output_dir}")
