"""Concatenate audio segment files."""

from pathlib import Path

import click

from audio_playground.app_context import AppContext
from audio_playground.cli.common import input_dir_option, pattern_option, target_option
from audio_playground.core.merger import concatenate_segments
from audio_playground.core.performance_tracker import PerformanceTracker


@click.command()
@input_dir_option(help_text="Input directory containing segment files")
@pattern_option(
    required=False,
    default="segment-*.wav",
    help_text='File pattern to match (e.g., "segment-*.wav" or "segment-*target-bass.wav")',
)
@target_option(help_text="Output file path for concatenated audio")
@click.pass_obj
def concat(
    app_context: AppContext,
    input_dir: Path,
    pattern: str,
    target: Path,
) -> None:
    """
    Concatenate audio segment files into a single output file.

    Loads all files matching the pattern in input-dir, concatenates them
    in sorted order, and saves the result to the target path.
    """
    logger = app_context.logger

    # Initialize performance tracker
    tracker = PerformanceTracker(
        command_name="merge concat",
        output_dir=target.parent,
        logger=logger,
    )
    tracker.start()

    try:
        click.echo(f"Searching for files matching '{pattern}' in {input_dir}...")
        logger.info(f"Concatenating files: pattern={pattern}, input_dir={input_dir}")

        # Find all matching files
        segment_files = sorted(input_dir.glob(pattern))

        if not segment_files:
            click.echo(
                f"Error: No files found matching pattern '{pattern}' in {input_dir}", err=True
            )
            raise click.Abort()

        # Add metadata
        tracker.add_metadata("input_dir", str(input_dir))
        tracker.add_metadata("pattern", pattern)
        tracker.add_metadata("files_found", len(segment_files))

        click.echo(f"Found {len(segment_files)} files to concatenate:")
        for i, seg_file in enumerate(segment_files[:5]):  # Show first 5
            click.echo(f"  {seg_file.name}")
        if len(segment_files) > 5:
            click.echo(f"  ... and {len(segment_files) - 5} more")

        # Concatenate segments
        logger.info(f"Concatenating {len(segment_files)} segments...")
        concatenated = concatenate_segments(segment_files)

        # Save output
        import torchaudio

        # Create parent directory if needed
        target.parent.mkdir(parents=True, exist_ok=True)

        # Get sample rate from first file
        _, sample_rate = torchaudio.load(segment_files[0])

        torchaudio.save(target.as_posix(), concatenated, int(sample_rate))

        # Add final metadata
        tracker.add_metadata("target_file", str(target))
        tracker.add_metadata("sample_rate", int(sample_rate))
        if target.exists():
            from audio_playground.core.wav_converter import load_audio_duration

            audio_duration = load_audio_duration(target)
            tracker.add_metadata("audio_duration_seconds", round(audio_duration, 2))
            tracker.add_metadata("output_size_mb", round(target.stat().st_size / (1024 * 1024), 2))

        click.echo(f"\nSuccessfully concatenated to: {target}")
        logger.info(f"Concatenation complete: {target}")

    finally:
        # Finalize and save performance report
        tracker.stop()
        tracker.save_report()
