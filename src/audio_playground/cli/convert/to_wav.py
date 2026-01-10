"""Convert audio files to WAV format."""

from pathlib import Path

import click

from audio_playground.app_context import AppContext
from audio_playground.cli.common import src_option, target_option
from audio_playground.core.performance_tracker import PerformanceTracker
from audio_playground.core.wav_converter import convert_to_wav as convert_to_wav_fn


@click.command()
@src_option(help_text="Source audio file (any format)")
@target_option(help_text="Target WAV file path")
@click.pass_obj
def to_wav(
    app_context: AppContext,
    src: Path,
    target: Path,
) -> None:
    """
    Convert any audio format to WAV.

    Supports MP4, MP3, and other formats via FFmpeg and pydub.
    """
    logger = app_context.logger

    # Initialize performance tracker
    tracker = PerformanceTracker(
        command_name="convert to-wav",
        output_dir=target.parent,
        logger=logger,
    )
    tracker.start()

    try:
        click.echo(f"Converting {src} to WAV format...")
        logger.info(f"Converting {src} to WAV format")

        # Add metadata
        tracker.add_metadata("source_file", str(src))
        tracker.add_metadata("target_file", str(target))

        # Ensure target directory exists
        target.parent.mkdir(parents=True, exist_ok=True)

        # Convert the file
        convert_to_wav_fn(src, target)

        # Track audio duration and file size
        if target.exists():
            from audio_playground.core.wav_converter import load_audio_duration

            audio_duration = load_audio_duration(target)
            tracker.add_metadata("audio_duration_seconds", round(audio_duration, 2))
            tracker.add_metadata("output_size_mb", round(target.stat().st_size / (1024 * 1024), 2))

        click.echo(f"Conversion complete. Output saved to {target}")
        logger.info(f"Conversion complete: {target}")

    except KeyboardInterrupt:
        # Stop tracker but don't save report on user interrupt
        tracker.stop()
        click.echo("\nInterrupted by user")
        raise
    finally:
        # Finalize and save performance report (only if not interrupted)
        if tracker.metrics.end_time is None:
            tracker.stop()
            report_path = tracker.save_report()
            click.echo(f"Performance report: {report_path}")
