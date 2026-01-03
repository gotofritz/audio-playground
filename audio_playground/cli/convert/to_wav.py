"""Convert audio files to WAV format."""

from pathlib import Path

import click

from audio_playground.app_context import AppContext
from audio_playground.cli.common import src_option, target_option
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

    click.echo(f"Converting {src} to WAV format...")
    logger.info(f"Converting {src} to WAV format")

    # Ensure target directory exists
    target.parent.mkdir(parents=True, exist_ok=True)

    # Convert the file
    convert_to_wav_fn(src, target)

    click.echo(f"Conversion complete. Output saved to {target}")
    logger.info(f"Conversion complete: {target}")
