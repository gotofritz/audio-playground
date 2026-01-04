"""Composite command for full Demucs extraction pipeline."""

import traceback
import uuid
from pathlib import Path

import click

from audio_playground.app_context import AppContext
from audio_playground.cli.common import output_dir_option, src_option, suffix_option


@click.command(name="demucs")
@src_option(required=True, help_text="Source audio file (any format)")
@output_dir_option(required=True, help_text="Output directory for separated stems")
@click.option(
    "--model",
    type=str,
    default=None,
    help="Demucs model to use. If not specified, uses config default.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use (auto, cpu, cuda). If not specified, uses config default.",
)
@click.option(
    "--shifts",
    type=int,
    default=None,
    help="Number of random shifts for equivariant stabilization. If not specified, uses config default.",
)
@click.option(
    "--num-workers",
    "-j",
    type=int,
    default=None,
    help="Number of worker threads. If not specified, uses config default.",
)
@click.option(
    "--progress/--no-progress",
    default=None,
    help="Show progress bar during processing. If not specified, uses config default.",
)
@suffix_option()
@click.pass_context
def demucs(
    ctx: click.Context,
    src: Path,
    output_dir: Path,
    model: str | None,
    device: str | None,
    shifts: int | None,
    num_workers: int | None,
    progress: bool | None,
    suffix: str | None,
) -> None:
    """
    Composite command: Full Demucs extraction pipeline.

    Workflow:
        1. convert to-wav (src → wav)
        2. extract process-demucs (wav → separated stems)

    This command orchestrates the atomic commands to provide a complete
    Demucs stem separation workflow. Unlike SAM-Audio, Demucs does not
    require segmentation and processes the entire audio file at once.

    Examples:

        # Basic usage
        audio-playground extract demucs --src input.mp4 --output-dir ./stems

        # Use specific model and device
        audio-playground extract demucs \\
            --src input.mp4 \\
            --output-dir ./stems \\
            --model htdemucs \\
            --device cuda

        # Increase quality with more shifts
        audio-playground extract demucs \\
            --src input.mp4 \\
            --output-dir ./stems \\
            --shifts 10
    """
    try:
        app_context: AppContext = ctx.obj
        logger = app_context.logger
        config = app_context.app_config

        # Use config defaults for any unspecified options
        model_name = model if model is not None else config.demucs_model
        device_value = device if device is not None else config.device
        shifts_value = shifts if shifts is not None else config.demucs_shifts
        num_workers_value = num_workers if num_workers is not None else config.demucs_num_workers
        show_progress = progress if progress is not None else config.demucs_progress
        suffix_value = suffix if suffix is not None else config.demucs_suffix

        # Log configuration
        logger.info("Starting Demucs extraction pipeline...")
        logger.info(f"Source file: {src}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Shifts: {shifts_value}")
        logger.info(f"Workers: {num_workers_value}")
        logger.info(f"Progress bar: {'enabled' if show_progress else 'disabled'}")

        # Generate unique temp directory for this run
        run_id = str(uuid.uuid4())
        base_tmp = Path(config.temp_dir)
        tmp_path = base_tmp / run_id
        tmp_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using temp directory: {tmp_path}")

        # Step 1: Convert to WAV
        logger.info("=== Step 1/2: Converting to WAV ===")
        from audio_playground.core.wav_converter import convert_to_wav

        wav_file = tmp_path / "audio.wav"
        convert_to_wav(src, wav_file)
        logger.info(f"Converted to: {wav_file}")

        # Step 2: Process with Demucs
        logger.info("=== Step 2/2: Processing with Demucs ===")
        from audio_playground.cli.extract.process_demucs import (
            process_audio_with_demucs,
        )

        # Determine device
        if device_value == "auto":
            import torch

            accelerator = (
                torch.accelerator.current_accelerator()
                if torch.accelerator.is_available()
                else None
            )
            device_value = accelerator.type if accelerator is not None else "cpu"
            logger.info(f"Auto-detected device: {device_value}")

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process audio with Demucs
        process_audio_with_demucs(
            audio_path=wav_file,
            output_dir=output_dir,
            model_name=model_name,
            device=device_value,
            shifts=shifts_value,
            num_workers=num_workers_value,
            logger=logger,
            suffix=suffix_value,
            show_progress=show_progress,
        )

        logger.info("All done!")
        logger.info(f"Separated stems saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error occurred: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        click.echo(f"CLI Error: {type(e).__name__}: {str(e) or '(no error message)'}", err=True)
        ctx.exit(1)
