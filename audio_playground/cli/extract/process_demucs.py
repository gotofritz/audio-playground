"""Process audio file with Demucs model to separate stems."""

import logging
import traceback
from pathlib import Path

import click

from audio_playground.app_context import AppContext
from audio_playground.cli.common import output_dir_option, src_option


def process_audio_with_demucs(
    audio_path: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    shifts: int,
    num_workers: int,
    logger: logging.Logger,
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
        show_progress: Show progress bar during processing
    """
    # Lazy imports for performance
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.audio import save_audio
    from demucs.pretrained import get_model

    # Setup model
    logger.info(f"Loading Demucs model: {model_name}")
    logger.info(f"Using device: {device}")

    model = get_model(model_name)
    model.to(device)
    model.eval()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    logger.info(f"Loading audio from {audio_path}")
    wav, sr = torchaudio.load(audio_path.as_posix())

    # Demucs expects stereo audio
    if wav.shape[0] == 1:
        # Convert mono to stereo by duplicating the channel
        wav = wav.repeat(2, 1)
        logger.debug("Converted mono to stereo")
    elif wav.shape[0] > 2:
        # Take first 2 channels if more than stereo
        wav = wav[:2]
        logger.warning(f"Audio has {wav.shape[0]} channels, using first 2")

    # Resample if needed
    if sr != model.samplerate:
        logger.info(f"Resampling from {sr}Hz to {model.samplerate}Hz")
        resampler = torchaudio.transforms.Resample(sr, model.samplerate)
        wav = resampler(wav)
        sr = model.samplerate

    # Move to device
    wav = wav.to(device)

    # Apply model
    logger.info(f"Separating audio with {shifts} shifts")
    with torch.inference_mode():
        sources = apply_model(
            model,
            wav.unsqueeze(0),  # Add batch dimension
            shifts=shifts,
            split=True,
            overlap=0.25,
            progress=show_progress,
        )[0]  # Remove batch dimension

    # Save stems
    logger.info(f"Saving separated stems to {output_dir}")
    for source_idx, source_name in enumerate(model.sources):
        stem_audio = sources[source_idx]

        # Build output path: {stem}.wav
        output_path = output_dir / f"{source_name}.wav"

        # Save audio using demucs.audio.save_audio for consistency
        # Convert to CPU and numpy for saving
        stem_cpu = stem_audio.cpu()
        save_audio(
            stem_cpu,
            output_path.as_posix(),
            samplerate=model.samplerate,
            clip="rescale",
            as_float=False,
            bits_per_sample=16,
        )

        logger.debug(f"Saved: {output_path}")

    logger.info(f"Completed! Separated {len(model.sources)} stems")


@click.command(name="process-demucs")
@src_option(required=True, help_text="Source audio file to process")
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
@click.pass_context
def process_demucs(
    ctx: click.Context,
    src: Path,
    output_dir: Path,
    model: str | None,
    device: str | None,
    shifts: int | None,
    num_workers: int | None,
    progress: bool | None,
) -> None:
    """
    Process audio file with Demucs model to separate stems.

    This command runs the Demucs source separation model on an audio file,
    producing separated stems (e.g., drums, bass, other, vocals) in the
    output directory.

    Demucs does not require segmentation - it processes the full file directly.

    Examples:

        # Process audio file with default settings
        audio-playground extract process-demucs \\
            --src audio.wav \\
            --output-dir ./stems

        # Use specific model and device
        audio-playground extract process-demucs \\
            --src audio.wav \\
            --output-dir ./stems \\
            --model htdemucs \\
            --device cuda

        # Increase quality with more shifts (slower but better)
        audio-playground extract process-demucs \\
            --src audio.wav \\
            --output-dir ./stems \\
            --shifts 10

        # Disable progress bar for scripting
        audio-playground extract process-demucs \\
            --src audio.wav \\
            --output-dir ./stems \\
            --no-progress
    """
    try:
        app_context: AppContext = ctx.obj
        logger = app_context.logger
        config = app_context.config

        # Use config defaults for any unspecified options
        model_name = model if model is not None else config.demucs_model
        device_value = device if device is not None else config.device
        shifts_value = shifts if shifts is not None else config.demucs_shifts
        num_workers_value = num_workers if num_workers is not None else config.demucs_num_workers
        show_progress = progress if progress is not None else config.demucs_progress

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

        # Log configuration
        logger.info(f"Source file: {src}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Shifts: {shifts_value}")
        logger.info(f"Workers: {num_workers_value}")
        logger.info(f"Progress bar: {'enabled' if show_progress else 'disabled'}")

        # Process audio
        process_audio_with_demucs(
            audio_path=src,
            output_dir=output_dir,
            model_name=model_name,
            device=device_value,
            shifts=shifts_value,
            num_workers=num_workers_value,
            logger=logger,
            show_progress=show_progress,
        )

        logger.info("All done!")

    except Exception as e:
        logger.error(f"Error occurred: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        click.echo(f"CLI Error: {type(e).__name__}: {str(e) or '(no error message)'}", err=True)
        ctx.exit(1)
