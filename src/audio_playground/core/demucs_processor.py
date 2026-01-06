"""Process audio files with Demucs model to separate stems."""

import logging
from pathlib import Path


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

        # Build output path: {stem}-{suffix}.wav or {stem}.wav if no suffix
        if suffix:
            output_filename = f"{source_name}-{suffix}.wav"
        else:
            output_filename = f"{source_name}.wav"
        output_path = output_dir / output_filename

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
