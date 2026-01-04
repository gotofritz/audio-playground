"""Audio segment merging utilities."""

from logging import Logger
from pathlib import Path
from typing import cast

import numpy as np
from torch import Tensor


def concatenate_segments(segment_files: list[Path]) -> Tensor:
    """
    Load and concatenate audio segments.

    Args:
        segment_files: List of paths to audio segment files (sorted)

    Returns:
        Concatenated audio tensor (channels, samples)

    Note:
        torch and torchaudio are imported lazily inside this function.
    """
    import torch
    import torchaudio

    if not segment_files:
        return torch.tensor([])

    if len(segment_files) == 1:
        audio, _ = torchaudio.load(segment_files[0])
        # casting purely for mypy; there is nothing wrong with just
        # returning audio.float()
        return cast(Tensor, audio.float())

    # Load all segments
    all_audio: list[Tensor] = []
    for seg_file in segment_files:
        audio, _ = torchaudio.load(seg_file)
        all_audio.append(cast(Tensor, audio.squeeze(0)))

    # Simple concatenation
    concatenated = np.concatenate(all_audio)

    return torch.from_numpy(concatenated).unsqueeze(0).float()


def find_prompts_from_files(tmp_dir: Path) -> dict[str, list[Path]]:
    """
    Scan directory for {segment}-target-{prompt}.wav patterns.

    Args:
        tmp_dir: Temporary directory containing segment files

    Returns:
        Dictionary mapping prompt names to sorted lists of segment files
    """
    prompts: dict[str, list[Path]] = {}
    for f in tmp_dir.glob("segment-*-target-*.wav"):
        # Extract prompt from filename: segment-000-target-bass.wav -> bass
        parts = f.stem.split("-target-")
        if len(parts) == 2:
            prompt = parts[1]
            if prompt not in prompts:
                prompts[prompt] = []
            prompts[prompt].append(f)

    # Sort files for each prompt
    for prompt in prompts:
        prompts[prompt] = sorted(prompts[prompt])

    return prompts


def merge_and_save(
    tmp_dir: Path,
    output_dir: Path,
    logger: Logger,
    chain_residuals: bool,
    sample_rate: int | None = None,
) -> None:
    """
    Merge all segments and save outputs.

    Args:
        tmp_dir: Temporary directory containing segment files
        output_dir: Output directory for final merged files
        logger: Logger instance from app_context
        chain_residuals: Whether to save cumulative residual as sam-other.wav
        sample_rate: Target sample rate in Hz. If None, uses original sample rate.

    Note:
        torchaudio is imported lazily inside this function.
        Creates sam-{prompt}.wav for each prompt and optionally sam-other.wav
        for the cumulative residual.
        If sample_rate is provided, output files are resampled to that rate.
    """
    import torchaudio

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sample rate from first TARGET file (what the model actually used)
    first_target = sorted(tmp_dir.glob("segment-*-target-*.wav"))
    if first_target:
        audio, sr_int = torchaudio.load(first_target[0])
        sr = int(sr_int)
        logger.info(f"Using sample rate from model output files: {sr}")
    else:
        # Fallback to original segment sample rate
        segment_files: list[Path] = sorted(
            [
                f
                for f in tmp_dir.glob("segment-*.wav")
                if "-target" not in f.name and "-residual" not in f.name and "-temp" not in f.name
            ]
        )
        if not segment_files:
            raise FileNotFoundError(f"No segment files found in {tmp_dir}")

        audio, sr_int = torchaudio.load(segment_files[0])
        sr = int(sr_int)
        logger.warning(f"No target files found yet, using original segment sample rate: {sr}")

    # Find all prompts from target files
    prompts = find_prompts_from_files(tmp_dir)

    # Determine output sample rate
    output_sr = sample_rate if sample_rate is not None else sr
    if sample_rate is not None and sample_rate != sr:
        logger.info(f"Will resample output from {sr} Hz to {sample_rate} Hz")

    # Concatenate and save each prompt as sam-{prompt}.wav
    for prompt in sorted(prompts.keys()):
        logger.info(f"Concatenating target files for prompt: {prompt}")
        target_files: list[Path] = prompts[prompt]

        concatenated_target = concatenate_segments(target_files)

        # Resample if needed
        if sample_rate is not None and sample_rate != sr:
            import torchaudio.transforms as T

            resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
            concatenated_target = resampler(concatenated_target)

        output = output_dir / f"sam-{prompt}.wav"
        torchaudio.save(output.as_posix(), concatenated_target, output_sr)
        logger.info(f"Saved {output}")

    # Concatenate and save cumulative residual as sam-other.wav (only if chaining was enabled)
    if chain_residuals:
        cumulative_files: list[Path] = sorted(
            tmp_dir.glob("segment-*-residual-cumulative-final.wav")
        )
        if cumulative_files:
            logger.info("Concatenating cumulative residual files...")

            concatenated_cumulative = concatenate_segments(cumulative_files)

            # Resample if needed
            if sample_rate is not None and sample_rate != sr:
                import torchaudio.transforms as T

                resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
                concatenated_cumulative = resampler(concatenated_cumulative)

            output = output_dir / "sam-other.wav"
            torchaudio.save(output.as_posix(), concatenated_cumulative, output_sr)
            logger.info(f"Saved {output}")
        else:
            logger.warning("No cumulative residual files found")
    else:
        logger.debug("Skipping cumulative residual output (--no-chain-residuals)")
