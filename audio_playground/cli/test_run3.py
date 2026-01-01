import json
import shutil
import subprocess
import uuid
from pathlib import Path

import click
import numpy as np

from audio_playground.app_context import AppContext


def create_segments(
    total_length: float, min_length: float = 9, max_length: float = 17
) -> list[float]:
    """Create even segments with target lengths between min and max"""
    target_length = (min_length + max_length) / 2
    num_segments = max(1, round(total_length / target_length))

    # Create equal segments
    segment_length = total_length / num_segments
    segments = [segment_length] * (num_segments - 1)

    # Last segment gets the remainder to ensure exact total
    segments.append(total_length - sum(segments))

    return segments


def concatenate_segments(segment_files: list):
    """
    Simple concatenation of audio segments.

    Args:
        segment_files: List of paths to audio segment files (sorted)

    Returns:
        Concatenated audio tensor (channels, samples)
    """
    import torch
    import torchaudio

    if not segment_files:
        return torch.tensor([])

    if len(segment_files) == 1:
        audio, _ = torchaudio.load(segment_files[0])
        return audio.float()

    # Load all segments
    all_audio = []
    for seg_file in segment_files:
        audio, _ = torchaudio.load(seg_file)
        all_audio.append(audio.squeeze(0).numpy())

    # Simple concatenation
    concatenated = np.concatenate(all_audio)

    return torch.from_numpy(concatenated).unsqueeze(0).float()


def phase_1_segment_and_process(
    config,
    logger,
    src: str | None,
    prompts: tuple[str, ...],
):
    """
    Phase 1: Create segments and process them with the model.
    Only runs if NOT using --continue-from.

    This phase imports torch and SAMAudio only if needed.
    """
    import torch
    import torchaudio
    from pydub import AudioSegment
    from sam_audio import SAMAudio, SAMAudioProcessor

    logger.info("=== PHASE 1: Segmentation and Processing ===")

    src_path = Path(config.source_file) if config.source_file else None
    if not src_path and not src:
        raise ValueError("No source file specified")
    if src:
        src_path = Path(src)

    # Generate unique temp directory for this run
    run_id = str(uuid.uuid4())
    base_tmp = Path(config.temp_dir)
    tmp_path = base_tmp / run_id
    tmp_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using temp directory: {tmp_path}")

    prompts_list = list(prompts) if prompts else config.prompts

    logger.info("Creating segments (no overlap, simple concatenation)")

    # Convert to WAV if needed
    wav_file = tmp_path / "audio.wav"
    if src_path.suffix.lower() == ".mp4":
        logger.info(f"Converting {src_path} to WAV...")
        subprocess.run(
            ["ffmpeg", "-i", src_path.as_posix(), "-c:a", "pcm_s16le", wav_file.as_posix()],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
    elif src_path.suffix.lower() == ".wav":
        shutil.copy(src_path, wav_file)
    else:
        # Try to load and convert
        audio = AudioSegment.from_file(src_path)
        audio.export(wav_file.as_posix(), format="wav")

    # Load audio and create variable-length segments
    audio = AudioSegment.from_file(wav_file.as_posix())
    total_length = audio.duration_seconds
    logger.info(f"Total audio length: {total_length:.2f} seconds")

    # Create segments with variable lengths (9-17s) using min/max
    segment_lengths = create_segments(
        total_length,
        min_length=config.min_segment_length,
        max_length=config.max_segment_length,
    )
    logger.info(
        f"Creating {len(segment_lengths)} segments: {[round(s, 2) for s in segment_lengths]}"
    )

    segment_files = []
    segment_metadata = []
    current_time_ms = 0

    for i, seg_length in enumerate(segment_lengths):
        # Simple concatenation - segments end-to-end
        seg_start_ms = int(current_time_ms)
        seg_end_ms = int(current_time_ms + seg_length * 1000)

        segment = audio[seg_start_ms:seg_end_ms]
        segment_path = tmp_path / f"segment-{i:03d}.wav"
        segment.export(segment_path.as_posix(), format="wav")
        segment_files.append(segment_path)

        # Load the saved segment to get actual duration (accounts for encoding)
        saved_audio = AudioSegment.from_file(segment_path.as_posix())
        actual_duration_s = saved_audio.duration_seconds
        start_time_s = seg_start_ms / 1000.0

        segment_metadata.append((start_time_s, actual_duration_s))

        logger.debug(
            f"Created {segment_path.name} (actual: {actual_duration_s:.2f}s at {start_time_s:.1f}s)"
        )

        current_time_ms += int(seg_length * 1000)

    logger.info(f"Created {len(segment_files)} segments")

    # Save segment metadata
    metadata_file = tmp_path / "segment_metadata.json"
    metadata = {"overlap_duration": 0.0, "segments": segment_metadata}
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)
    logger.info(f"Saved segment metadata to {metadata_file}")

    # Setup model
    device = (
        config.device
        if config.device != "auto"
        else (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
    )
    logger.info(f"Using {device} device")

    model = (
        SAMAudio.from_pretrained(
            config.model_name,
            map_location=device,
        )
        .to(device)
        .eval()
    )
    processor = SAMAudioProcessor.from_pretrained(
        "facebook/sam-audio-small",
    )

    # Dictionary to store output files for each prompt
    target_files_by_prompt = {prompt: [] for prompt in prompts_list}
    residual_files_by_prompt = {prompt: [] for prompt in prompts_list}

    # Process each segment
    with torch.inference_mode():
        for idx, audio_path in enumerate(segment_files):
            logger.info(f"Processing {audio_path.name} ({idx + 1}/{len(segment_files)})")

            sr = None  # Will be set from processor
            segment_residuals = {}  # Store residuals for each prompt from original audio

            # Step 1: Run all prompts on the original audio to get clean targets
            for prompt_idx, prompt in enumerate(prompts_list):
                safe_prompt = prompt.replace(" ", "_").replace("/", "_")
                logger.debug(
                    f"Processing prompt {prompt_idx + 1}/{len(prompts_list)} on original: {prompt}"
                )

                inputs = processor(
                    audios=[audio_path.as_posix()],
                    descriptions=[prompt],
                ).to(device)

                result = model.separate(
                    inputs,
                    predict_spans=config.predict_spans,
                    reranking_candidates=config.reranking_candidates,
                )

                sr = processor.audio_sampling_rate

                # Save target for this prompt
                target_out = tmp_path / f"{audio_path.stem}-target-{safe_prompt}.wav"
                target_audio = result.target[0].unsqueeze(0).cpu()
                torchaudio.save(target_out.as_posix(), target_audio, sr)
                target_files_by_prompt[prompt].append(target_out)

                # Save residual for this prompt (from original)
                residual_out = tmp_path / f"{audio_path.stem}-residual-{safe_prompt}.wav"
                residual_tensor = result.residual[0]
                torchaudio.save(residual_out.as_posix(), residual_tensor.unsqueeze(0).cpu(), sr)
                residual_files_by_prompt[prompt].append(residual_out)

                # Store residual tensor for chaining (only keep, don't process yet)
                segment_residuals[prompt] = residual_tensor

            # Step 2: Chain subsequent prompts on residuals to build cumulative residual
            if len(prompts_list) > 1:
                current_residual_tensor = segment_residuals[
                    prompts_list[0]
                ]  # Start with residual of first prompt

                for prompt_idx in range(1, len(prompts_list)):
                    prompt = prompts_list[prompt_idx]
                    safe_prompt = prompt.replace(" ", "_").replace("/", "_")
                    logger.debug(f"Processing prompt {prompt} on residual chain")

                    # Save current residual to temp file (processor needs a file path)
                    temp_residual_path = (
                        tmp_path / f"{audio_path.stem}-temp-residual-chain-{prompt_idx}.wav"
                    )
                    torchaudio.save(
                        temp_residual_path.as_posix(),
                        current_residual_tensor.unsqueeze(0).cpu(),
                        sr,
                    )

                    inputs = processor(
                        audios=[temp_residual_path.as_posix()],
                        descriptions=[prompt],
                    ).to(device)

                    result = model.separate(
                        inputs,
                        predict_spans=config.predict_spans,
                        reranking_candidates=config.reranking_candidates,
                    )

                    # Keep the residual for the next iteration
                    current_residual_tensor = result.residual[0]

                # The final residual in the chain is our cumulative residual
                # Save it with a special name so we can reference it later
                cumulative_residual_out = (
                    tmp_path / f"{audio_path.stem}-residual-cumulative-final.wav"
                )
                torchaudio.save(
                    cumulative_residual_out.as_posix(),
                    current_residual_tensor.unsqueeze(0).cpu(),
                    sr,
                )

            logger.info("...done")

    return tmp_path


def phase_2_blend_and_save(
    config,
    logger,
    tmp_path: Path,
    target: str | None,
):
    """
    Phase 2: Blend segments and save final output files.
    Runs regardless of whether we're continuing or not.

    This phase only imports torchaudio (lightweight).
    """
    import torchaudio

    logger.info("=== PHASE 2: Concatenation and Final Output ===")

    target_path = Path(target).expanduser() if target else config.target_dir
    target_path.mkdir(parents=True, exist_ok=True)

    # Load segment files
    segment_files = sorted(
        [
            f
            for f in tmp_path.glob("segment-*.wav")
            if "-target" not in f.name and "-residual" not in f.name and "-temp" not in f.name
        ]
    )

    if not segment_files:
        raise FileNotFoundError(f"No segment files found in {tmp_path}")

    logger.info(f"Found {len(segment_files)} segments")

    # Get sample rate from first TARGET file (what the model actually used)
    first_target = sorted(tmp_path.glob("segment-*-target-*.wav"))
    if first_target:
        audio, sr_int = torchaudio.load(first_target[0])
        sr = int(sr_int)
        logger.info(f"Using sample rate from model output files: {sr}")
    else:
        # Fallback to original segment sample rate
        audio, sr_int = torchaudio.load(segment_files[0])
        sr = int(sr_int)
        logger.warning(f"No target files found yet, using original segment sample rate: {sr}")

    # Find all prompts from target files
    prompts = {}
    for f in tmp_path.glob("segment-*-target-*.wav"):
        # Extract prompt from filename: segment-000-target-bass.wav -> bass
        parts = f.stem.split("-target-")
        if len(parts) == 2:
            prompt = parts[1]
            if prompt not in prompts:
                prompts[prompt] = []
            prompts[prompt].append(f)

    # Concatenate and save each prompt as sam-{prompt}.wav
    for prompt in sorted(prompts.keys()):
        logger.info(f"Concatenating target files for prompt: {prompt}")
        target_files = sorted(prompts[prompt])

        concatenated_target = concatenate_segments(target_files)
        output = target_path / f"sam-{prompt}.wav"
        torchaudio.save(output.as_posix(), concatenated_target, sr)
        logger.info(f"Saved {output}")

    # Concatenate and save cumulative residual as sam-other.wav
    cumulative_files = sorted(tmp_path.glob("segment-*-residual-cumulative-final.wav"))
    if cumulative_files:
        logger.info("Concatenating cumulative residual files...")

        concatenated_cumulative = concatenate_segments(cumulative_files)
        output = target_path / "sam-other.wav"
        torchaudio.save(output.as_posix(), concatenated_cumulative, sr)
        logger.info(f"Saved {output}")
    else:
        logger.warning("No cumulative residual files found")


@click.command(name="test-run3")
@click.option(
    "--src",
    type=click.Path(exists=True),
    help="Source audio file (MP4 or WAV). Overrides config.",
)
@click.option(
    "--target",
    type=click.Path(),
    help="Target output directory. Overrides config.",
)
@click.option(
    "--prompts",
    multiple=True,
    type=str,
    help="Text prompts to separate. Overrides config.",
)
@click.option(
    "--continue-from",
    type=click.Path(exists=True),
    help="Continue from existing temp directory (skip phase 1).",
)
@click.pass_context
def test_run3(
    ctx: click.Context,
    src: str | None,
    target: str | None,
    prompts: tuple[str, ...],
    continue_from: str | None,
) -> None:
    """
    Separate audio sources using SAM-Audio with two-phase processing.

    Phase 1: Segment audio and process with model (skipped if --continue-from)
    Phase 2: Blend segments and save final outputs
    """
    try:
        app_context: AppContext = ctx.obj
        logger = app_context.logger
        config = app_context.app_config

        logger.info("Starting...")
        logger.info(f"Target: {target or config.target_dir}")
        logger.info(f"Prompts: {list(prompts) if prompts else config.prompts}")

        # Override config with CLI arguments if provided
        if src:
            config.source_file = Path(src)
        if target:
            config.target_dir = Path(target).expanduser()
        if prompts:
            config.prompts = list(prompts)

        # Phase 1: Segment and process (only if not continuing)
        if continue_from:
            logger.info(f"Continuing from: {continue_from}")
            tmp_path = Path(continue_from)
        else:
            tmp_path = phase_1_segment_and_process(config, logger, src, prompts)

        # Phase 2: Blend and save (always runs)
        phase_2_blend_and_save(config, logger, tmp_path, target)

        logger.info("All done")

    except Exception as e:
        click.echo(f"CLI Error: {str(e)}")
        ctx.exit(1)
