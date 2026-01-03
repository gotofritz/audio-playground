import logging
import traceback
import uuid
from pathlib import Path

import click

from audio_playground.app_context import AppContext
from audio_playground.config.app_config import AudioPlaygroundConfig, Model
from audio_playground.core import segmenter, wav_converter


def batch_items(items: list[str], batch_size: int) -> list[list[str]]:
    """
    Split a list of items into batches of specified size.

    Args:
        items: List of items to batch
        batch_size: Maximum size of each batch (must be >= 1)

    Returns:
        List of batches, where each batch is a list of items

    Example:
        >>> batch_items(["a", "b", "c", "d"], 2)
        [["a", "b"], ["c", "d"]]
        >>> batch_items(["a", "b", "c"], 2)
        [["a", "b"], ["c"]]
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    batches: list[list[str]] = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i : i + batch_size])
    return batches


def phase_1_segment_and_process(
    config: AudioPlaygroundConfig,
    logger: logging.Logger,
    src: str | None,
    prompts: tuple[str, ...],
) -> Path:
    """
    Phase 1: Create segments and process them with the model.
    Only runs if NOT using --continue-from.

    This phase imports torch and SAMAudio only if needed.
    """
    import torch
    import torchaudio
    from sam_audio import SAMAudio, SAMAudioProcessor

    logger.info("=== PHASE 1: Segmentation and Processing ===")

    src_path = Path(config.source_file) if config.source_file else None
    if not src_path and not src:
        raise ValueError("No source file specified")
    if src:
        src_path = Path(src)

    # mypy couldn't work this out by itself
    assert src_path is not None

    # Generate unique temp directory for this run
    run_id = str(uuid.uuid4())
    base_tmp = Path(config.temp_dir)
    tmp_path = base_tmp / run_id
    tmp_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using temp directory: {tmp_path}")

    prompts_list: list[str] = list(prompts) if prompts else config.prompts

    logger.info("Creating segments (no overlap, simple concatenation)")

    # Convert to WAV if needed
    wav_file = tmp_path / "audio.wav"
    logger.info(f"Converting {src_path} to WAV...")
    wav_converter.convert_to_wav(src_path, wav_file)

    # Load audio duration and create fixed-size segments
    total_length = wav_converter.load_audio_duration(wav_file)
    logger.info(f"Total audio length: {total_length:.2f} seconds")

    # Create segments with fixed window size
    segment_lengths: list[float] = segmenter.create_segments(
        total_length,
        window_size=config.segment_window_size,
        max_segments=config.max_segments,
    )
    if config.max_segments:
        logger.info(f"Limiting to max {config.max_segments} segments for testing")
    logger.info(
        f"Creating {len(segment_lengths)} segments ({config.segment_window_size}s window): "
        f"{[round(s, 2) for s in segment_lengths]}"
    )

    # Split audio into segment files
    segment_files, segment_metadata = segmenter.split_to_files(wav_file, tmp_path, segment_lengths)
    logger.info(f"Created {len(segment_files)} segments")

    for i, (start_time_s, actual_duration_s) in enumerate(segment_metadata):
        logger.debug(
            f"Created segment-{i:03d}.wav (actual: {actual_duration_s:.2f}s at {start_time_s:.1f}s)"
        )

    # Setup model
    accelerator = (
        torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
    )
    device = (
        config.device
        if config.device != "auto"
        else (accelerator.type if accelerator is not None else "cpu")
    )
    logger.info(f"Using {device} device")

    model = (
        SAMAudio.from_pretrained(
            config.model_item.value,
            map_location=device,
        )
        .to(device)
        .eval()
    )
    processor = SAMAudioProcessor.from_pretrained(
        config.model_item.value,
    )

    # Dictionary to store output files for each prompt
    target_files_by_prompt: dict[str, list[Path]] = {prompt: [] for prompt in prompts_list}
    residual_files_by_prompt: dict[str, list[Path]] = {prompt: [] for prompt in prompts_list}

    # Process each segment
    with torch.inference_mode():
        for idx, audio_path in enumerate(segment_files):
            logger.info(f"Processing {audio_path.name} ({idx + 1}/{len(segment_files)})")

            sr: int | None = None  # Will be set from processor
            segment_residuals: dict[
                str, "torch.Tensor"
            ] = {}  # Store residuals for each prompt from original audio

            # Step 1: Run all prompts on the original audio to get clean targets
            # Process prompts in batches for efficiency
            prompt_batches = batch_items(prompts_list, config.batch_prompts)
            logger.debug(
                f"Processing {len(prompts_list)} prompts in {len(prompt_batches)} batch(es) "
                f"(batch_size={config.batch_prompts})"
            )

            for batch_idx, prompt_batch in enumerate(prompt_batches):
                logger.debug(
                    f"Processing batch {batch_idx + 1}/{len(prompt_batches)}: {prompt_batch}"
                )

                # Process all prompts in this batch together
                # SAM-Audio processor requires len(audios) == len(descriptions)
                # So we duplicate the audio path for each prompt in the batch
                inputs = processor(
                    audios=[audio_path.as_posix()] * len(prompt_batch),
                    descriptions=prompt_batch,
                ).to(device)

                result = model.separate(
                    inputs,
                    predict_spans=config.predict_spans,
                    reranking_candidates=config.reranking_candidates,
                )

                sr = processor.audio_sampling_rate

                # Extract and save individual results for each prompt in the batch
                for prompt_idx_in_batch, prompt in enumerate(prompt_batch):
                    safe_prompt = prompt.replace(" ", "_").replace("/", "_")

                    # Save target for this prompt
                    target_out = tmp_path / f"{audio_path.stem}-target-{safe_prompt}.wav"
                    target_audio = result.target[prompt_idx_in_batch].unsqueeze(0).cpu()
                    torchaudio.save(target_out.as_posix(), target_audio, sr)
                    target_files_by_prompt[prompt].append(target_out)

                    # Save residual for this prompt (from original)
                    residual_out = tmp_path / f"{audio_path.stem}-residual-{safe_prompt}.wav"
                    residual_tensor = result.residual[prompt_idx_in_batch]
                    torchaudio.save(residual_out.as_posix(), residual_tensor.unsqueeze(0).cpu(), sr)
                    residual_files_by_prompt[prompt].append(residual_out)

                    # Store residual tensor for chaining (only keep, don't process yet)
                    segment_residuals[prompt] = residual_tensor

            # Step 2: Chain subsequent prompts on residuals to build cumulative residual
            # Only run if chain_residuals is enabled and there are multiple prompts
            if config.chain_residuals and len(prompts_list) > 1:
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
            else:
                if not config.chain_residuals and len(prompts_list) > 1:
                    logger.debug("Residual chaining disabled (--no-chain-residuals)")

            logger.info("...done")

    return tmp_path


def phase_2_blend_and_save(
    config: AudioPlaygroundConfig,
    logger: logging.Logger,
    tmp_path: Path,
    target: str | None,
) -> None:
    """
    Phase 2: Blend segments and save final output files.
    Runs regardless of whether we're continuing or not.

    This phase uses merger module which lazily imports torchaudio.
    """
    from audio_playground.core import merger

    logger.info("=== PHASE 2: Concatenation and Final Output ===")

    target_path = Path(target).expanduser() if target else config.target_dir

    # Use merger to handle all merging logic
    merger.merge_and_save(tmp_path, target_path, logger, config.chain_residuals, config.sample_rate)


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
@click.option(
    "--model",
    type=click.Choice(Model, case_sensitive=False),
    help="What model to use. Use -h to view list",
)
@click.option(
    "--chain-residuals/--no-chain-residuals",
    default=None,
    help="Chain residuals to compute cumulative residual (sam-other.wav) when multiple prompts used. If not specified, uses config default.",
)
@click.option(
    "--sample-rate",
    type=int,
    help="Target sample rate in Hz for output files (e.g., 44100, 48000). If not specified, uses original sample rate.",
)
@click.option(
    "--max-segments",
    type=int,
    help="Maximum number of segments to create (useful for testing). If not specified, uses calculated number.",
)
@click.option(
    "--segment-window-size",
    type=float,
    help="Fixed segment length in seconds. All segments except the last will be this size. If not specified, uses config default.",
)
@click.pass_context
def sam_audio(
    ctx: click.Context,
    src: str | None,
    target: str | None,
    prompts: tuple[str, ...],
    continue_from: str | None,
    model: Model = Model.LARGE,
    chain_residuals: bool | None = None,
    sample_rate: int | None = None,
    max_segments: int | None = None,
    segment_window_size: float | None = None,
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

        # Override config with CLI arguments if provided
        if src:
            config.source_file = Path(src)
        if target:
            config.target_dir = Path(target).expanduser()
        if prompts:
            config.prompts = list(prompts)
        if chain_residuals is not None:
            config.chain_residuals = chain_residuals
        if sample_rate is not None:
            config.sample_rate = sample_rate
        if max_segments is not None:
            config.max_segments = max_segments
        if segment_window_size is not None:
            config.segment_window_size = segment_window_size

        # Log final configuration
        logger.info("Starting...")
        logger.info(f"Target: {config.target_dir}")
        logger.info(f"Prompts: {config.prompts}")
        logger.info(f"Chain residuals: {config.chain_residuals}")
        logger.info(f"Segment window size: {config.segment_window_size}s")
        if config.sample_rate:
            logger.info(f"Target sample rate: {config.sample_rate} Hz")
        if config.max_segments:
            logger.info(f"Max segments: {config.max_segments}")

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
        logger.error(f"Error occurred: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        click.echo(f"CLI Error: {type(e).__name__}: {str(e) or '(no error message)'}")
        ctx.exit(1)
