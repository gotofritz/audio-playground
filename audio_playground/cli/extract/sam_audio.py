"""Composite command for full SAM-Audio extraction pipeline."""

import traceback
import uuid
from pathlib import Path

import click

from audio_playground.app_context import AppContext
from audio_playground.cli.common import (
    max_segments_option,
    output_dir_option,
    src_option,
    window_size_option,
)
from audio_playground.config.app_config import Model


@click.command(name="sam-audio")
@src_option(required=False, help_text="Source audio file (MP4 or WAV). Overrides config.")
@output_dir_option(required=False, help_text="Target output directory. Overrides config.")
@click.option(
    "--prompts",
    multiple=True,
    type=str,
    help="Text prompts to separate. Overrides config.",
)
@click.option(
    "--continue-from",
    type=click.Path(exists=True),
    help="Continue from existing temp directory (skip conversion/segmentation).",
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
@max_segments_option()
@window_size_option()
@click.pass_context
def sam_audio(
    ctx: click.Context,
    src: Path | None,
    output_dir: Path | None,
    prompts: tuple[str, ...],
    continue_from: str | None,
    model: Model | None = None,
    chain_residuals: bool | None = None,
    sample_rate: int | None = None,
    max_segments: int | None = None,
    window_size: float | None = None,
) -> None:
    """
    Composite command: Full SAM-Audio extraction pipeline.

    Workflow:
        1. convert to-wav (src → wav)
        2. segment split (wav → segments)
        3. extract process-sam-audio (segments → processed segments)
        4. merge concat (processed segments → final outputs)

    This command orchestrates the atomic commands to provide a complete
    SAM-Audio extraction workflow with optional residual chaining.
    """
    try:
        app_context: AppContext = ctx.obj
        logger = app_context.logger
        config = app_context.app_config

        # Override config with CLI arguments if provided
        if src:
            config.source_file = src
        if output_dir:
            config.target_dir = output_dir.expanduser()
        if prompts:
            config.prompts = list(prompts)
        if chain_residuals is not None:
            config.chain_residuals = chain_residuals
        if sample_rate is not None:
            config.sample_rate = sample_rate
        if max_segments is not None:
            config.max_segments = max_segments
        if window_size is not None:
            config.segment_window_size = window_size
        if model is not None:
            config.model_item = model

        # Validate required parameters
        src_path = Path(config.source_file) if config.source_file else None
        if not src_path and not src and not continue_from:
            raise ValueError("No source file specified (use --src or set in config)")

        # Log final configuration
        logger.info("Starting SAM-Audio extraction pipeline...")
        logger.info(f"Target: {config.target_dir}")
        logger.info(f"Prompts: {config.prompts}")
        logger.info(f"Chain residuals: {config.chain_residuals}")
        logger.info(f"Segment window size: {config.segment_window_size}s")
        logger.info(f"Model: {config.model_item.value}")
        if config.sample_rate:
            logger.info(f"Target sample rate: {config.sample_rate} Hz")
        if config.max_segments:
            logger.info(f"Max segments: {config.max_segments}")

        # Determine temp directory
        if continue_from:
            logger.info(f"Continuing from: {continue_from}")
            tmp_path = Path(continue_from)
            wav_file = tmp_path / "audio.wav"
            if not wav_file.exists():
                raise FileNotFoundError(f"Cannot continue: {wav_file} not found in {tmp_path}")
        else:
            # Generate unique temp directory for this run
            run_id = str(uuid.uuid4())
            base_tmp = Path(config.temp_dir)
            tmp_path = base_tmp / run_id
            tmp_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using temp directory: {tmp_path}")

            # Step 1: Convert to WAV
            logger.info("=== Step 1/4: Converting to WAV ===")
            from audio_playground.core.wav_converter import convert_to_wav

            # Ensure src_path is not None (validated earlier)
            assert src_path is not None, "Source path must be specified"

            wav_file = tmp_path / "audio.wav"
            convert_to_wav(src_path, wav_file)
            logger.info(f"Converted to: {wav_file}")

            # Step 2: Segment audio
            logger.info("=== Step 2/4: Segmenting audio ===")
            from audio_playground.core.segmenter import (
                create_segments,
                split_to_files,
            )
            from audio_playground.core.wav_converter import load_audio_duration

            total_duration = load_audio_duration(wav_file)
            logger.info(f"Total audio length: {total_duration:.2f} seconds")

            segment_lengths = create_segments(
                total_duration,
                window_size=config.segment_window_size,
                max_segments=config.max_segments,
            )
            logger.info(
                f"Creating {len(segment_lengths)} segments "
                f"({config.segment_window_size}s window): "
                f"{[round(s, 2) for s in segment_lengths]}"
            )

            segment_files, segment_metadata = split_to_files(wav_file, tmp_path, segment_lengths)
            logger.info(f"Created {len(segment_files)} segments in {tmp_path}")

        # Step 3: Process segments with SAM-Audio
        logger.info("=== Step 3/4: Processing with SAM-Audio ===")
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Import processing function
        from audio_playground.cli.extract.process_sam_audio import (
            process_segments_with_sam_audio,
        )

        # Find segment files if continuing
        if continue_from:
            segment_files = sorted(tmp_path.glob("segment-*.wav"))
            if not segment_files:
                raise FileNotFoundError(f"No segment-*.wav files found in {tmp_path}")

        # Determine device
        device = config.device
        if device == "auto":
            import torch

            accelerator = (
                torch.accelerator.current_accelerator()
                if torch.accelerator.is_available()
                else None
            )
            device = accelerator.type if accelerator is not None else "cpu"
            logger.info(f"Auto-detected device: {device}")

        # Process segments
        process_segments_with_sam_audio(
            segment_files=segment_files,
            prompts=config.prompts,
            output_dir=processed_dir,
            model_name=config.model_item.value,
            device=device,
            batch_size=config.batch_prompts,
            logger=logger,
            predict_spans=config.predict_spans,
            reranking_candidates=config.reranking_candidates,
        )

        # Step 4: Merge segments and save final output
        logger.info("=== Step 4/4: Merging and saving final output ===")

        target_path = config.target_dir
        target_path.mkdir(parents=True, exist_ok=True)

        # Merge processed segments for each prompt
        import torchaudio

        from audio_playground.core.merger import concatenate_segments

        for prompt in config.prompts:
            safe_prompt = prompt.replace(" ", "_").replace("/", "_")

            # Find all processed segments for this prompt
            pattern = f"segment-*-{safe_prompt}.wav"
            prompt_segments = sorted(processed_dir.glob(pattern))

            if not prompt_segments:
                logger.warning(
                    f"No processed segments found for prompt '{prompt}' (pattern: {pattern})"
                )
                continue

            logger.info(f"Merging {len(prompt_segments)} segments for prompt '{prompt}'")

            # Concatenate segments
            concatenated = concatenate_segments(prompt_segments)

            # Determine output filename
            output_filename = f"{safe_prompt}-sam.wav"
            output_path = target_path / output_filename

            # Get sample rate from first segment
            _, sr = torchaudio.load(prompt_segments[0])

            # Apply target sample rate if specified
            if config.sample_rate and config.sample_rate != sr:
                logger.info(f"Resampling from {sr}Hz to {config.sample_rate}Hz")
                resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
                concatenated = resampler(concatenated)
                sr = config.sample_rate

            # Save output
            torchaudio.save(output_path.as_posix(), concatenated, int(sr))
            logger.info(f"Saved: {output_path}")

        # Handle residuals if chain_residuals is enabled
        # NOTE: This simplified version does not implement full residual chaining
        # For now, we just log a warning if chain_residuals is enabled
        if config.chain_residuals and len(config.prompts) > 1:
            logger.warning(
                "Residual chaining is not yet implemented in the composite command. "
                "This feature will be added in a future update."
            )

        logger.info("All done!")
        logger.info(f"Output saved to: {target_path}")

    except Exception as e:
        logger.error(f"Error occurred: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        click.echo(f"CLI Error: {type(e).__name__}: {str(e) or '(no error message)'}")
        ctx.exit(1)
