"""Process audio segments with SAM-Audio model."""

import glob
import logging
import traceback
from pathlib import Path

import click

from audio_playground.app_context import AppContext
from audio_playground.cli.common import output_dir_option, suffix_option
from audio_playground.config.app_config import Model


def expand_segment_paths(segment_args: tuple[str, ...]) -> list[Path]:
    """
    Expand segment paths including glob patterns.

    Args:
        segment_args: Tuple of segment paths (may include glob patterns)

    Returns:
        List of unique, sorted Path objects

    Example:
        >>> expand_segment_paths(("segment-000.wav", "./segments/segment*.wav"))
        [Path("segment-000.wav"), Path("./segments/segment-001.wav"), ...]
    """
    expanded_paths: set[Path] = set()

    for arg in segment_args:
        # Check if this looks like a glob pattern
        if "*" in arg or "?" in arg or "[" in arg:
            # Expand glob pattern
            matches = glob.glob(arg)
            if not matches:
                raise FileNotFoundError(f"No files matched pattern: {arg}")
            expanded_paths.update(Path(m) for m in matches)
        else:
            # Regular file path
            path = Path(arg)
            if not path.exists():
                raise FileNotFoundError(f"Segment file not found: {path}")
            expanded_paths.add(path)

    # Return sorted list for consistent ordering
    return sorted(expanded_paths)


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


def process_segments_with_sam_audio(
    segment_files: list[Path],
    prompts: list[str],
    output_dir: Path,
    suffix: str,
    model_name: str,
    device: str,
    batch_size: int,
    logger: logging.Logger,
    predict_spans: int,
    reranking_candidates: int,
) -> None:
    """
    Process audio segments with SAM-Audio model.

    Args:
        segment_files: List of segment file paths to process
        prompts: List of text prompts for separation
        output_dir: Output directory for processed files
        suffix: Suffix to append to output filenames (empty string for no suffix)
        model_name: Model name/path for SAM-Audio
        device: Device to use (cpu, cuda, mps, etc.)
        batch_size: Number of prompts to process in a batch
        logger: Logger instance
        predict_spans: Number of spans to predict
        reranking_candidates: Number of reranking candidates
    """
    # Lazy imports for performance
    import torch
    import torchaudio
    from sam_audio import SAMAudio, SAMAudioProcessor

    # Setup model
    logger.info(f"Loading SAM-Audio model: {model_name}")
    logger.info(f"Using device: {device}")

    model = (
        SAMAudio.from_pretrained(
            model_name,
            map_location=device,
        )
        .to(device)
        .eval()
    )
    processor = SAMAudioProcessor.from_pretrained(model_name)
    sr = processor.audio_sampling_rate

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each segment
    with torch.inference_mode():
        for idx, audio_path in enumerate(segment_files):
            logger.info(f"Processing {audio_path.name} ({idx + 1}/{len(segment_files)})")

            # Process prompts in batches for efficiency
            prompt_batches = batch_items(prompts, batch_size)
            logger.debug(
                f"Processing {len(prompts)} prompts in {len(prompt_batches)} batch(es) "
                f"(batch_size={batch_size})"
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
                    predict_spans=predict_spans,
                    reranking_candidates=reranking_candidates,
                )

                # Extract and save individual results for each prompt in the batch
                for prompt_idx_in_batch, prompt in enumerate(prompt_batch):
                    safe_prompt = prompt.replace(" ", "_").replace("/", "_")

                    # Build output filename: {segment_stem}-{prompt}-{suffix}.wav
                    # If suffix is empty, use: {segment_stem}-{prompt}.wav
                    if suffix:
                        output_filename = f"{audio_path.stem}-{safe_prompt}-{suffix}.wav"
                    else:
                        output_filename = f"{audio_path.stem}-{safe_prompt}.wav"

                    output_path = output_dir / output_filename
                    target_audio = result.target[prompt_idx_in_batch].unsqueeze(0).cpu()
                    torchaudio.save(output_path.as_posix(), target_audio, sr)
                    logger.debug(f"Saved: {output_path}")

            logger.info(f"Completed processing {audio_path.name}")


@click.command(name="process-sam-audio")
@click.option(
    "--segment",
    multiple=True,
    required=True,
    type=str,
    help="Segment file(s) to process. Can be specified multiple times or use glob patterns (e.g., './segments/segment*.wav').",
)
@click.option(
    "--prompts",
    required=True,
    type=str,
    help="Comma-separated list of text prompts to separate (e.g., 'bass,vocals,drums').",
)
@output_dir_option(required=True, help_text="Output directory for processed segment files")
@suffix_option(default_suffix="sam")
@click.option(
    "--model",
    type=click.Choice([m.value for m in Model], case_sensitive=False),
    default=Model.LARGE.value,
    help="SAM-Audio model to use.",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device to use (auto, cpu, cuda, mps). Default: auto (detect best available).",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Number of prompts to process in a batch. Default: 8.",
)
@click.option(
    "--predict-spans",
    type=int,
    default=8,
    help="Number of spans to predict. Default: 8.",
)
@click.option(
    "--reranking-candidates",
    type=int,
    default=3,
    help="Number of reranking candidates. Default: 3.",
)
@click.pass_context
def process_sam_audio(
    ctx: click.Context,
    segment: tuple[str, ...],
    prompts: str,
    output_dir: Path,
    suffix: str,
    model: str,
    device: str,
    batch_size: int,
    predict_spans: int,
    reranking_candidates: int,
) -> None:
    """
    Process audio segment(s) with SAM-Audio model to separate audio sources.

    This command runs the SAM-Audio model on one or more audio segments,
    producing separated audio files for each prompt.

    Examples:

        # Process single segment
        audio-playground extract process-sam-audio \\
            --segment segment-000.wav \\
            --prompts "bass,vocals" \\
            --output-dir ./out

        # Process multiple segments
        audio-playground extract process-sam-audio \\
            --segment segment-000.wav \\
            --segment segment-001.wav \\
            --prompts "bass,vocals" \\
            --output-dir ./out

        # Process using glob pattern
        audio-playground extract process-sam-audio \\
            --segment "./segments/segment*.wav" \\
            --prompts "bass,vocals" \\
            --output-dir ./out

        # Custom suffix
        audio-playground extract process-sam-audio \\
            --segment segment-000.wav \\
            --prompts "bass,vocals" \\
            --output-dir ./out \\
            --suffix custom

        # No suffix
        audio-playground extract process-sam-audio \\
            --segment segment-000.wav \\
            --prompts "bass,vocals" \\
            --output-dir ./out \\
            --suffix ""
    """
    try:
        app_context: AppContext = ctx.obj
        logger = app_context.logger

        # Parse prompts
        prompts_list = [p.strip() for p in prompts.split(",") if p.strip()]
        if not prompts_list:
            raise ValueError("At least one prompt must be specified")

        # Expand segment paths (including globs)
        segment_files = expand_segment_paths(segment)
        logger.info(f"Found {len(segment_files)} segment(s) to process")
        for seg_file in segment_files:
            logger.debug(f"  - {seg_file}")

        # Determine device
        if device == "auto":
            import torch

            accelerator = (
                torch.accelerator.current_accelerator()
                if torch.accelerator.is_available()
                else None
            )
            device = accelerator.type if accelerator is not None else "cpu"
            logger.info(f"Auto-detected device: {device}")

        # Log configuration
        logger.info(f"Prompts: {prompts_list}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Suffix: '{suffix}'" if suffix else "Suffix: (none)")
        logger.info(f"Model: {model}")
        logger.info(f"Batch size: {batch_size}")

        # Process segments
        process_segments_with_sam_audio(
            segment_files=segment_files,
            prompts=prompts_list,
            output_dir=output_dir,
            suffix=suffix,
            model_name=model,
            device=device,
            batch_size=batch_size,
            logger=logger,
            predict_spans=predict_spans,
            reranking_candidates=reranking_candidates,
        )

        logger.info("All done!")

    except Exception as e:
        logger.error(f"Error occurred: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        click.echo(f"CLI Error: {type(e).__name__}: {str(e) or '(no error message)'}", err=True)
        ctx.exit(1)
