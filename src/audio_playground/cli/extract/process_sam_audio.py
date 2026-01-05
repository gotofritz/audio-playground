"""Process audio segments with SAM-Audio model."""

import glob
import logging
import traceback
from pathlib import Path

import click

from audio_playground.app_context import AppContext
from audio_playground.cli.common import output_dir_option
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
    model_name: str,
    device: str,
    batch_size: int,
    logger: logging.Logger,
    predict_spans: bool,
    reranking_candidates: int,
    streaming: bool = False,
    solver: str = "midpoint",
    solver_steps: int = 32,
    chunk_duration: float = 30.0,
    chunk_overlap: float = 2.0,
    crossfade_type: str = "cosine",
    enable_prompt_caching: bool = True,
) -> None:
    """
    Process audio segments with SAM-Audio model.

    Args:
        segment_files: List of segment file paths to process
        prompts: List of text prompts for separation
        output_dir: Output directory for processed files
        model_name: Model name/path for SAM-Audio
        device: Device to use (cpu, cuda, mps, etc.)
        batch_size: Number of prompts to process in a batch
        logger: Logger instance
        predict_spans: Enable span prediction
        reranking_candidates: Number of reranking candidates
        streaming: Enable streaming mode
        solver: ODE solver method
        solver_steps: Number of ODE solver steps
        chunk_duration: Duration for chunked processing
        chunk_overlap: Overlap duration between chunks
        crossfade_type: Type of crossfade
        enable_prompt_caching: Enable prompt caching
    """
    # Lazy imports for performance
    import torch
    import torchaudio
    from sam_audio import SAMAudio, SAMAudioProcessor

    from audio_playground.core.sam_audio_optimizer import (
        SolverConfig,
        process_long_audio,
        process_streaming,
    )

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

    # Setup solver config
    solver_config = SolverConfig(method=solver, steps=solver_steps)  # type: ignore[arg-type]

    # Process each segment
    with torch.inference_mode():
        for idx, audio_path in enumerate(segment_files):
            logger.info(f"Processing {audio_path.name} ({idx + 1}/{len(segment_files)})")

            # Check if we should use streaming or chunked processing
            if streaming:
                # Streaming mode: yield chunks as ready
                logger.info("Using streaming mode")
                chunk_results: dict[str, list[Any]] = {prompt: [] for prompt in prompts}

                for prompt, chunk_audio, chunk_idx in process_streaming(
                    audio_path=audio_path,
                    prompts=prompts,
                    model=model,
                    processor=processor,
                    device=device,
                    chunk_duration=chunk_duration,
                    solver_config=solver_config,
                ):
                    logger.debug(f"Received chunk {chunk_idx} for prompt '{prompt}'")
                    chunk_results[prompt].append(chunk_audio)

                # Concatenate all chunks for each prompt and save
                for prompt, chunks in chunk_results.items():
                    if chunks:
                        safe_prompt = prompt.replace(" ", "_").replace("/", "_")
                        output_filename = f"{audio_path.stem}-{safe_prompt}.wav"
                        output_path = output_dir / output_filename

                        concatenated = torch.cat(chunks, dim=1)
                        torchaudio.save(output_path.as_posix(), concatenated, sr)
                        logger.debug(f"Saved: {output_path}")

            else:
                # Use optimized chunked processing
                logger.info("Using chunked processing with crossfade")
                results = process_long_audio(
                    audio_path=audio_path,
                    prompts=prompts,
                    model=model,
                    processor=processor,
                    device=device,
                    chunk_duration=chunk_duration,
                    overlap_duration=chunk_overlap,
                    crossfade_type=crossfade_type,  # type: ignore[arg-type]
                    solver_config=solver_config,
                    enable_caching=enable_prompt_caching,
                )

                # Save results for each prompt
                for prompt, audio_tensor in results.items():
                    safe_prompt = prompt.replace(" ", "_").replace("/", "_")
                    output_filename = f"{audio_path.stem}-{safe_prompt}.wav"
                    output_path = output_dir / output_filename

                    # Ensure audio is 2D (channels, samples)
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)

                    torchaudio.save(output_path.as_posix(), audio_tensor, sr)
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
    is_flag=True,
    default=False,
    help="Enable span prediction.",
)
@click.option(
    "--reranking-candidates",
    type=int,
    default=3,
    help="Number of reranking candidates. Default: 3.",
)
@click.option(
    "--streaming",
    is_flag=True,
    default=False,
    help="Enable streaming mode to yield chunks as ready (for progress monitoring).",
)
@click.option(
    "--solver",
    type=click.Choice(["euler", "midpoint"], case_sensitive=False),
    default=None,
    help="ODE solver method (euler=faster, midpoint=higher quality). Default: midpoint.",
)
@click.option(
    "--solver-steps",
    type=int,
    default=None,
    help="Number of ODE solver steps (lower=faster but lower quality). Default: 32.",
)
@click.option(
    "--chunk-duration",
    type=float,
    default=None,
    help="Duration in seconds for chunked processing of long audio. Default: 30.0.",
)
@click.option(
    "--chunk-overlap",
    type=float,
    default=None,
    help="Overlap duration in seconds between chunks. Default: 2.0.",
)
@click.option(
    "--crossfade-type",
    type=click.Choice(["cosine", "linear"], case_sensitive=False),
    default=None,
    help="Type of crossfade for blending chunks. Default: cosine.",
)
@click.option(
    "--no-prompt-cache",
    is_flag=True,
    default=False,
    help="Disable prompt caching (caching enabled by default for 20-30%% speedup).",
)
@click.pass_context
def process_sam_audio(
    ctx: click.Context,
    segment: tuple[str, ...],
    prompts: str,
    output_dir: Path,
    model: str,
    device: str,
    batch_size: int,
    predict_spans: bool,
    reranking_candidates: int,
    streaming: bool,
    solver: str | None,
    solver_steps: int | None,
    chunk_duration: float | None,
    chunk_overlap: float | None,
    crossfade_type: str | None,
    no_prompt_cache: bool,
) -> None:
    """
    Process audio segment(s) with SAM-Audio model to separate audio sources.

    This command runs the SAM-Audio model on one or more audio segments,
    producing separated audio files for each prompt. Output files are named
    {segment}-{prompt}.wav (e.g., segment-000-bass.wav).

    The suffix for final merged outputs should be applied in the merge step,
    not here.

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

        # Get app config for defaults
        app_config = ctx.obj.app_config

        # Apply defaults from app_config if not specified
        final_solver = solver or app_config.ode_solver
        final_solver_steps = solver_steps or app_config.ode_steps
        final_chunk_duration = chunk_duration or app_config.chunk_duration
        final_chunk_overlap = chunk_overlap or app_config.chunk_overlap
        final_crossfade_type = crossfade_type or app_config.crossfade_type
        enable_prompt_caching = not no_prompt_cache and app_config.enable_prompt_caching
        use_streaming = streaming or app_config.streaming_mode

        # Log configuration
        logger.info(f"Prompts: {prompts_list}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Model: {model}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Streaming mode: {use_streaming}")
        logger.info(f"ODE solver: {final_solver} (steps={final_solver_steps})")
        logger.info(f"Chunk duration: {final_chunk_duration}s (overlap={final_chunk_overlap}s)")
        logger.info(f"Crossfade type: {final_crossfade_type}")
        logger.info(f"Prompt caching: {'enabled' if enable_prompt_caching else 'disabled'}")

        # Process segments
        process_segments_with_sam_audio(
            segment_files=segment_files,
            prompts=prompts_list,
            output_dir=output_dir,
            model_name=model,
            device=device,
            batch_size=batch_size,
            logger=logger,
            predict_spans=predict_spans,
            reranking_candidates=reranking_candidates,
            streaming=use_streaming,
            solver=final_solver,
            solver_steps=final_solver_steps,
            chunk_duration=final_chunk_duration,
            chunk_overlap=final_chunk_overlap,
            crossfade_type=final_crossfade_type,
            enable_prompt_caching=enable_prompt_caching,
        )

        logger.info("All done!")

    except Exception as e:
        logger.error(f"Error occurred: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        click.echo(f"CLI Error: {type(e).__name__}: {str(e) or '(no error message)'}", err=True)
        ctx.exit(1)
