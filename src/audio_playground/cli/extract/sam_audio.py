"""Composite command for full SAM-Audio extraction pipeline."""

import traceback
import uuid
from datetime import datetime
from pathlib import Path

import click
import torch

from audio_playground.app_context import AppContext
from audio_playground.cli.common import output_dir_option, src_option
from audio_playground.config.app_config import Model
from audio_playground.core.performance_tracker import PerformanceTracker
from audio_playground.core.wav_converter import convert_to_wav, load_audio_duration


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
    "--solver",
    type=click.Choice(["euler", "midpoint"]),
    help="ODE solver method: 'euler' (2x faster) or 'midpoint' (higher quality, default).",
)
@click.option(
    "--solver-steps",
    type=int,
    help="Number of ODE solver steps. Lower = faster but lower quality  .",
)
@click.option(
    "--chunk-duration",
    type=float,
    help="Duration in seconds for each processing chunk  .",
)
@click.option(
    "--chunk-overlap",
    type=float,
    help="Overlap duration in seconds between chunks for smooth transitions  .",
)
@click.option(
    "--crossfade-type",
    type=click.Choice(["cosine", "linear"]),
    help="Crossfade type for blending chunks: 'cosine' (constant power, default) or 'linear'.",
)
@click.option(
    "--no-chunks",
    is_flag=True,
    help="Disable chunking - process entire audio as single chunk (for testing short files).",
)
@click.pass_context
def sam_audio(
    ctx: click.Context,
    src: Path | None,
    output_dir: Path | None,
    prompts: tuple[str, ...],
    model: Model | None = None,
    chain_residuals: bool | None = None,
    sample_rate: int | None = None,
    solver: str | None = None,
    solver_steps: int | None = None,
    chunk_duration: float | None = None,
    chunk_overlap: float | None = None,
    crossfade_type: str | None = None,
    no_chunks: bool = False,
) -> None:
    """
    SAM-Audio extraction pipeline with smart chunking.

    Workflow:
        1. Convert to WAV
        2. Process with SAM-Audio (optimizer handles chunking/crossfade internally)
        3. Save outputs

    The optimizer automatically chunks long audio with overlap and crossfade for
    smooth transitions. Use --no-chunks to disable chunking for short test files.
    """
    try:
        app_context: AppContext = ctx.obj
        logger = app_context.logger
        config = app_context.app_config

        # Override config with CLI arguments
        if src:
            config.source_file = src
        if output_dir:
            config.target_dir = output_dir.expanduser()
        if prompts:
            config.prompts = list(prompts)
        if sample_rate is not None:
            config.sample_rate = sample_rate
        if model is not None:
            config.model_item = model
        if chain_residuals is not None:
            config.chain_residuals = chain_residuals

        # Validate required parameters
        src_path = Path(config.source_file) if config.source_file else None
        if not src_path:
            raise ValueError("No source file specified (use --src or set in config)")

        # Log configuration
        logger.info("Starting SAM-Audio extraction pipeline...")
        logger.info(f"Source: {src_path}")
        logger.info(f"Target: {config.target_dir}")
        logger.info(f"Prompts: {config.prompts}")
        logger.info(f"Model: {config.model_item.value}")
        logger.info(f"Chunking: {'disabled' if no_chunks else 'enabled'}")
        if not no_chunks:
            logger.info(
                f"Chunk settings: duration={chunk_duration or config.chunk_duration}s, "
                f"overlap={chunk_overlap or config.chunk_overlap}s, "
                f"crossfade={crossfade_type or config.crossfade_type}"
            )

        # Generate unique temp directory for this run
        run_id = str(uuid.uuid4())
        base_tmp = Path(config.temp_dir)
        tmp_path = base_tmp / run_id
        tmp_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using temp directory: {tmp_path}")

        # Determine device
        device = config.device
        if device == "auto":
            accelerator = (
                torch.accelerator.current_accelerator()
                if torch.accelerator.is_available()
                else None
            )
            device = accelerator.type if accelerator is not None else "cpu"
            logger.info(f"Auto-detected device: {device}")

        # Initialize performance tracking
        perf_tracker = PerformanceTracker(
            source_file=src_path,
            output_dir=config.target_dir,
            device=device,
        )
        perf_tracker.__enter__()

        # Add metadata
        perf_tracker.add_metadata("prompts", config.prompts)
        perf_tracker.add_metadata("model", config.model_item.value)
        perf_tracker.add_metadata("chunking_enabled", not no_chunks)
        if solver:
            perf_tracker.add_metadata("solver", solver or config.ode_solver)
        if solver_steps:
            perf_tracker.add_metadata("solver_steps", solver_steps or config.ode_steps)

        # Step 1: Convert to WAV
        logger.info("=== Step 1/3: Converting to WAV ===")
        wav_file = tmp_path / "audio.wav"
        convert_to_wav(src_path, wav_file)
        logger.info(f"Converted to: {wav_file}")

        # Load audio duration for tracking
        total_duration = load_audio_duration(wav_file)
        logger.info(f"Total audio length: {total_duration:.2f} seconds")
        perf_tracker.metrics.audio_duration_seconds = total_duration

        # Step 2: Process with SAM-Audio optimizer
        logger.info("=== Step 2/3: Processing with SAM-Audio ===")

        # Lazy imports for performance
        import soundfile as sf
        import torchaudio
        from sam_audio import SAMAudio, SAMAudioProcessor

        from audio_playground.core.sam_audio_optimizer import SolverConfig, process_long_audio

        # Load model
        logger.info(f"Loading SAM-Audio model: {config.model_item.value}")
        model_instance = (
            SAMAudio.from_pretrained(
                config.model_item.value,
                map_location=device,
            )
            .to(device)
            .eval()
        )
        processor = SAMAudioProcessor.from_pretrained(config.model_item.value)
        sr = processor.audio_sampling_rate

        # Setup solver config
        solver_method = solver or config.ode_solver
        steps = solver_steps or config.ode_steps
        solver_config = SolverConfig(method=solver_method, steps=steps)  # type: ignore[arg-type]

        # Process audio (optimizer handles chunking internally)
        with torch.inference_mode():
            # If no_chunks, use very large chunk_duration to process as single chunk
            effective_chunk_duration = (
                999999.0 if no_chunks else (chunk_duration or config.chunk_duration)
            )

            results = process_long_audio(
                audio_path=wav_file,
                prompts=config.prompts,
                model=model_instance,
                processor=processor,
                device=device,
                chunk_duration=effective_chunk_duration,
                overlap_duration=chunk_overlap or config.chunk_overlap,
                crossfade_type=crossfade_type or config.crossfade_type,  # type: ignore[arg-type]
                solver_config=solver_config,
            )

        # Step 3: Save outputs
        logger.info("=== Step 3/3: Saving outputs ===")

        target_path = config.target_dir
        target_path.mkdir(parents=True, exist_ok=True)

        for prompt, audio_tensor in results.items():
            safe_prompt = prompt.replace(" ", "_").replace("/", "_")
            output_filename = f"{safe_prompt}-sam.wav"
            output_path = target_path / output_filename

            # Apply target sample rate if specified
            if config.sample_rate and config.sample_rate != sr:
                logger.info(f"Resampling {prompt} from {sr}Hz to {config.sample_rate}Hz")
                resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
                audio_tensor = resampler(audio_tensor)
                output_sr = config.sample_rate
            else:
                output_sr = sr

            # Save output (move to CPU and convert to numpy for soundfile)
            audio_np = audio_tensor.cpu().numpy()
            sf.write(output_path.as_posix(), audio_np.T, int(output_sr))
            logger.info(f"Saved: {output_path}")

        # Stop performance tracking and save report
        perf_tracker.__exit__(None, None, None)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_path = tmp_path / f"report-{timestamp}-sam-audio.yml"
        perf_tracker.save_report(report_path, format="yaml")

        logger.info("All done!")
        logger.info(f"Output saved to: {target_path}")
        logger.info(f"Performance report: {report_path}")

    except Exception as e:
        logger.error(f"Error occurred: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        click.echo(f"CLI Error: {type(e).__name__}: {str(e) or '(no error message)'}")
        ctx.exit(1)
