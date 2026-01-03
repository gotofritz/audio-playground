"""Doctor commands for diagnostics and troubleshooting."""

from pathlib import Path

import click

from audio_playground.app_context import AppContext


@click.group()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """
    Diagnostic commands for troubleshooting audio processing issues.
    """


@doctor.command(name="check-durations")
@click.argument("tmp_dir", type=click.Path(exists=True))
@click.pass_context
def check_durations(ctx: click.Context, tmp_dir: str) -> None:
    """
    Check segment durations to diagnose padding issues.

    Compares original segments with SAM-Audio processed segments to show
    where duration padding is occurring.

    TMP_DIR: Path to temporary directory (e.g., /tmp/sam_audio_split/<uuid>)
    """
    import torchaudio

    app_context: AppContext = ctx.obj
    logger = app_context.logger

    tmp_path = Path(tmp_dir)

    # Check original segments (before model processing)
    logger.info("=== Original Segments (before model) ===")
    segment_files = sorted(tmp_path.glob("segment-[0-9]*.wav"))
    segment_files = [
        f for f in segment_files if "-target" not in f.name and "-residual" not in f.name
    ]

    total_original = 0.0
    for f in segment_files:
        audio, sr = torchaudio.load(f)
        duration = audio.shape[1] / sr
        total_original += duration
        logger.info(f"{f.name}: {duration:.6f}s ({audio.shape[1]} samples @ {sr} Hz)")

    logger.info(f"\nTotal original segments: {total_original:.6f}s")

    # Check target segments (after model processing)
    logger.info("\n=== Target Segments (after model) ===")
    target_files = sorted(tmp_path.glob("segment-*-target-*.wav"))

    if target_files:
        total_target = 0.0
        prompt = target_files[0].stem.split("-target-")[1]
        prompt_files = [f for f in target_files if f"-target-{prompt}.wav" in f.name]

        for f in sorted(prompt_files):
            audio, sr = torchaudio.load(f)
            duration = audio.shape[1] / sr
            total_target += duration
            logger.info(f"{f.name}: {duration:.6f}s ({audio.shape[1]} samples @ {sr} Hz)")

        logger.info(f"\nTotal target segments: {total_target:.6f}s")
        logger.info(f"Difference: {total_target - total_original:.6f}s")

        # Show padding per segment
        if len(segment_files) == len(prompt_files):
            logger.info("\n=== Per-Segment Padding ===")
            for orig_f, target_f in zip(sorted(segment_files), sorted(prompt_files)):
                orig_audio, orig_sr = torchaudio.load(orig_f)
                target_audio, target_sr = torchaudio.load(target_f)
                orig_dur = orig_audio.shape[1] / orig_sr
                target_dur = target_audio.shape[1] / target_sr
                padding = target_dur - orig_dur
                logger.info(
                    f"{orig_f.name} -> {target_f.name}: {padding:+.6f}s "
                    f"({orig_audio.shape[1]} -> {target_audio.shape[1]} samples)"
                )
    else:
        logger.warning("No target files found")
