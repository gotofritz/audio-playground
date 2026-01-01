import shutil
import subprocess
from pathlib import Path

import click
import torch
import torchaudio
from pydub import AudioSegment
from sam_audio import SAMAudio, SAMAudioProcessor

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


@click.command(name="test-run2")
@click.option(
    "--src",
    required=True,
    type=click.Path(exists=True),
    help="Source audio file (MP4 or WAV)",
)
@click.option(
    "--target",
    default="../wav",
    type=click.Path(),
    help="Target output directory",
)
@click.pass_context
def test_run2(ctx: click.Context, src: str, target: str) -> None:
    """
    Separate audio sources using SAM-Audio
    """
    try:
        app_context: AppContext = ctx.obj
        logger = app_context.logger
        logger.info("Starting...")

        src_path = Path(src)
        target_path = Path(target).expanduser()
        target_path.mkdir(parents=True, exist_ok=True)

        tmp_path = Path("/tmp/sam_audio_split")
        tmp_path.mkdir(parents=True, exist_ok=True)

        # Convert to WAV if needed
        wav_file = tmp_path / "audio.wav"
        if src_path.suffix.lower() == ".mp4":
            logger.info(f"Converting {src_path} to WAV...")
            subprocess.run(
                ["ffmpeg", "-i", src_path.as_posix(), "-c:a", "pcm_s16le", wav_file.as_posix()],
                check=True,
                capture_output=True,
            )
        elif src_path.suffix.lower() == ".wav":
            shutil.copy(src_path, wav_file)
        else:
            # Try to load and convert
            audio = AudioSegment.from_file(src_path)
            audio.export(wav_file.as_posix(), format="wav")

        # Load audio and split into segments
        audio = AudioSegment.from_file(wav_file.as_posix())
        total_length = audio.duration_seconds
        logger.info(f"Total audio length: {total_length:.2f} seconds")

        segments = create_segments(total_length, min_length=9, max_length=17)
        logger.info(f"Creating {len(segments)} segments: {[round(s, 2) for s in segments]}")

        # Split audio into segments
        segment_files = []
        current_time = 0
        for i, segment_length in enumerate(segments):
            segment = audio[int(current_time) : int(current_time + segment_length * 1000)]
            segment_path = tmp_path / f"segment-{i:03d}.wav"
            segment.export(segment_path.as_posix(), format="wav")
            segment_files.append(segment_path)
            current_time += segment_length * 1000
            logger.debug(f"Created {segment_path.name} ({segment_length:.2f}s)")

        # Setup model
        device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        logger.info(f"Using {device} device")

        model = (
            SAMAudio.from_pretrained(
                "facebook/sam-audio-large",
                map_location=device,
            )
            .to(device)
            .eval()
        )
        processor = SAMAudioProcessor.from_pretrained(
            "facebook/sam-audio-large",
        )

        bass_description = "the bass"
        target_files = []
        residual_files = []

        # Process each segment
        with torch.inference_mode():
            for idx, audio_path in enumerate(segment_files):
                logger.info(f"Processing {audio_path.name} ({idx + 1}/{len(segment_files)})")
                inputs = processor(
                    audios=[audio_path.as_posix()],
                    descriptions=[bass_description],
                ).to(device)

                result = model.separate(
                    inputs,
                    predict_spans=False,
                    reranking_candidates=8,
                )

                sr = processor.audio_sampling_rate

                # Save target
                target_out = tmp_path / f"{audio_path.stem}-target.wav"
                target_audio = result.target[0].unsqueeze(0).cpu()
                torchaudio.save(target_out.as_posix(), target_audio, sr)
                target_files.append(target_out)

                # Save residual
                residual_out = tmp_path / f"{audio_path.stem}-residual.wav"
                residual_audio = result.residual[0].unsqueeze(0).cpu()
                torchaudio.save(residual_out.as_posix(), residual_audio, sr)
                residual_files.append(residual_out)

                logger.info("...done")

        # Concatenate all target files
        logger.info("Concatenating target files...")
        concat_txt = tmp_path / "concat_target.txt"
        with open(concat_txt, "w") as f:
            for file in target_files:
                f.write(f"file '{file.as_posix()}'\n")

        target_output = target_path / "target.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_txt.as_posix(),
                "-c",
                "copy",
                target_output.as_posix(),
            ],
            check=True,
            capture_output=True,
        )
        logger.info(f"Saved target to {target_output}")

        # Concatenate all residual files
        logger.info("Concatenating residual files...")
        concat_txt = tmp_path / "concat_residual.txt"
        with open(concat_txt, "w") as f:
            for file in residual_files:
                f.write(f"file '{file.as_posix()}'\n")

        residual_output = target_path / "residual.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_txt.as_posix(),
                "-c",
                "copy",
                residual_output.as_posix(),
            ],
            check=True,
            capture_output=True,
        )
        logger.info(f"Saved residual to {residual_output}")

        logger.info("All done")

    except Exception as e:
        click.echo(f"CLI Error: {str(e)}")
        ctx.exit(1)
