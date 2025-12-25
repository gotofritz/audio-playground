from pathlib import Path

import click
import torch
import torchaudio
from sam_audio import SAMAudio, SAMAudioProcessor

from audio_playground.app_context import AppContext


@click.command(name="test-run")
@click.pass_context
def test_run(
    ctx: click.Context,
) -> None:
    """
    Just a test run
    """
    try:
        app_context: AppContext = ctx.obj
        logger = app_context.logger
        logger.info("Starting...")

        # up to here it's my usual CLI set up.
        # Now starts the SAM-audio specific code
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

        # Load and process
        audio_path = (Path(".").parent / "wav").absolute().as_posix
        source_path = audio_path + "sources/"
        dest_path = audio_path + "processed/"

        bass_source = source_path + "slap-bass.wav"
        bass_description = "The bass track"
        bass_dest = dest_path + "slap-"

        voice_source = source_path + "voice-over-clapping.wav"
        voice_description = "A man talking"
        voice_dest = dest_path + "voice-"

        with torch.inference_mode():
            inputs = processor(
                audios=[bass_source, voice_source],
                descriptions=[bass_description, voice_description],
            ).to(device)

            result = model.separate(
                inputs,
                predict_spans=False,
                reranking_candidates=8,
            )

            sr = processor.audio_sampling_rate

            target = result.target[0].unsqueeze(0).cpu()
            torchaudio.save(bass_dest + "target.wav", target, sr)
            residual = result.residual[0].unsqueeze(0).cpu()
            torchaudio.save(bass_dest + "residual.wav", residual, sr)

            target = result.target[1].unsqueeze(0).cpu()
            torchaudio.save(voice_dest + "target.wav", target, sr)
            residual = result.residual[1].unsqueeze(0).cpu()
            torchaudio.save(voice_dest + "residual.wav", residual, sr)

        logger.info("Done")

    except Exception as e:
        click.echo(f"CLI Error: {str(e)}")
        ctx.exit(1)
