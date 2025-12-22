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
        # Initialize
        device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        logger.info(f"Using {device} device")

        model = (
            SAMAudio.from_pretrained("facebook/sam-audio-large", map_location=device)
            .to(device)
            .eval()
        )

        processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")

        # Load and process
        audio_path = "/Users/fritz/.yarkie/videos/r/rZp956sEDHY.mp4"
        description = "Extract Bass"

        with torch.inference_mode():
            inputs = processor(audios=[audio_path], descriptions=[description]).to(device)

            # Separate
            result = model.separate(inputs)
            # result = model.separate(inputs, predict_spans=True, reranking_candidates=8)

            # Save
            sr = processor.audio_sampling_rate
            target = result.target[0].unsqueeze(0).cpu()
            torchaudio.save("bass.wav", target, sr)
            residual = result.residual[0].unsqueeze(0).cpu()
            torchaudio.save("background.wav", residual, sr)

        logger.info("Done")

    except Exception as e:
        click.echo(f"CLI Error: {str(e)}")
        ctx.exit(1)
