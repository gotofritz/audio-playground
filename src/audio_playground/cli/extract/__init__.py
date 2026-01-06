import click

from audio_playground.cli.extract.demucs import demucs
from audio_playground.cli.extract.sam_audio import sam_audio


@click.group()
@click.pass_context
def extract(
    ctx: click.Context,
) -> None:
    """
    Commands to split sound files into instruments
    """


extract.add_command(sam_audio, name="sam-audio")
extract.add_command(demucs)
