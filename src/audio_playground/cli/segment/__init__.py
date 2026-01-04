import click

from audio_playground.cli.segment.split import split


@click.group()
@click.pass_context
def segment(
    ctx: click.Context,
) -> None:
    """
    Commands to split audio files into segments
    """


segment.add_command(split, name="split")
