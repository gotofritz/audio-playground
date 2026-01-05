import click

from audio_playground.cli.convert.to_wav import to_wav


@click.group()
@click.pass_context
def convert(
    ctx: click.Context,
) -> None:
    """
    Commands to convert audio files between formats
    """


convert.add_command(to_wav, name="to-wav")
