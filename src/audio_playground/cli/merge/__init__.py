import click

from audio_playground.cli.merge.concat import concat


@click.group()
@click.pass_context
def merge(
    ctx: click.Context,
) -> None:
    """
    Commands to merge audio segments
    """


merge.add_command(concat, name="concat")
