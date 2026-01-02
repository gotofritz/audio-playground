# audio-playground/cli/__main__.py

"""
CLI for the App.

for more info, run

```sh
audio-playground --help
```
"""

import os

import click
from rich.console import Console

from audio_playground.app_context import AppContext
from audio_playground.cli.extract import extract

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], default_map={"obj": {}})


console = Console()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@click.version_option(None, "--version", "-v")
@click.group(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    Main entry point for the CLI.
    """
    # Putting all objects in context so that they don't have to be
    # recreated for each command
    ctx.ensure_object(AppContext)


cli.add_command(extract)


if __name__ == "__main__":
    cli()
