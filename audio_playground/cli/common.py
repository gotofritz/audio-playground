"""Common CLI decorators and options shared across commands."""

from pathlib import Path
from typing import Callable

import click


def src_option(required: bool = True, help_text: str = "Source audio file") -> Callable:
    """
    Decorator for --src option.

    Args:
        required: Whether the option is required
        help_text: Help text for the option

    Returns:
        Click option decorator
    """
    return click.option(
        "--src",
        type=click.Path(exists=True, path_type=Path),
        required=required,
        help=help_text,
    )


def target_option(required: bool = True, help_text: str = "Target output file path") -> Callable:
    """
    Decorator for --target option.

    Args:
        required: Whether the option is required
        help_text: Help text for the option

    Returns:
        Click option decorator
    """
    return click.option(
        "--target",
        type=click.Path(path_type=Path),
        required=required,
        help=help_text,
    )


def output_dir_option(
    required: bool = True, help_text: str = "Output directory for results"
) -> Callable:
    """
    Decorator for --output-dir option.

    Args:
        required: Whether the option is required
        help_text: Help text for the option

    Returns:
        Click option decorator
    """
    return click.option(
        "--output-dir",
        type=click.Path(path_type=Path),
        required=required,
        help=help_text,
    )


def input_dir_option(
    required: bool = True, help_text: str = "Input directory containing files"
) -> Callable:
    """
    Decorator for --input-dir option.

    Args:
        required: Whether the option is required
        help_text: Help text for the option

    Returns:
        Click option decorator
    """
    return click.option(
        "--input-dir",
        type=click.Path(exists=True, path_type=Path),
        required=required,
        help=help_text,
    )
