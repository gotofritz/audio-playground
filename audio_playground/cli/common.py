"""Common CLI decorators and options shared across commands."""

from pathlib import Path
from typing import Any, Callable, TypeVar

import click

F = TypeVar("F", bound=Callable[..., Any])


def src_option(required: bool = True, help_text: str = "Source audio file") -> Callable[[F], F]:
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


def target_option(
    required: bool = True, help_text: str = "Target output file path"
) -> Callable[[F], F]:
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
) -> Callable[[F], F]:
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
) -> Callable[[F], F]:
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


def window_size_option(required: bool = False) -> Callable[[F], F]:
    """
    Decorator for --window-size option.

    Args:
        required: Whether the option is required

    Returns:
        Click option decorator

    Note:
        Default value comes from app_config.segment_window_size (10.0)
    """
    return click.option(
        "--window-size",
        type=float,
        default=None,
        required=required,
        help="Segment length in seconds. If not specified, uses config default.",
    )


def max_segments_option() -> Callable[[F], F]:
    """
    Decorator for --max-segments option.

    Returns:
        Click option decorator
    """
    return click.option(
        "--max-segments",
        type=int,
        default=None,
        help="Maximum number of segments (for testing)",
    )
