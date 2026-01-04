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


def pattern_option(
    required: bool = True, default: str | None = None, help_text: str = "File pattern to match"
) -> Callable[[F], F]:
    """
    Decorator for --pattern option.

    Args:
        required: Whether the option is required
        default: Default pattern if not specified
        help_text: Help text for the option

    Returns:
        Click option decorator
    """
    return click.option(
        "--pattern",
        type=str,
        default=default,
        required=required,
        help=help_text,
    )


def suffix_option(default_suffix: str = "sam") -> Callable[[F], F]:
    """
    Decorator for --suffix option used in processing commands.

    The suffix is appended to output filenames (e.g., "bass-sam.wav").

    Args:
        default_suffix: Default suffix if not specified (e.g., "sam", "demucs")

    Returns:
        Click option decorator

    Usage:
        - Not provided: Uses default suffix (e.g., "sam" → "bass-sam.wav")
        - String value: Uses custom suffix (e.g., "my" → "bass-my.wav")
        - False/"": No suffix (e.g., "bass.wav")

    Examples:
        --suffix my          → "bass-my.wav"
        (not provided)       → "bass-sam.wav" (default)
        --suffix ""          → "bass.wav" (no suffix)
    """
    return click.option(
        "--suffix",
        type=str,
        default=default_suffix,
        help=f"Suffix for output files. Default: '{default_suffix}'. Use empty string for no suffix.",
    )
