"""Separator wrappers for audio separation models."""

from audio_playground.core.separator.demucs import SeparatorDemucs
from audio_playground.core.separator.sam_audio import SeparatorSAMAudio

__all__ = ["SeparatorDemucs", "SeparatorSAMAudio"]
