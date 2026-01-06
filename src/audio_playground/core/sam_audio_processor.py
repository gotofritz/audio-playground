"""Process audio segments with SAM-Audio model."""

import logging
from pathlib import Path

from audio_playground.core.separator.sam_audio import SeparatorSAMAudio


def process_segments_with_sam_audio(
    segment_files: list[Path],
    prompts: list[str],
    output_dir: Path,
    model_name: str,
    device: str,
    batch_size: int,
    logger: logging.Logger,
    predict_spans: bool,
    reranking_candidates: int,
) -> None:
    """
    Process audio segments with SAM-Audio model.

    Args:
        segment_files: List of segment file paths to process
        prompts: List of text prompts for separation
        output_dir: Output directory for processed files
        model_name: Model name/path for SAM-Audio
        device: Device to use (cpu, cuda, mps, etc.)
        batch_size: Number of prompts to process in a batch
        logger: Logger instance
        predict_spans: Enable span prediction
        reranking_candidates: Number of reranking candidates
    """
    # Use the wrapper class to hide implementation details
    separator = SeparatorSAMAudio(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        logger=logger,
        predict_spans=predict_spans,
        reranking_candidates=reranking_candidates,
    )
    separator.separate_segments(
        segment_files=segment_files,
        prompts=prompts,
        output_dir=output_dir,
    )
