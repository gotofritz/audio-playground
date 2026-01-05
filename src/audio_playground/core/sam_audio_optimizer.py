"""PyTorch performance optimizations for SAM-Audio processing.

This module provides platform-agnostic optimizations that work on all platforms
(Windows, Linux, Mac, CUDA, CPU) to improve processing speed and memory efficiency.

Features:
- Chunked processing: Process long audio files in overlapping chunks with crossfade
- Streaming mode: Yield results chunk-by-chunk as they're ready
- Memory management: Explicit cache clearing between chunks/batches
"""

import logging
import math
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """Configuration for ODE solver parameters.

    Attributes:
        method: Solver method - "euler" (faster) or "midpoint" (higher quality)
        steps: Number of solver steps. Lower=faster but lower quality
    """

    method: Literal["euler", "midpoint"] = "midpoint"
    steps: int = 32


def clear_caches(device: str) -> None:
    """Explicit cache clearing between chunks/batches.

    Args:
        device: Device string (e.g., "cuda", "cpu", "mps")
    """
    # Lazy import to avoid loading torch at module level
    import torch

    if device.startswith("cuda"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared CUDA cache")
    elif device == "mps":
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()  # type: ignore[attr-defined]
            logger.debug("Cleared MPS cache")

    # Clear general PyTorch cache
    if hasattr(torch, "clear_autocast_cache"):
        torch.clear_autocast_cache()


def create_crossfade_window(
    overlap_samples: int, crossfade_type: Literal["cosine", "linear"] = "cosine"
) -> Any:
    """Create a crossfade window for blending overlapping chunks.

    Args:
        overlap_samples: Number of samples in the overlap region
        crossfade_type: Type of crossfade curve ("cosine" or "linear")

    Returns:
        Tensor with crossfade weights for the overlap region
    """
    import torch

    if crossfade_type == "cosine":
        # Cosine fade provides smoother transitions
        # fade_out: 1 -> 0, fade_in: 0 -> 1
        t = torch.linspace(0, 1, overlap_samples)
        fade_out = torch.cos(t * math.pi / 2)
        fade_in = torch.sin(t * math.pi / 2)
    elif crossfade_type == "linear":
        # Linear fade is simpler but may have slight artifacts
        fade_out = torch.linspace(1, 0, overlap_samples)
        fade_in = torch.linspace(0, 1, overlap_samples)
    else:
        raise ValueError(f"Unknown crossfade_type: {crossfade_type}")

    return fade_out, fade_in


def process_long_audio(
    audio_path: Path,
    prompts: list[str],
    model: Any,
    processor: Any,
    device: str,
    chunk_duration: float = 30.0,
    overlap_duration: float = 2.0,
    crossfade_type: Literal["cosine", "linear"] = "cosine",
    solver_config: SolverConfig | None = None,
) -> dict[str, Any]:
    """Process long audio files in overlapping chunks to reduce peak memory.

    Blends chunks with cosine/linear crossfade to avoid artifacts.
    Enables processing of arbitrarily long audio files that would otherwise
    exceed available memory.

    Args:
        audio_path: Path to audio file
        prompts: List of text prompts for separation
        model: SAMAudio model instance
        processor: SAMAudioProcessor instance
        device: Device to use for processing
        chunk_duration: Duration of each chunk in seconds (default: 30.0)
        overlap_duration: Duration of overlap between chunks in seconds (default: 2.0)
        crossfade_type: Type of crossfade ("cosine" or "linear")
        solver_config: Optional solver configuration (NOT YET SUPPORTED by SAMAudio API)

    Returns:
        Dictionary mapping prompt names to separated audio tensors
    """
    # Load audio metadata to determine total duration
    # Using soundfile which is a torchaudio dependency and has cross-version compatibility
    import soundfile as sf
    import torch

    info = sf.info(audio_path.as_posix())
    sample_rate = info.samplerate
    total_duration = info.duration

    logger.info(
        f"Processing {audio_path.name}: duration={total_duration:.1f}s, "
        f"chunk_duration={chunk_duration}s, overlap={overlap_duration}s"
    )

    # If audio is shorter than chunk duration, process normally without chunking
    if total_duration <= chunk_duration:
        logger.debug("Audio shorter than chunk duration, processing without chunking")
        inputs = processor(  # type: ignore[call-non-callable]
            audios=[audio_path.as_posix()] * len(prompts),
            descriptions=prompts,
        ).to(device)

        with torch.inference_mode():
            result = model.separate(inputs)

        # Move to CPU before returning (MPS doesn't support torchaudio.save)
        return {prompt: result.target[i].cpu() for i, prompt in enumerate(prompts)}

    # Calculate chunk parameters
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    hop_samples = chunk_samples - overlap_samples

    # Calculate number of chunks needed
    num_frames = info.frames
    num_chunks = math.ceil((num_frames - overlap_samples) / hop_samples)

    logger.info(f"Will process {num_chunks} chunks with {overlap_duration}s overlap")

    # Initialize output tensors
    outputs: dict[str, list[Any]] = {prompt: [] for prompt in prompts}

    # Process chunks
    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * hop_samples
        end_sample = min(start_sample + chunk_samples, num_frames)

        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate

        logger.debug(
            f"Processing chunk {chunk_idx + 1}/{num_chunks}: {start_time:.1f}s - {end_time:.1f}s"
        )

        # Load chunk using soundfile (avoids torchcodec dependency)
        waveform_np, sr = sf.read(
            audio_path.as_posix(),
            start=start_sample,
            stop=end_sample,
            dtype="float32",
        )

        # Save temporary chunk file (SAMAudio requires file path)
        temp_chunk_path = audio_path.parent / f"_temp_chunk_{chunk_idx}.wav"
        # Use soundfile for compatibility
        sf.write(temp_chunk_path.as_posix(), waveform_np, sr)

        try:
            # Process chunk
            inputs = processor(  # type: ignore[call-non-callable]
                audios=[temp_chunk_path.as_posix()] * len(prompts),
                descriptions=prompts,
            ).to(device)

            with torch.inference_mode():
                # Note: solver_config is not yet supported by SAMAudio model API
                # Kept in function signature for future compatibility
                result = model.separate(inputs)

            # Extract results for each prompt
            for i, prompt in enumerate(prompts):
                chunk_result = result.target[i].cpu()

                # Apply crossfade if not the first chunk
                if chunk_idx > 0 and overlap_samples > 0:
                    fade_out, fade_in = create_crossfade_window(overlap_samples, crossfade_type)

                    # Get the last chunk's overlapping tail
                    prev_tail = outputs[prompt][-1][:, -overlap_samples:]

                    # Get current chunk's overlapping head
                    curr_head = chunk_result[:, :overlap_samples]

                    # Blend the overlap region
                    blended = prev_tail * fade_out + curr_head * fade_in

                    # Replace the tail of previous chunk and head of current chunk
                    outputs[prompt][-1] = outputs[prompt][-1][:, :-overlap_samples]
                    chunk_result = torch.cat([blended, chunk_result[:, overlap_samples:]], dim=1)

                outputs[prompt].append(chunk_result)

        finally:
            # Clean up temporary chunk file
            if temp_chunk_path.exists():
                temp_chunk_path.unlink()

        # Clear GPU cache after each chunk
        clear_caches(device)

    # Concatenate all chunks for each prompt
    final_outputs = {prompt: torch.cat(chunks, dim=1) for prompt, chunks in outputs.items()}

    return final_outputs


def process_streaming(
    audio_path: Path,
    prompts: list[str],
    model: Any,
    processor: Any,
    device: str,
    chunk_duration: float = 15.0,
    solver_config: SolverConfig | None = None,
) -> Generator[tuple[str, Any, int], None, None]:
    """Yield results chunk-by-chunk as they're ready.

    This enables interactive applications and progress monitoring, with first
    audio available in ~10-15s instead of waiting for the full file to process.

    Args:
        audio_path: Path to audio file
        prompts: List of text prompts for separation
        model: SAMAudio model instance
        processor: SAMAudioProcessor instance
        device: Device to use for processing
        chunk_duration: Duration of each chunk in seconds (default: 15.0)
        solver_config: Optional solver configuration (NOT YET SUPPORTED by SAMAudio API)

    Yields:
        Tuples of (prompt, chunk_audio_tensor, chunk_index)
    """
    # Load audio metadata using soundfile for cross-version compatibility
    import soundfile as sf
    import torch

    info = sf.info(audio_path.as_posix())
    sample_rate = info.samplerate
    total_duration = info.duration
    num_frames = info.frames

    chunk_samples = int(chunk_duration * sample_rate)
    num_chunks = math.ceil(num_frames / chunk_samples)

    logger.info(
        f"Streaming {audio_path.name}: duration={total_duration:.1f}s, "
        f"{num_chunks} chunks of {chunk_duration}s each"
    )

    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * chunk_samples
        end_sample = min(start_sample + chunk_samples, num_frames)

        logger.debug(f"Streaming chunk {chunk_idx + 1}/{num_chunks}")

        # Load chunk using soundfile (avoids torchcodec dependency)
        waveform_np, sr = sf.read(
            audio_path.as_posix(),
            start=start_sample,
            stop=end_sample,
            dtype="float32",
        )

        # Save temporary chunk file
        temp_chunk_path = audio_path.parent / f"_temp_stream_chunk_{chunk_idx}.wav"
        # Use soundfile for compatibility
        sf.write(temp_chunk_path.as_posix(), waveform_np, sr)

        try:
            # Process chunk
            inputs = processor(  # type: ignore[call-non-callable]
                audios=[temp_chunk_path.as_posix()] * len(prompts),
                descriptions=prompts,
            ).to(device)

            with torch.inference_mode():
                # Note: solver_config is not yet supported by SAMAudio model API
                # Kept in function signature for future compatibility
                result = model.separate(inputs)

            # Yield results for each prompt
            for i, prompt in enumerate(prompts):
                chunk_result = result.target[i].cpu()
                yield (prompt, chunk_result, chunk_idx)

        finally:
            # Clean up temporary chunk file
            if temp_chunk_path.exists():
                temp_chunk_path.unlink()

        # Clear GPU cache after each chunk
        clear_caches(device)


def get_memory_stats(device: str) -> dict[str, float]:
    """Get current memory usage statistics.

    Args:
        device: Device string (e.g., "cuda", "cpu", "mps")

    Returns:
        Dictionary with memory statistics in MB
    """
    import torch

    stats: dict[str, float] = {}

    if device.startswith("cuda") and torch.cuda.is_available():
        stats["allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        stats["reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        stats["max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    elif device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        stats["allocated_mb"] = (
            torch.mps.current_allocated_memory() / 1024**2  # type: ignore[attr-defined]
        )

    return stats
