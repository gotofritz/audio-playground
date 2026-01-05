import enum
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from audio_playground import __app_name__


class Model(enum.StrEnum):
    LARGE = "facebook/sam-audio-small"
    SMALL = "facebook/sam-audio-large"


class AudioPlaygroundConfig(BaseSettings):
    app_name: str = __app_name__
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    # Audio processing settings
    source_file: Path | None = None
    target_dir: Path = Path("../wav")
    temp_dir: Path = Path("/tmp/sam_audio_split")
    prompts: list[str] = ["bass"]

    # Model configuration
    model_item: Model = Model.SMALL
    predict_spans: bool = False
    reranking_candidates: int = 1
    overlap_ms: int = 1000

    device: str = "auto"

    batch_prompts: int = Field(
        default=2, description="Number of prompts to process in batch (1=sequential, 2+=batch)"
    )

    chain_residuals: bool = Field(
        default=False,
        description="Chain residuals to compute cumulative residual (sam-other.wav) when multiple prompts are used",
    )

    sample_rate: int | None = Field(
        default=None,
        description="Target sample rate in Hz for output files. If None, uses original sample rate.",
    )

    # Demucs model configuration
    demucs_model: str = Field(
        default="htdemucs_ft",
        description="Demucs model name (e.g., htdemucs, htdemucs_ft, htdemucs_6s)",
    )
    demucs_shifts: int = Field(
        default=6,
        description="Number of random shifts for equivariant stabilization in Demucs (higher=better quality but slower)",
    )
    demucs_num_workers: int = Field(
        default=4,
        description="Number of worker threads for Demucs processing",
    )
    demucs_progress: bool = Field(
        default=True,
        description="Show progress bar during Demucs processing",
    )
    demucs_suffix: str = Field(
        default="demucs",
        description="Suffix for Demucs output files (e.g., 'drums-demucs.wav'). Use empty string for no suffix.",
    )

    # Performance optimization settings (Phase 4)
    chunk_duration: float = Field(
        default=10.0,
        description="Duration in seconds for chunked processing of long audio files",
    )
    chunk_overlap: float = Field(
        default=2.0,
        description="Overlap duration in seconds between chunks (for smooth crossfading)",
    )
    crossfade_type: Literal["cosine", "linear"] = Field(
        default="cosine",
        description="Type of crossfade for blending chunks (cosine=smoother, linear=simpler)",
    )
    ode_solver: Literal["euler", "midpoint"] = Field(
        default="midpoint",
        description="ODE solver method (euler=faster but lower quality, midpoint=higher quality)",
    )
    ode_steps: int = Field(
        default=32,
        description="Number of ODE solver steps (lower=faster but lower quality, typical range: 16-64)",
    )
    streaming_mode: bool = Field(
        default=False,
        description="Enable streaming mode to yield chunks as ready (enables progress monitoring)",
    )

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
        validate_assignment=True,
    )
