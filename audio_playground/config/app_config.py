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

    # Segment configuration
    segment_window_size: float = Field(
        default=10.0,
        description="Fixed segment length in seconds. All segments except the last will be this size.",
    )
    max_segments: int | None = Field(
        default=None,
        description="Maximum number of segments to create (None = no limit). Useful for testing.",
    )

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

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
        validate_assignment=True,
    )
