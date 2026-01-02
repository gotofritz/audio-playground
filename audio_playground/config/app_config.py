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
    min_segment_length: float = 9.0
    max_segment_length: float = 17.0

    # Model configuration
    model_item: Model = Model.SMALL
    predict_spans: bool = False
    reranking_candidates: int = 8
    overlap_ms: int = 1000

    device: str = "auto"

    batch_prompts: int = Field(
        default=2, description="Number of prompts to process in batch (1=sequential, 2+=batch)"
    )

    chain_residuals: bool = Field(
        default=False,
        description="Chain residuals to compute cumulative residual (sam-other.wav) when multiple prompts are used",
    )

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
        validate_assignment=True,
    )
