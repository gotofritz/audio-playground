"""Performance tracking and reporting for audio processing commands."""

import json
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during audio processing."""

    # File information
    source_file: str
    output_dir: str
    audio_duration_seconds: float | None = None

    # Timing
    start_time: str = ""
    end_time: str = ""
    processing_time_seconds: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0

    # Device
    device: str = "cpu"
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0

    # Additional metrics
    speedup_factor: float | None = None  # audio_duration / processing_time
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """Context manager for tracking performance metrics during audio processing."""

    def __init__(
        self,
        source_file: Path | str,
        output_dir: Path | str,
        audio_duration: float | None = None,
        device: str = "cpu",
    ) -> None:
        """
        Initialize performance tracker.

        Args:
            source_file: Path to source audio file
            output_dir: Path to output directory
            audio_duration: Duration of audio in seconds (if known)
            device: Device being used (cpu, cuda, mps, etc.)
        """
        self.metrics = PerformanceMetrics(
            source_file=str(source_file),
            output_dir=str(output_dir),
            audio_duration_seconds=audio_duration,
            device=device,
        )
        self._start_time = 0.0

    def __enter__(self) -> "PerformanceTracker":
        """Start tracking performance."""
        # Start time tracking
        self._start_time = time.perf_counter()
        self.metrics.start_time = datetime.now().isoformat()

        # Start memory tracking
        tracemalloc.start()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop tracking and calculate final metrics."""
        # Calculate processing time
        end_time = time.perf_counter()
        self.metrics.end_time = datetime.now().isoformat()
        self.metrics.processing_time_seconds = end_time - self._start_time

        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        self.metrics.current_memory_mb = current / (1024 * 1024)
        self.metrics.peak_memory_mb = peak / (1024 * 1024)
        tracemalloc.stop()

        # Get GPU memory if using GPU
        if self.metrics.device in ("cuda", "mps"):
            try:
                import torch

                if self.metrics.device == "cuda" and torch.cuda.is_available():
                    self.metrics.gpu_memory_allocated_mb = (
                        torch.cuda.memory_allocated() / (1024 * 1024)
                    )
                    self.metrics.gpu_memory_reserved_mb = (
                        torch.cuda.memory_reserved() / (1024 * 1024)
                    )
                elif self.metrics.device == "mps" and torch.backends.mps.is_available():
                    # MPS doesn't have direct memory query APIs like CUDA
                    # We'll leave these as 0 for now
                    pass
            except Exception:
                # If GPU memory query fails, just continue
                pass

        # Calculate speedup factor if we have audio duration
        if self.metrics.audio_duration_seconds and self.metrics.processing_time_seconds > 0:
            self.metrics.speedup_factor = (
                self.metrics.audio_duration_seconds / self.metrics.processing_time_seconds
            )

    def add_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata to the performance report."""
        self.metrics.metadata[key] = value

    def save_report(self, report_path: Path | str, format: str = "yaml") -> Path:
        """
        Save performance report to file.

        Args:
            report_path: Path where report should be saved
            format: Output format ('yaml' or 'json')

        Returns:
            Path to saved report file
        """
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_dict = asdict(self.metrics)

        if format == "json":
            with open(report_path, "w") as f:
                json.dump(metrics_dict, f, indent=2)
        else:  # yaml format (manual writing to avoid pyyaml dependency)
            with open(report_path, "w") as f:
                self._write_yaml_dict(f, metrics_dict)

        return report_path

    def _write_yaml_dict(self, f: Any, data: dict[str, Any], indent: int = 0) -> None:
        """Write a dictionary as YAML format."""
        for key, value in data.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                if value:  # Non-empty dict
                    f.write(f"{prefix}{key}:\n")
                    self._write_yaml_dict(f, value, indent + 1)
                else:  # Empty dict
                    f.write(f"{prefix}{key}: {{}}\n")
            elif isinstance(value, list):
                f.write(f"{prefix}{key}:\n")
                for item in value:
                    if isinstance(item, dict):
                        f.write(f"{prefix}  -\n")
                        self._write_yaml_dict(f, item, indent + 2)
                    else:
                        f.write(f"{prefix}  - {self._yaml_value(item)}\n")
            elif value is None:
                f.write(f"{prefix}{key}: null\n")
            else:
                f.write(f"{prefix}{key}: {self._yaml_value(value)}\n")

    def _yaml_value(self, value: Any) -> str:
        """Format a value for YAML output."""
        if isinstance(value, str):
            # Escape strings with special characters
            if any(c in value for c in ["\n", ":", "#", '"', "'"]):
                return f'"{value}"'
            return value
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        return str(value)
