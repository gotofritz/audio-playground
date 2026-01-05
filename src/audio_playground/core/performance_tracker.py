"""Performance tracking for CLI commands.

This module provides utilities for tracking execution time, memory usage,
and other performance metrics for audio processing commands.
"""

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator

import psutil
import yaml


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    command_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_seconds: float | None = None
    peak_memory_mb: float = 0.0
    start_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0
    additional_metadata: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        """Finalize metrics by computing derived values."""
        if self.end_time is None:
            self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for YAML serialization."""
        return {
            "command": self.command_name,
            "execution_time": {
                "duration_seconds": round(self.duration_seconds or 0.0, 3),
                "duration_formatted": self._format_duration(self.duration_seconds or 0.0),
            },
            "memory_usage": {
                "start_mb": round(self.start_memory_mb, 2),
                "peak_mb": round(self.peak_memory_mb, 2),
                "delta_mb": round(self.memory_delta_mb, 2),
            },
            "metadata": self.additional_metadata,
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        if minutes < 60:
            return f"{minutes}m {secs:.1f}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


class PerformanceTracker:
    """Track and report performance metrics for CLI commands."""

    def __init__(
        self,
        command_name: str,
        output_dir: Path | None = None,
        enabled: bool = True,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize performance tracker.

        Args:
            command_name: Name of the command being tracked
            output_dir: Directory to save performance report (None = current directory)
            enabled: Whether tracking is enabled
            logger: Optional logger for output
        """
        self.command_name = command_name
        self.output_dir = output_dir or Path.cwd()
        self.enabled = enabled
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = PerformanceMetrics(command_name=command_name)
        self._process = psutil.Process()

    def start(self) -> None:
        """Start tracking performance."""
        if not self.enabled:
            return

        self.metrics.start_time = time.time()
        self.metrics.start_memory_mb = self._get_memory_usage_mb()
        self.metrics.peak_memory_mb = self.metrics.start_memory_mb
        self.logger.debug(
            f"Performance tracking started for '{self.command_name}' "
            f"(initial memory: {self.metrics.start_memory_mb:.2f} MB)"
        )

    def update_peak_memory(self) -> None:
        """Update peak memory usage."""
        if not self.enabled:
            return

        current_memory = self._get_memory_usage_mb()
        if current_memory > self.metrics.peak_memory_mb:
            self.metrics.peak_memory_mb = current_memory

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add custom metadata to the performance report.

        Args:
            key: Metadata key
            value: Metadata value (must be YAML-serializable)
        """
        if not self.enabled:
            return

        self.metrics.additional_metadata[key] = value

    def stop(self) -> PerformanceMetrics:
        """
        Stop tracking and finalize metrics.

        Returns:
            Performance metrics
        """
        if not self.enabled:
            return self.metrics

        # Update final memory stats
        self.update_peak_memory()
        end_memory = self._get_memory_usage_mb()
        self.metrics.memory_delta_mb = end_memory - self.metrics.start_memory_mb

        # Finalize metrics
        self.metrics.finalize()

        self.logger.debug(
            f"Performance tracking stopped for '{self.command_name}' "
            f"(duration: {self.metrics.duration_seconds:.2f}s, "
            f"peak memory: {self.metrics.peak_memory_mb:.2f} MB)"
        )

        return self.metrics

    def save_report(self, filename: str | None = None) -> Path:
        """
        Save performance report as YAML file.

        Args:
            filename: Custom filename (None = auto-generate)

        Returns:
            Path to saved report
        """
        if not self.enabled:
            raise RuntimeError("Cannot save report: tracking is disabled")

        if filename is None:
            # Auto-generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_command = self.command_name.replace(" ", "_").replace("/", "_")
            filename = f"performance_{safe_command}_{timestamp}.yaml"

        report_path = self.output_dir / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "performance_report": self.metrics.to_dict(),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(report_path, "w") as f:
            yaml.dump(report_data, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Performance report saved to: {report_path}")
        return report_path

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self._process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    @contextmanager
    def track(self) -> Generator[PerformanceMetrics, None, None]:
        """
        Context manager for tracking a code block.

        Usage:
            with tracker.track() as metrics:
                # code to track
                pass
            # metrics are automatically finalized
        """
        self.start()
        try:
            yield self.metrics
        finally:
            self.stop()


def performance_tracker(
    command_name: str | None = None,
    output_dir: Path | None = None,
    save_report: bool = True,
    enabled: bool = True,
) -> Callable:
    """
    Decorator for tracking performance of CLI commands.

    Args:
        command_name: Name of the command (None = use function name)
        output_dir: Directory to save report (None = current directory)
        save_report: Whether to automatically save report
        enabled: Whether tracking is enabled

    Usage:
        @performance_tracker(command_name="convert to-wav")
        def to_wav(app_context, src, target):
            # command implementation
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine command name
            cmd_name = command_name or func.__name__

            # Try to extract output_dir from arguments
            report_dir = output_dir
            if report_dir is None:
                # Check if there's an output_dir in kwargs
                if "output_dir" in kwargs and kwargs["output_dir"]:
                    report_dir = Path(kwargs["output_dir"])
                # Check if first arg is AppContext and has config
                elif args and hasattr(args[0], "app_config"):
                    app_context = args[0]
                    if hasattr(app_context.app_config, "target_dir"):
                        report_dir = Path(app_context.app_config.target_dir)

            # Try to get logger from arguments
            logger = None
            if args and hasattr(args[0], "logger"):
                logger = args[0].logger

            # Create tracker
            tracker = PerformanceTracker(
                command_name=cmd_name,
                output_dir=report_dir,
                enabled=enabled,
                logger=logger,
            )

            # Track execution
            tracker.start()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                tracker.stop()
                if save_report and tracker.enabled:
                    try:
                        tracker.save_report()
                    except Exception as e:
                        if tracker.logger:
                            tracker.logger.warning(f"Failed to save performance report: {e}")

        return wrapper

    return decorator
