"""Tests for the performance tracker module."""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from audio_playground.core.performance_tracker import (
    PerformanceMetrics,
    PerformanceTracker,
    performance_tracker,
)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""

    def test_metrics_initialization(self) -> None:
        """Test that metrics are initialized with correct defaults."""
        metrics = PerformanceMetrics(command_name="test-command")

        assert metrics.command_name == "test-command"
        assert metrics.end_time is None
        assert metrics.duration_seconds is None
        assert metrics.peak_memory_mb == 0.0
        assert metrics.start_memory_mb == 0.0
        assert metrics.memory_delta_mb == 0.0
        assert isinstance(metrics.additional_metadata, dict)
        assert len(metrics.additional_metadata) == 0

    def test_metrics_finalize(self) -> None:
        """Test that finalize computes duration correctly."""
        start = time.time()
        metrics = PerformanceMetrics(command_name="test-command", start_time=start)

        time.sleep(0.1)  # Sleep for 100ms
        metrics.finalize()

        assert metrics.end_time is not None
        assert metrics.duration_seconds is not None
        assert metrics.duration_seconds >= 0.1
        assert metrics.duration_seconds < 0.2  # Should be around 100ms

    def test_metrics_to_dict(self) -> None:
        """Test that metrics convert to dictionary correctly."""
        metrics = PerformanceMetrics(
            command_name="test-command",
            start_time=time.time(),
            peak_memory_mb=100.5,
            start_memory_mb=50.2,
            memory_delta_mb=50.3,
        )
        metrics.additional_metadata = {"key": "value", "count": 42}
        metrics.finalize()

        result = metrics.to_dict()

        assert result["command"] == "test-command"
        assert "execution_time" in result
        assert "duration_seconds" in result["execution_time"]
        assert "duration_formatted" in result["execution_time"]
        assert "memory_usage" in result
        assert result["memory_usage"]["peak_mb"] == 100.5
        assert result["memory_usage"]["start_mb"] == 50.2
        assert result["memory_usage"]["delta_mb"] == 50.3
        assert result["metadata"] == {"key": "value", "count": 42}

    def test_format_duration_seconds(self) -> None:
        """Test duration formatting for seconds."""
        assert PerformanceMetrics._format_duration(5.5) == "5.50s"
        assert PerformanceMetrics._format_duration(0.123) == "0.12s"
        assert PerformanceMetrics._format_duration(59.9) == "59.90s"

    def test_format_duration_minutes(self) -> None:
        """Test duration formatting for minutes."""
        assert PerformanceMetrics._format_duration(60) == "1m 0.0s"
        assert PerformanceMetrics._format_duration(90.5) == "1m 30.5s"
        assert PerformanceMetrics._format_duration(3599) == "59m 59.0s"

    def test_format_duration_hours(self) -> None:
        """Test duration formatting for hours."""
        assert PerformanceMetrics._format_duration(3600) == "1h 0m 0s"
        assert PerformanceMetrics._format_duration(3661) == "1h 1m 1s"
        assert PerformanceMetrics._format_duration(7325) == "2h 2m 5s"


class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""

    def test_tracker_initialization(self, tmp_path: Path) -> None:
        """Test tracker initialization."""
        logger = Mock()
        tracker = PerformanceTracker(
            command_name="test-command",
            output_dir=tmp_path,
            enabled=True,
            logger=logger,
        )

        assert tracker.command_name == "test-command"
        assert tracker.output_dir == tmp_path
        assert tracker.enabled is True
        assert tracker.logger == logger
        assert isinstance(tracker.metrics, PerformanceMetrics)

    def test_tracker_disabled(self, tmp_path: Path) -> None:
        """Test that disabled tracker does nothing."""
        tracker = PerformanceTracker(
            command_name="test-command",
            output_dir=tmp_path,
            enabled=False,
        )

        tracker.start()
        tracker.add_metadata("key", "value")
        tracker.update_peak_memory()
        metrics = tracker.stop()

        # Metrics should be created but not populated
        assert metrics.duration_seconds is None
        assert metrics.additional_metadata == {}

    def test_tracker_start_stop(self, tmp_path: Path) -> None:
        """Test basic start/stop functionality."""
        tracker = PerformanceTracker(
            command_name="test-command",
            output_dir=tmp_path,
        )

        tracker.start()
        time.sleep(0.1)
        metrics = tracker.stop()

        assert metrics.duration_seconds is not None
        assert metrics.duration_seconds >= 0.1
        assert metrics.peak_memory_mb > 0.0  # Should have some memory usage

    def test_add_metadata(self, tmp_path: Path) -> None:
        """Test adding metadata."""
        tracker = PerformanceTracker(
            command_name="test-command",
            output_dir=tmp_path,
        )

        tracker.add_metadata("key1", "value1")
        tracker.add_metadata("key2", 42)
        tracker.add_metadata("key3", {"nested": "dict"})

        assert tracker.metrics.additional_metadata["key1"] == "value1"
        assert tracker.metrics.additional_metadata["key2"] == 42
        assert tracker.metrics.additional_metadata["key3"] == {"nested": "dict"}

    def test_update_peak_memory(self, tmp_path: Path) -> None:
        """Test peak memory tracking."""
        tracker = PerformanceTracker(
            command_name="test-command",
            output_dir=tmp_path,
        )

        tracker.start()
        initial_peak = tracker.metrics.peak_memory_mb

        # Update peak memory
        tracker.update_peak_memory()

        # Peak should be at least the initial value
        assert tracker.metrics.peak_memory_mb >= initial_peak

    def test_save_report(self, tmp_path: Path) -> None:
        """Test saving performance report."""
        tracker = PerformanceTracker(
            command_name="test-command",
            output_dir=tmp_path,
        )

        tracker.start()
        tracker.add_metadata("test_key", "test_value")
        time.sleep(0.1)
        tracker.stop()

        report_path = tracker.save_report(filename="test_report.yaml")

        assert report_path.exists()
        assert report_path.parent == tmp_path
        assert report_path.name == "test_report.yaml"

        # Load and verify report content
        with open(report_path) as f:
            report_data = yaml.safe_load(f)

        assert "performance_report" in report_data
        assert "generated_at" in report_data
        assert report_data["performance_report"]["command"] == "test-command"
        assert report_data["performance_report"]["metadata"]["test_key"] == "test_value"

    def test_save_report_auto_filename(self, tmp_path: Path) -> None:
        """Test saving report with auto-generated filename."""
        tracker = PerformanceTracker(
            command_name="test command",
            output_dir=tmp_path,
        )

        tracker.start()
        tracker.stop()

        report_path = tracker.save_report()

        assert report_path.exists()
        assert report_path.parent == tmp_path
        assert "performance_test_command" in report_path.name
        assert report_path.suffix == ".yaml"

    def test_save_report_disabled_raises_error(self, tmp_path: Path) -> None:
        """Test that saving report when disabled raises error."""
        tracker = PerformanceTracker(
            command_name="test-command",
            output_dir=tmp_path,
            enabled=False,
        )

        with pytest.raises(RuntimeError, match="Cannot save report: tracking is disabled"):
            tracker.save_report()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test using tracker as context manager."""
        tracker = PerformanceTracker(
            command_name="test-command",
            output_dir=tmp_path,
        )

        with tracker.track() as metrics:
            assert isinstance(metrics, PerformanceMetrics)
            time.sleep(0.1)

        # Metrics should be finalized after context
        assert metrics.duration_seconds is not None
        assert metrics.duration_seconds >= 0.1

    @patch("audio_playground.core.performance_tracker.psutil.Process")
    def test_memory_tracking_error_handling(self, mock_process_class: Mock, tmp_path: Path) -> None:
        """Test that memory tracking errors are handled gracefully."""
        # Make memory_info() raise an exception
        mock_process = Mock()
        mock_process.memory_info.side_effect = Exception("Memory error")
        mock_process_class.return_value = mock_process

        logger = Mock()
        tracker = PerformanceTracker(
            command_name="test-command",
            output_dir=tmp_path,
            logger=logger,
        )

        # Should not raise, just log warning and return 0
        memory = tracker._get_memory_usage_mb()
        assert memory == 0.0
        logger.warning.assert_called_once()


class TestPerformanceTrackerDecorator:
    """Tests for the performance_tracker decorator."""

    def test_decorator_basic(self, tmp_path: Path) -> None:
        """Test basic decorator functionality."""

        @performance_tracker(command_name="test-func", output_dir=tmp_path, save_report=False)
        def test_function() -> str:
            time.sleep(0.1)
            return "result"

        result = test_function()

        assert result == "result"
        # Report file should not exist since save_report=False
        reports = list(tmp_path.glob("performance_*.yaml"))
        assert len(reports) == 0

    def test_decorator_with_save_report(self, tmp_path: Path) -> None:
        """Test decorator with automatic report saving."""

        @performance_tracker(command_name="test-func", output_dir=tmp_path, save_report=True)
        def test_function() -> str:
            time.sleep(0.1)
            return "result"

        result = test_function()

        assert result == "result"
        # Report file should exist
        reports = list(tmp_path.glob("performance_*.yaml"))
        assert len(reports) == 1

    def test_decorator_disabled(self, tmp_path: Path) -> None:
        """Test that disabled decorator doesn't save report."""

        @performance_tracker(
            command_name="test-func", output_dir=tmp_path, save_report=True, enabled=False
        )
        def test_function() -> str:
            return "result"

        result = test_function()

        assert result == "result"
        # No report should be saved when disabled
        reports = list(tmp_path.glob("performance_*.yaml"))
        assert len(reports) == 0

    def test_decorator_extracts_output_dir_from_kwargs(self, tmp_path: Path) -> None:
        """Test that decorator extracts output_dir from function kwargs."""

        @performance_tracker(command_name="test-func", save_report=True)
        def test_function(output_dir: Path) -> str:
            return "result"

        result = test_function(output_dir=tmp_path)

        assert result == "result"
        # Report should be saved to the output_dir from kwargs
        reports = list(tmp_path.glob("performance_*.yaml"))
        assert len(reports) == 1

    def test_decorator_extracts_logger_from_app_context(self, tmp_path: Path) -> None:
        """Test that decorator extracts logger from AppContext."""

        class MockAppContext:
            def __init__(self):
                self.logger = Mock()
                self.app_config = Mock()
                self.app_config.target_dir = tmp_path

        @performance_tracker(command_name="test-func", save_report=True)
        def test_function(app_context: MockAppContext) -> str:
            return "result"

        mock_context = MockAppContext()
        result = test_function(mock_context)

        assert result == "result"
        # Should have used logger from app_context
        assert mock_context.logger.debug.call_count > 0

    def test_decorator_uses_function_name_as_default(self, tmp_path: Path) -> None:
        """Test that decorator uses function name when command_name not provided."""

        @performance_tracker(output_dir=tmp_path, save_report=True)
        def my_custom_function() -> str:
            return "result"

        result = my_custom_function()

        assert result == "result"
        # Check that report uses function name
        reports = list(tmp_path.glob("performance_my_custom_function_*.yaml"))
        assert len(reports) == 1

    def test_decorator_handles_exceptions(self, tmp_path: Path) -> None:
        """Test that decorator still saves report even if function raises exception."""

        @performance_tracker(command_name="test-func", output_dir=tmp_path, save_report=True)
        def failing_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        # Report should still be saved
        reports = list(tmp_path.glob("performance_*.yaml"))
        assert len(reports) == 1


class TestPerformanceTrackerIntegration:
    """Integration tests for performance tracker."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow from start to report."""
        logger = Mock()
        tracker = PerformanceTracker(
            command_name="integration-test",
            output_dir=tmp_path,
            logger=logger,
        )

        # Start tracking
        tracker.start()

        # Simulate some work
        time.sleep(0.1)
        tracker.add_metadata("files_processed", 10)
        tracker.add_metadata("input_size_mb", 125.5)

        # Update memory during execution
        tracker.update_peak_memory()

        # Stop tracking
        metrics = tracker.stop()

        # Save report
        report_path = tracker.save_report()

        # Verify metrics
        assert metrics.duration_seconds >= 0.1
        assert metrics.peak_memory_mb > 0
        assert metrics.additional_metadata["files_processed"] == 10
        assert metrics.additional_metadata["input_size_mb"] == 125.5

        # Verify report file
        assert report_path.exists()
        with open(report_path) as f:
            report_data = yaml.safe_load(f)

        assert report_data["performance_report"]["command"] == "integration-test"
        assert report_data["performance_report"]["metadata"]["files_processed"] == 10
        assert report_data["performance_report"]["execution_time"]["duration_seconds"] >= 0.1

        # Verify logger was called
        assert logger.debug.call_count >= 2  # Start and stop
        assert logger.info.call_count >= 1  # Report saved
