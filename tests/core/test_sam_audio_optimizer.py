"""Tests for SAM-Audio optimizer module."""

import math
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torchaudio


@pytest.fixture
def mock_model():
    """Create a mock SAMAudio model."""
    model = MagicMock()
    model.separate = MagicMock()
    return model


@pytest.fixture
def mock_processor():
    """Create a mock SAMAudioProcessor."""
    processor = MagicMock()
    processor.audio_sampling_rate = 44100
    processor.return_value = MagicMock()
    return processor


@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    # Create a simple sine wave
    sample_rate = 44100
    duration = 5.0  # seconds
    frequency = 440.0  # Hz (A4 note)

    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * math.pi * frequency * t).unsqueeze(0)

    audio_path = tmp_path / "test_audio.wav"
    torchaudio.save(audio_path.as_posix(), waveform, sample_rate)

    return audio_path


class TestPromptCache:
    """Tests for PromptCache class."""

    def test_cache_initialization(self):
        """Test that cache initializes with empty state."""
        from audio_playground.core.sam_audio_optimizer import PromptCache

        cache = PromptCache()
        assert cache.hit_rate == 0.0
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["cache_size"] == 0

    def test_hash_prompts_consistency(self):
        """Test that same prompts produce same hash."""
        from audio_playground.core.sam_audio_optimizer import PromptCache

        cache = PromptCache()
        prompts1 = ["bass", "vocals", "drums"]
        prompts2 = ["bass", "vocals", "drums"]

        hash1 = cache._hash_prompts(prompts1)
        hash2 = cache._hash_prompts(prompts2)

        assert hash1 == hash2

    def test_hash_prompts_order_independence(self):
        """Test that prompt order doesn't affect hash (sorted internally)."""
        from audio_playground.core.sam_audio_optimizer import PromptCache

        cache = PromptCache()
        prompts1 = ["bass", "vocals", "drums"]
        prompts2 = ["drums", "bass", "vocals"]

        hash1 = cache._hash_prompts(prompts1)
        hash2 = cache._hash_prompts(prompts2)

        assert hash1 == hash2

    def test_cache_miss(self):
        """Test cache miss behavior."""
        from audio_playground.core.sam_audio_optimizer import PromptCache

        cache = PromptCache()
        prompts = ["bass", "vocals"]
        encoder_fn = Mock(return_value="encoded_result")

        result = cache.get_or_encode(prompts, encoder_fn)

        assert result == "encoded_result"
        encoder_fn.assert_called_once()
        assert cache.stats()["misses"] == 1
        assert cache.stats()["hits"] == 0

    def test_cache_hit(self):
        """Test cache hit behavior."""
        from audio_playground.core.sam_audio_optimizer import PromptCache

        cache = PromptCache()
        prompts = ["bass", "vocals"]
        encoder_fn = Mock(return_value="encoded_result")

        # First call - cache miss
        result1 = cache.get_or_encode(prompts, encoder_fn)
        # Second call - cache hit
        result2 = cache.get_or_encode(prompts, encoder_fn)

        assert result1 == result2
        # Encoder should only be called once
        encoder_fn.assert_called_once()
        assert cache.stats()["misses"] == 1
        assert cache.stats()["hits"] == 1
        assert cache.hit_rate == 0.5

    def test_cache_clear(self):
        """Test cache clearing."""
        from audio_playground.core.sam_audio_optimizer import PromptCache

        cache = PromptCache()
        prompts = ["bass"]
        encoder_fn = Mock(return_value="encoded")

        cache.get_or_encode(prompts, encoder_fn)
        assert cache.stats()["cache_size"] == 1

        cache.clear()
        assert cache.stats()["cache_size"] == 0
        assert cache.stats()["hits"] == 0
        assert cache.stats()["misses"] == 0


class TestSolverConfig:
    """Tests for SolverConfig dataclass."""

    def test_default_config(self):
        """Test default solver configuration."""
        from audio_playground.core.sam_audio_optimizer import SolverConfig

        config = SolverConfig()
        assert config.method == "midpoint"
        assert config.steps == 32

    def test_custom_config(self):
        """Test custom solver configuration."""
        from audio_playground.core.sam_audio_optimizer import SolverConfig

        config = SolverConfig(method="euler", steps=16)
        assert config.method == "euler"
        assert config.steps == 16


class TestCrossfadeWindow:
    """Tests for crossfade window creation."""

    def test_cosine_crossfade(self):
        """Test cosine crossfade window creation."""
        from audio_playground.core.sam_audio_optimizer import create_crossfade_window

        overlap_samples = 100
        fade_out, fade_in = create_crossfade_window(overlap_samples, "cosine")

        assert fade_out.shape == (overlap_samples,)
        assert fade_in.shape == (overlap_samples,)

        # Check that fade_out starts at ~1 and ends at ~0
        assert fade_out[0] > 0.99
        assert fade_out[-1] < 0.01

        # Check that fade_in starts at ~0 and ends at ~1
        assert fade_in[0] < 0.01
        assert fade_in[-1] > 0.99

        # For cosine crossfade, the squares should sum to 1 (constant power)
        # This maintains equal loudness during the crossfade
        power_sum = fade_out**2 + fade_in**2
        assert torch.allclose(power_sum, torch.ones_like(power_sum), atol=0.01)

    def test_linear_crossfade(self):
        """Test linear crossfade window creation."""
        from audio_playground.core.sam_audio_optimizer import create_crossfade_window

        overlap_samples = 100
        fade_out, fade_in = create_crossfade_window(overlap_samples, "linear")

        assert fade_out.shape == (overlap_samples,)
        assert fade_in.shape == (overlap_samples,)

        # Linear should start at 1 and end at 0
        assert fade_out[0] == 1.0
        assert fade_out[-1] == 0.0

        # Linear should start at 0 and end at 1
        assert fade_in[0] == 0.0
        assert fade_in[-1] == 1.0

    def test_invalid_crossfade_type(self):
        """Test that invalid crossfade type raises error."""
        from audio_playground.core.sam_audio_optimizer import create_crossfade_window

        with pytest.raises(ValueError, match="Unknown crossfade_type"):
            create_crossfade_window(100, "invalid")  # type: ignore[arg-type]


class TestClearCaches:
    """Tests for cache clearing functionality."""

    def test_clear_cuda_cache(self):
        """Test CUDA cache clearing."""
        from audio_playground.core.sam_audio_optimizer import clear_caches

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty:
                with patch("torch.cuda.synchronize") as mock_sync:
                    clear_caches("cuda")
                    mock_empty.assert_called_once()
                    mock_sync.assert_called_once()

    def test_clear_mps_cache(self):
        """Test MPS cache clearing."""
        from audio_playground.core.sam_audio_optimizer import clear_caches

        mock_mps = MagicMock()
        with patch("torch.mps", mock_mps):
            clear_caches("mps")
            mock_mps.empty_cache.assert_called_once()

    def test_clear_cpu_cache(self):
        """Test CPU cache clearing (no-op but shouldn't error)."""
        from audio_playground.core.sam_audio_optimizer import clear_caches

        # Should not raise any errors
        clear_caches("cpu")


class TestProcessLongAudio:
    """Tests for chunked processing with crossfade."""

    def test_short_audio_no_chunking(self, temp_audio_file, mock_model, mock_processor):
        """Test that short audio doesn't use chunking."""
        from audio_playground.core.sam_audio_optimizer import process_long_audio

        prompts = ["bass", "vocals"]

        # Mock the model.separate to return a proper result
        mock_result = MagicMock()
        mock_result.target = [torch.randn(1, 220500), torch.randn(1, 220500)]
        mock_model.separate.return_value = mock_result

        # Mock processor to return inputs
        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs

        # Mock torchaudio.info to return audio metadata
        mock_info = MagicMock()
        mock_info.sample_rate = 44100
        mock_info.num_frames = 220500  # 5 seconds at 44100 Hz

        with patch("torchaudio.info", return_value=mock_info):
            results = process_long_audio(
                audio_path=temp_audio_file,
                prompts=prompts,
                model=mock_model,
                processor=mock_processor,
                device="cpu",
                chunk_duration=30.0,  # Longer than test audio
                overlap_duration=2.0,
            )

        assert len(results) == len(prompts)
        # Model should be called once (no chunking)
        assert mock_model.separate.call_count == 1

    def test_long_audio_chunking(self, tmp_path, mock_model, mock_processor):
        """Test that long audio is processed in chunks."""
        from audio_playground.core.sam_audio_optimizer import process_long_audio

        # Create a longer audio file (60 seconds)
        sample_rate = 44100
        duration = 60.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = torch.sin(2 * math.pi * 440 * t).unsqueeze(0)
        audio_path = tmp_path / "long_audio.wav"
        torchaudio.save(audio_path.as_posix(), waveform, sample_rate)

        prompts = ["bass"]

        # Mock the model to return chunks
        def mock_separate(*args, **kwargs):
            result = MagicMock()
            # Return a chunk of audio
            result.target = [torch.randn(1, 1323000)]  # ~30 seconds
            return result

        mock_model.separate.side_effect = mock_separate

        # Mock processor
        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs

        # Mock torchaudio.info for long audio
        mock_info = MagicMock()
        mock_info.sample_rate = sample_rate
        mock_info.num_frames = int(sample_rate * duration)  # 60 seconds

        with patch("torchaudio.info", return_value=mock_info):
            with patch("torchaudio.load") as mock_load:
                # Mock loading chunks
                mock_load.return_value = (torch.randn(1, 1323000), sample_rate)

                results = process_long_audio(
                    audio_path=audio_path,
                    prompts=prompts,
                    model=mock_model,
                    processor=mock_processor,
                    device="cpu",
                    chunk_duration=30.0,
                    overlap_duration=2.0,
                )

        assert "bass" in results
        # Should have called model multiple times (once per chunk)
        assert mock_model.separate.call_count >= 2


class TestProcessStreaming:
    """Tests for streaming processing."""

    def test_streaming_yields_chunks(self, temp_audio_file, mock_model, mock_processor):
        """Test that streaming mode yields chunks."""
        from audio_playground.core.sam_audio_optimizer import process_streaming

        prompts = ["bass"]

        # Mock model
        mock_result = MagicMock()
        mock_result.target = [torch.randn(1, 661500)]  # ~15 seconds
        mock_model.separate.return_value = mock_result

        # Mock processor
        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs

        # Mock torchaudio.info to return audio metadata
        mock_info = MagicMock()
        mock_info.sample_rate = 44100
        mock_info.num_frames = 220500  # 5 seconds at 44100 Hz

        # Mock torchaudio.load for chunk loading
        with patch("torchaudio.info", return_value=mock_info):
            with patch("torchaudio.load", return_value=(torch.randn(1, 661500), 44100)):
                with patch("torchaudio.save"):  # Mock save to avoid file I/O
                    chunks_received = []
                    for prompt, chunk_audio, chunk_idx in process_streaming(
                        audio_path=temp_audio_file,
                        prompts=prompts,
                        model=mock_model,
                        processor=mock_processor,
                        device="cpu",
                        chunk_duration=15.0,
                    ):
                        chunks_received.append((prompt, chunk_idx))

        # Should receive at least one chunk per prompt
        assert len(chunks_received) >= 1
        assert all(prompt == "bass" for prompt, _ in chunks_received)


class TestMemoryStats:
    """Tests for memory statistics."""

    def test_get_memory_stats_cuda(self):
        """Test getting CUDA memory stats."""
        from audio_playground.core.sam_audio_optimizer import get_memory_stats

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.memory_allocated", return_value=1024**3):
                with patch("torch.cuda.memory_reserved", return_value=2 * 1024**3):
                    with patch("torch.cuda.max_memory_allocated", return_value=3 * 1024**3):
                        stats = get_memory_stats("cuda")

        assert "allocated_mb" in stats
        assert "reserved_mb" in stats
        assert "max_allocated_mb" in stats
        assert stats["allocated_mb"] == 1024.0  # 1GB in MB

    def test_get_memory_stats_cpu(self):
        """Test getting CPU memory stats (no stats available)."""
        from audio_playground.core.sam_audio_optimizer import get_memory_stats

        stats = get_memory_stats("cpu")
        # CPU doesn't have detailed memory stats, should return empty dict
        assert isinstance(stats, dict)
