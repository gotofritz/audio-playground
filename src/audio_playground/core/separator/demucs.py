"""Wrapper class for Demucs audio separation model."""

import logging
from pathlib import Path


class SeparatorDemucs:
    """Wrapper for Demucs model to hide implementation details."""

    def __init__(
        self,
        model_name: str,
        device: str,
        shifts: int,
        num_workers: int,
        logger: logging.Logger,
    ):
        """
        Initialize Demucs separator.

        Args:
            model_name: Demucs model name (e.g., htdemucs_ft)
            device: Device to use (cpu, cuda, etc.)
            shifts: Number of random shifts for equivariant stabilization
            num_workers: Number of worker threads
            logger: Logger instance
        """
        self.model_name = model_name
        self.device = device
        self.shifts = shifts
        self.num_workers = num_workers
        self.logger = logger
        self.model = None
        self._samplerate = None

    def load_model(self) -> None:
        """Load the Demucs model."""
        from demucs.pretrained import get_model

        self.logger.info(f"Loading Demucs model: {self.model_name}")
        self.logger.info(f"Using device: {self.device}")

        self.model = get_model(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self._samplerate = self.model.samplerate

    def separate(
        self,
        audio_path: Path,
        output_dir: Path,
        suffix: str = "",
        show_progress: bool = True,
    ) -> None:
        """
        Separate audio file into stems.

        Args:
            audio_path: Path to the audio file to process
            output_dir: Output directory for separated stems
            suffix: Suffix for output files (e.g., 'demucs' for 'drums-demucs.wav')
            show_progress: Show progress bar during processing
        """
        # Lazy imports for performance
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.audio import save_audio

        if self.model is None:
            self.load_model()

        # Type assertions - load_model() ensures these are set
        assert self.model is not None, "Model must be loaded"
        assert self._samplerate is not None, "Sample rate must be set"

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load audio
        self.logger.info(f"Loading audio from {audio_path}")
        wav, sr = torchaudio.load(audio_path.as_posix())

        # Demucs expects stereo audio
        if wav.shape[0] == 1:
            # Convert mono to stereo by duplicating the channel
            wav = wav.repeat(2, 1)
            self.logger.debug("Converted mono to stereo")
        elif wav.shape[0] > 2:
            # Take first 2 channels if more than stereo
            wav = wav[:2]
            self.logger.warning(f"Audio has {wav.shape[0]} channels, using first 2")

        # Resample if needed
        if sr != self._samplerate:
            self.logger.info(f"Resampling from {sr}Hz to {self._samplerate}Hz")
            resampler = torchaudio.transforms.Resample(sr, self._samplerate)
            wav = resampler(wav)
            sr = self._samplerate

        # Move to device
        wav = wav.to(self.device)

        # Apply model
        self.logger.info(f"Separating audio with {self.shifts} shifts")
        with torch.inference_mode():
            sources = apply_model(
                self.model,
                wav.unsqueeze(0),  # Add batch dimension
                shifts=self.shifts,
                split=True,
                overlap=0.25,
                progress=show_progress,
            )[0]  # Remove batch dimension

        # Save stems
        self.logger.info(f"Saving separated stems to {output_dir}")
        for source_idx, source_name in enumerate(self.model.sources):
            stem_audio = sources[source_idx]

            # Build output path: {stem}-{suffix}.wav or {stem}.wav if no suffix
            if suffix:
                output_filename = f"{source_name}-{suffix}.wav"
            else:
                output_filename = f"{source_name}.wav"
            output_path = output_dir / output_filename

            # Save audio using demucs.audio.save_audio for consistency
            # Convert to CPU and numpy for saving
            stem_cpu = stem_audio.cpu()
            save_audio(
                stem_cpu,
                output_path.as_posix(),
                samplerate=self._samplerate,
                clip="rescale",
                as_float=False,
                bits_per_sample=16,
            )

            self.logger.debug(f"Saved: {output_path}")

        self.logger.info(f"Completed! Separated {len(self.model.sources)} stems")
