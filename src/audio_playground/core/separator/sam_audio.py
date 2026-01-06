"""Wrapper class for SAM-Audio separation model."""

import logging
from pathlib import Path


class SeparatorSAMAudio:
    """Wrapper for SAM-Audio model to hide implementation details."""

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        logger: logging.Logger,
        predict_spans: bool,
        reranking_candidates: int,
    ):
        """
        Initialize SAM-Audio separator.

        Args:
            model_name: Model name/path for SAM-Audio
            device: Device to use (cpu, cuda, mps, etc.)
            batch_size: Number of prompts to process in a batch
            logger: Logger instance
            predict_spans: Enable span prediction
            reranking_candidates: Number of reranking candidates
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.logger = logger
        self.predict_spans = predict_spans
        self.reranking_candidates = reranking_candidates
        self.model = None
        self.processor = None
        self._sample_rate = None

    def load_model(self) -> None:
        """Load the SAM-Audio model."""
        from sam_audio import SAMAudio, SAMAudioProcessor

        self.logger.info(f"Loading SAM-Audio model: {self.model_name}")
        self.logger.info(f"Using device: {self.device}")

        self.model = (
            SAMAudio.from_pretrained(
                self.model_name,
                map_location=self.device,
            )
            .to(self.device)
            .eval()
        )
        self.processor = SAMAudioProcessor.from_pretrained(self.model_name)
        self._sample_rate = self.processor.audio_sampling_rate

    def separate_segments(
        self,
        segment_files: list[Path],
        prompts: list[str],
        output_dir: Path,
    ) -> None:
        """
        Process audio segments with SAM-Audio model.

        Args:
            segment_files: List of segment file paths to process
            prompts: List of text prompts for separation
            output_dir: Output directory for processed files
        """
        # Lazy imports for performance
        import torch
        import torchaudio

        if self.model is None:
            self.load_model()

        # Type assertions - load_model() ensures these are set
        assert self.model is not None, "Model must be loaded"
        assert self.processor is not None, "Processor must be loaded"
        assert self._sample_rate is not None, "Sample rate must be set"

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each segment
        with torch.inference_mode():
            for idx, audio_path in enumerate(segment_files):
                self.logger.info(f"Processing {audio_path.name} ({idx + 1}/{len(segment_files)})")

                # Process prompts in batches for efficiency
                prompt_batches = self._batch_items(prompts, self.batch_size)
                self.logger.debug(
                    f"Processing {len(prompts)} prompts in {len(prompt_batches)} batch(es) "
                    f"(batch_size={self.batch_size})"
                )

                for batch_idx, prompt_batch in enumerate(prompt_batches):
                    self.logger.debug(
                        f"Processing batch {batch_idx + 1}/{len(prompt_batches)}: {prompt_batch}"
                    )

                    # Process all prompts in this batch together
                    # SAM-Audio processor requires len(audios) == len(descriptions)
                    # So we duplicate the audio path for each prompt in the batch
                    inputs = self.processor(  # type: ignore[call-non-callable]
                        audios=[audio_path.as_posix()] * len(prompt_batch),
                        descriptions=prompt_batch,
                    ).to(self.device)

                    result = self.model.separate(
                        inputs,
                        predict_spans=self.predict_spans,
                        reranking_candidates=self.reranking_candidates,
                    )

                    # Extract and save individual results for each prompt in the batch
                    for prompt_idx_in_batch, prompt in enumerate(prompt_batch):
                        safe_prompt = prompt.replace(" ", "_").replace("/", "_")

                        # Build output filename: {segment_stem}-{prompt}.wav
                        output_filename = f"{audio_path.stem}-{safe_prompt}.wav"

                        output_path = output_dir / output_filename
                        target_audio = result.target[prompt_idx_in_batch].unsqueeze(0).cpu()
                        torchaudio.save(output_path.as_posix(), target_audio, self._sample_rate)
                        self.logger.debug(f"Saved: {output_path}")

                self.logger.info(f"Completed processing {audio_path.name}")

    @staticmethod
    def _batch_items(items: list[str], batch_size: int) -> list[list[str]]:
        """
        Split a list of items into batches of specified size.

        Args:
            items: List of items to batch
            batch_size: Maximum size of each batch (must be >= 1)

        Returns:
            List of batches, where each batch is a list of items
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        batches: list[list[str]] = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i : i + batch_size])
        return batches
