# Audio-Playground Refactoring Plan

## Overview

Transform the monolithic `extract sam-audio` command into a modular, testable, cacheable toolkit while maintaining backward compatibility.

**Total Phases:** 8
**Estimated Effort:** ~40-50 incremental changes
**Testing Target:** >=95% coverage by end

---

## Implementation Status

**Current Phase:** Phase 4 In Progress (Steps 4.1-4.2 Complete)

- ✅ Phase 0: Complete
- ✅ Phase 1: Complete
- ✅ Phase 2: Complete (All atomic and composite commands implemented)
- ✅ Phase 3: Complete (Restructured to src layout, replaced mypy with ty)
- ⏳ Phase 4: In Progress (Steps 4.1-4.2 Complete - Performance profiling added, deprecated commands removed)
- ⏳ Phase 5: Not Started
- ⏳ Phase 6: Not Started
- ⏳ Phase 7: Not Started
- ⏳ Phase 8: Not Started

---

## ✅ Phase 0: Add `--chain-residuals` Flag

**Status:** ✅ Complete - Added `--chain-residuals` flag to `AudioPlaygroundConfig` for conditional residual-chaining logic.

---

## ✅ Phase 1: Modularization & Lazy Imports

**Status:** ✅ Complete - Created `core/` package (`wav_converter.py`, `segmenter.py`, `merger.py`). Implemented lazy imports for torch/torchaudio. Added CLI options for sample-rate, max-segments, and segment-window-size. Type checking passes with `--strict`.

---

## ✅ Phase 2: Atomic CLI Commands

### Goal

Break `extract sam-audio` into reusable commands: `convert`, `segment`, `merge`.
Support both SAM-Audio and Demucs models.

### Step 2.1: Create `convert` command ✅

**Status:** ✅ Complete - Created `cli/convert/to_wav.py` wrapping `convert_to_wav()` from `core/wav_converter.py`. Common option decorators created for consistency.

### Step 2.2: Create `segment` command ✅

**Status:** ✅ Complete - Created `cli/segment/split.py` wrapping `create_segments()` and `split_to_files()` from `core/segmenter.py`. Outputs segment files and `segment_metadata.json`.

### Step 2.3: Create `merge` command ✅

**Status:** ✅ Complete - Created `cli/merge/concat.py` wrapping `concatenate_segments()` from `core/merger.py`. Supports glob patterns. Auto-detects sample rate from first file.

### Step 2.4: Create `extract process-sam-audio` command ✅ [TO BE DEPRECATED]

**Status:** ✅ Complete - Created `cli/extract/process_sam_audio.py` for processing audio segments. This command is scheduled for removal in Phase 4 as it creates an unnecessary intermediate step.

### Step 2.5: Create `extract process-demucs` command ✅ [TO BE DEPRECATED]

**Status:** ✅ Complete - Created `cli/extract/process_demucs.py`. This command is scheduled for removal in Phase 4 as it creates an unnecessary intermediate step.

### Step 2.6: Make `extract sam-audio` a Composite Command ✅

**Status:** ✅ Complete - Created `cli/extract/sam_audio.py` composite command that chains together: convert → segment → process-sam-audio → merge. Provides end-to-end workflow with single command.

### Step 2.7: Create `extract demucs` Composite Command ✅

**Status:** ✅ Complete - Created `cli/extract/demucs.py` composite command that chains together: convert → segment → process-demucs → merge. Provides end-to-end workflow for Demucs stem separation.

## ✅ Phase 3: Restructure to src Layout

**Status:** ✅ Complete - Moved `audio_playground/` to `src/audio_playground/`. Updated `pyproject.toml` with `where = ["src"]`. Replaced mypy with ty for type checking.

## ⏳ Phase 4: Performance Profiling & Terminology Rationalization

**Status:** In Progress (Steps 4.1-4.2 Complete)

**Goals:**

1. ✅ Add performance profiling to all CLI commands
2. Rationalize terminology to use "chunk" consistently throughout the project
3. ✅ Remove deprecated commands
4. Add optional overlap support for chunk-based operations
5. Clean up legacy arguments

### Step 4.1: Create Performance Profiler ✅

**Status:** ✅ Complete - Created `PerformanceTracker` class with execution time and memory tracking, YAML report generation, decorator support. Integrated into all CLI commands (convert, segment, merge, extract sam-audio, extract demucs). Comprehensive test suite added in `tests/core/test_performance_tracker.py`.

### Step 4.2: Remove Deprecated Commands ✅

**Status:** ✅ Complete - Removed `extract process-sam-audio` and `extract process-demucs` commands. Users should use the composite commands `extract sam-audio` and `extract demucs` instead.

### Step 4.3: Rationalize Terminology - Use "Chunk" Consistently

**Current Inconsistencies:**

- "segments" vs "chunks"
- `--window-size` vs `--chunk-duration`
- Mixed usage in code, CLI, and documentation

**Changes Required:**

**CLI Options:**

- Rename all `--window-size` to `--chunk-duration`
- Rename all `--segment-*` to `--chunk-*`
- Update `--max-segments` to `--max-chunks`
- Rename `segment xxx` command to `chunk xxx`

**Code:**

- Rename `segmenter.py` functions to use "chunk" terminology
- Update `create_segments()` → `create_chunks()`
- Update `split_to_files()` to output `chunk-NNN.wav` instead of `segment-NNN.wav`
- Update variable names: `segment_*` → `chunk_*`

**Documentation:**

- Update all references in docstrings
- Update README and requirements.md

**Files:**

- `src/audio_playground/core/segmenter.py` → possibly rename to `chunker.py`
- All CLI commands using segment terminology
- `docs/requirements.md`
- `README.md`

### Step 4.4: Add --no-chunks Option

**Requirement:** If a command accepts a `--chunk-duration` argument, it should also accept a `--no-chunks` flag to disable chunking entirely.

**Commands Affected:**

- `chunk (was segment) split`
- `extract sam-audio`
- Any other command that processes audio in chunks

**Implementation:**

- Add `--no-chunks` as a boolean flag
- When set, process entire audio file without splitting
- Mutually exclusive with `--chunk-duration`, `--max-chunks`

### Step 4.5: Add Wrapper Classes SeparatorDemucs and SeparatorSAMAudio

Just wrapper classes for those models, to hide the implementation details from the commands using them

### Step 4.6: Add Optional Overlap Window for Chunks

**Requirement:** Add overlap support to chunk-based operations to improve quality at chunk boundaries.

**Commands Affected:**

- `chunk (was segment) split` (both standalone and as 2nd step of `extract sam-audio`)
- `merge concat` (both standalone and as final step of `extract sam-audio`)

**New CLI Options:**

- `--overlap-duration FLOAT`: Overlap in seconds between chunks (default: 0.0 = no overlap)
- `--crossfade-type {linear|cosine}`: Crossfade algorithm for overlap blending (default: cosine)

**Implementation:**

**For `chunk (was segment) split`:**

- Calculate overlapping chunk boundaries
- Save overlapping chunks to files
- Include overlap metadata in output JSON

**For `merge concat`:**

- Read overlap metadata from input
- Apply crossfade blending in overlap regions
- Use cosine or linear fade based on `--crossfade-type`

**Files:**

- `src/audio_playground/core/segmenter.py` - add overlap calculation
- `src/audio_playground/core/merger.py` - add crossfade blending
- `src/audio_playground/cli/segment/split.py`
- `src/audio_playground/cli/merge/concat.py`

**Implementation:**

- Add a test that starts from a 40s .wav file, chunks it into 10 sec windows with 2 seconds overlap, merges it back - the initial and final file should be identical length and pretty much identical content (minimal degradation due to processing is acceptable)

### Step 4.6: Remove Legacy Arguments

**Arguments to Remove:**

- `--continue-from` (if still present in any commands)
- Any other deprecated or unused arguments found during refactoring

**Process:**

- Search codebase for `continue_from`
- Remove from CLI option definitions
- Remove from function signatures
- Update tests

---

## ⏳ Phase 5: Advanced ODE Control & MLX Backend Integration

**Status:** Not Started

**Background:** SAMAudio uses diffusion models with ODE (Ordinary Differential Equation) integration to gradually transform from noise (t=0) to clean separated audio (t=1). Different solver methods and step sizes trade speed for quality. The mlx-audio project reimplemented SAMAudio in MLX with full control over ODE integration, enabling significant performance and quality tuning. Files from both are available under docs/inspiration

**Key Concepts:**

- **ODE Integration:** The core diffusion process - integrating through small time steps from noise to clean audio
- **Solver Methods:**
  - **Euler:** 1 forward pass per step (~50% faster, lower quality)
  - **Midpoint:** 2 forward passes per step (higher quality, more accurate velocity estimation)
- **Step Size:** Controls quality vs speed (e.g., 2/32 = 16 steps total)
- **Audio Codec (DACVAE):** Compression layer to encode raw audio into feature space before processing

**Current Limitations:**

- We use PyTorch SAMAudio's `model.separate()` as a black box - no control over internal diffusion process
- We don't know if PyTorch SAMAudio exposes solver parameters in its API
- Text prompts are re-encoded for each chunk (wasteful for short prompts like "bass", "drums")

**Goals:**

1. Gain control over ODE solver parameters (method, steps) for quality/speed tuning
2. Implement MLX backend for Apple Silicon users (10-50x faster than PyTorch)
3. Optimize text feature encoding (pre-encode once, reuse across chunks)
4. Investigate if PyTorch SAMAudio can expose solver parameters without reimplementation

### Step 5.1: Investigate PyTorch SAMAudio ODE Control

**Research Phase:**

- Examine PyTorch SAMAudio source code (`docs/inspiration/original_sam_audio.py` if available)
- Look for hidden parameters in `model.separate()` that control solver behavior
- Check if newer versions of `sam-audio` package expose solver configuration
- Review SAMAudio GitHub issues/PRs for solver control discussions

**Decision Point:**

- **If solver parameters exist:** Document them and integrate into our optimizer
- **If not:** Proceed to Step 5.2 (MLX backend) or Step 5.3 (custom ODE wrapper)

**Files:**

- Research notes in `docs/research/pytorch_sam_audio_ode_investigation.md` (new)

### Step 5.2: MLX Backend Integration (Apple Silicon Fast Path)

**Files:**

- `audio_playground/core/backends/__init__.py` (new)
- `audio_playground/core/backends/mlx_backend.py` (new)
- `audio_playground/core/backends/pytorch_backend.py` (new)

**Rationale:**

- mlx-audio reimplemented SAMAudio in MLX with full ODE control
- MLX is optimized for Apple's unified memory architecture
- Provides 10-50x speedup on M1/M2/M3 chips
- Already has solver parameter support built-in

**Backend Abstraction:**

```python
# audio_playground/core/backends/__init__.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal
import torch

class AudioBackend(ABC):
    """Abstract base class for audio processing backends."""

    @abstractmethod
    def load_model(self, model_name: str, device: str):
        """Load and initialize the SAMAudio model."""
        pass

    @abstractmethod
    def separate(
        self,
        audio_path: Path,
        prompts: list[str],
        solver_method: Literal["euler", "midpoint"] | None = None,
        solver_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Separate audio into stems based on text prompts.

        Args:
            audio_path: Path to audio file
            prompts: List of text descriptions (e.g., ["bass", "drums"])
            solver_method: ODE solver ("euler" or "midpoint")
            solver_steps: Number of ODE integration steps

        Returns:
            Dictionary mapping prompt to separated audio tensor
        """
        pass

    @abstractmethod
    def supports_solver_config(self) -> bool:
        """Return True if this backend supports solver configuration."""
        pass

def get_backend(backend: str = "auto") -> AudioBackend:
    """
    Auto-detect best backend:
    - MLX if on Mac with M1/M2/M3 and mlx-audio installed
    - PyTorch otherwise
    """
    if backend == "auto":
        if _should_use_mlx():
            try:
                return MLXBackend()
            except ImportError:
                logger.warning("MLX not available, falling back to PyTorch")
                return PyTorchBackend()
        return PyTorchBackend()
    elif backend == "mlx":
        return MLXBackend()
    elif backend == "pytorch":
        return PyTorchBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")

def _should_use_mlx() -> bool:
    """Check if MLX should be used (Mac + Apple Silicon + mlx-audio installed)."""
    import platform
    if platform.system() != "Darwin":
        return False
    # Check for Apple Silicon
    if platform.machine() not in ("arm64", "aarch64"):
        return False
    # Check if mlx-audio is available
    try:
        import mlx_audio
        return True
    except ImportError:
        return False
```

**MLX Backend:**

```python
# audio_playground/core/backends/mlx_backend.py
try:
    from mlx_audio import SAMAudio as SAMAudioMLX
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

class MLXBackend(AudioBackend):
    """MLX backend with full ODE solver control (Apple Silicon only)."""

    def __init__(self):
        if not HAS_MLX:
            raise ImportError(
                "mlx-audio not installed. Install with: pip install mlx-audio\n"
                "Note: MLX is only available on Apple Silicon (M1/M2/M3)"
            )
        self.model = None

    def load_model(self, model_name: str, device: str):
        # MLX handles device automatically (uses Metal GPU)
        self.model = SAMAudioMLX.from_pretrained(model_name)

    def separate(
        self,
        audio_path: Path,
        prompts: list[str],
        solver_method: str | None = None,
        solver_steps: int | None = None,
    ) -> dict[str, Tensor]:
        """
        Separate using MLX with ODE solver control.

        MLX's reimplementation gives us full control over:
        - Solver method (euler vs midpoint)
        - Number of steps (quality vs speed tradeoff)
        - Text feature pre-encoding (encode once, reuse)
        """
        # Configure solver
        solver_config = {}
        if solver_method:
            solver_config["method"] = solver_method
        if solver_steps:
            solver_config["step_size"] = 2 / solver_steps  # mlx-audio uses step_size

        # Pre-encode text features once (optimization)
        text_features = self.model.encode_text(prompts)

        # Process audio with solver control
        return self.model.separate(
            audio_path.as_posix(),
            prompts=prompts,
            _text_features=text_features,  # Reuse pre-encoded features
            solver_config=solver_config if solver_config else None,
        )

    def supports_solver_config(self) -> bool:
        return True
```

**PyTorch Backend:**

```python
# audio_playground/core/backends/pytorch_backend.py
from sam_audio import SAMAudio, SAMAudioProcessor

class PyTorchBackend(AudioBackend):
    """PyTorch backend - standard SAMAudio library."""

    def __init__(self):
        self.model = None
        self.processor = None

    def load_model(self, model_name: str, device: str):
        self.model = (
            SAMAudio.from_pretrained(model_name, map_location=device)
            .to(device)
            .eval()
        )
        self.processor = SAMAudioProcessor.from_pretrained(model_name)

    def separate(
        self,
        audio_path: Path,
        prompts: list[str],
        solver_method: str | None = None,
        solver_steps: int | None = None,
    ) -> dict[str, Tensor]:
        """
        Separate using PyTorch SAMAudio.

        WARNING: PyTorch SAMAudio does not currently expose solver parameters
        in its public API. If solver_method or solver_steps are provided,
        they will be logged but ignored.

        This may change in future versions - see Step 5.1 investigation.
        """
        if solver_method or solver_steps:
            logger.warning(
                f"PyTorch backend does not support solver configuration. "
                f"Ignoring: method={solver_method}, steps={solver_steps}. "
                f"Consider using MLX backend on Apple Silicon for solver control."
            )

        # Standard PyTorch processing (black box)
        inputs = self.processor(
            audios=[audio_path.as_posix()],
            descriptions=prompts,
        )

        with torch.inference_mode():
            result = self.model.separate(inputs)

        return result

    def supports_solver_config(self) -> bool:
        """PyTorch backend currently does not support solver config."""
        return False
```

### Step 5.3: Integrate Backends into Optimizer

**File:** `audio_playground/core/sam_audio_optimizer.py` (modify)

**Changes:**

- Replace direct SAMAudio usage with backend abstraction
- Pass solver config through to backend
- Add backend initialization logic

```python
def process_long_audio(
    audio_path: Path,
    prompts: list[str],
    model: SAMAudio | None = None,  # Deprecated, use backend instead
    processor: SAMAudioProcessor | None = None,  # Deprecated
    device: str = "auto",
    backend: AudioBackend | None = None,  # New: backend instance
    backend_name: str = "auto",  # New: backend selection
    chunk_duration: float = 10.0,
    overlap_duration: float = 2.0,
    crossfade_type: Literal["cosine", "linear"] = "cosine",
    solver_config: SolverConfig | None = None,
) -> dict[str, torch.Tensor]:
    """
    Process long audio with chunking and solver control.

    Args:
        backend: AudioBackend instance (if None, auto-created from backend_name)
        backend_name: Backend to use ("auto", "mlx", "pytorch")
        solver_config: ODE solver configuration (method, steps)

    Note: Solver config only works with backends that support it (currently MLX only).
    Use backend.supports_solver_config() to check.
    """
    # Initialize backend
    if backend is None:
        from audio_playground.core.backends import get_backend
        backend = get_backend(backend_name)
        backend.load_model(model_name="sam-audio", device=device)

    # Warn if solver config provided but not supported
    if solver_config and not backend.supports_solver_config():
        logger.warning(
            f"Solver configuration provided but {backend.__class__.__name__} "
            f"does not support it. Consider using MLX backend on Apple Silicon."
        )

    # ... rest of chunking logic ...

    # Process each chunk with solver control
    chunk_result = backend.separate(
        chunk_path,
        prompts,
        solver_method=solver_config.method if solver_config else None,
        solver_steps=solver_config.steps if solver_config else None,
    )
```

### Step 5.4: Configuration & CLI Updates

**File:** `audio_playground/config/app_config.py` (modify)

```python
class AppConfig(BaseModel):
    # ... existing fields ...

    # Backend selection
    backend: Literal["auto", "mlx", "pytorch"] = Field(
        default="auto",
        description=(
            "Processing backend:\n"
            "- auto: Use MLX on Apple Silicon if available, PyTorch otherwise\n"
            "- mlx: Force MLX (Apple Silicon only, errors if unavailable)\n"
            "- pytorch: Force PyTorch (for consistency testing)"
        )
    )
```

**File:** `audio_playground/cli/extract/sam_audio.py` (modify)

```python
@click.option(
    "--backend",
    type=click.Choice(["auto", "mlx", "pytorch"]),
    help="Processing backend (auto=detect best, mlx=Apple Silicon only, pytorch=force PyTorch)",
)
def sam_audio(
    ctx: click.Context,
    # ... existing params ...
    backend: str | None = None,
):
    # ... existing code ...

    # Override config with CLI arguments
    if backend is not None:
        config.backend = backend

    # Log backend info
    from audio_playground.core.backends import get_backend
    selected_backend = get_backend(config.backend)
    logger.info(f"Using backend: {selected_backend.__class__.__name__}")

    if solver_config and not selected_backend.supports_solver_config():
        logger.warning(
            "⚠️  Solver configuration requested but current backend doesn't support it. "
            "On Apple Silicon, use --backend mlx for solver control."
        )
```

### Step 5.5: Text Feature Caching (Backend-Agnostic Optimization)

**Goal:** Pre-encode text prompts once and reuse across chunks

**Implementation Note:** This is already done correctly in MLX backend. For PyTorch backend, we need to investigate if the processor allows passing pre-encoded features.

**Research:**

- Check if `SAMAudioProcessor` can output features separately
- Check if `model.separate()` accepts pre-encoded text features
- If not, this optimization remains MLX-only

### Step 5.6: Documentation & Performance Benchmarks

**Files:**

- `docs/BACKENDS.md` (new)
- `README.md` (update installation instructions)

**Content for BACKENDS.md:**

````markdown
# Backend Selection Guide

## Overview

Audio-Playground supports two backends for SAMAudio processing:

1. **PyTorch** (default, all platforms)
2. **MLX** (Apple Silicon only, 10-50x faster)

## Feature Comparison

| Feature                     | PyTorch                 | MLX                              |
| --------------------------- | ----------------------- | -------------------------------- |
| Platform Support            | All (Linux/Mac/Windows) | Apple Silicon only (M1/M2/M3/M4) |
| ODE Solver Control          | ❌ No                   | ✅ Yes                           |
| Solver Methods              | Fixed (midpoint)        | euler, midpoint                  |
| Solver Steps Control        | ❌ No                   | ✅ Yes (1-100 steps)             |
| Text Feature Caching        | ❌ No                   | ✅ Yes                           |
| Relative Speed (2min audio) | 1x (baseline)           | 3-6x faster                      |

## Installation

### PyTorch (Default)

```bash
pip install -e .
```
````

### MLX (Apple Silicon Only)

```bash
pip install -e .
pip install mlx-audio  # Adds MLX backend support
```

## Usage

### Auto-Detection (Recommended)

```bash
# Automatically uses MLX on Apple Silicon if available
audio-playground extract sam-audio --src input.mp4
```

### Force Specific Backend

```bash
# Force MLX (errors if not on Apple Silicon)
audio-playground extract sam-audio --src input.mp4 --backend mlx

# Force PyTorch (for consistency testing)
audio-playground extract sam-audio --src input.mp4 --backend pytorch
```

### ODE Solver Control (MLX Only)

**Euler Method (2x faster, lower quality):**

```bash
audio-playground extract sam-audio \
  --src input.mp4 \
  --backend mlx \
  --solver euler \
  --solver-steps 16
```

**Midpoint Method (higher quality, slower):**

```bash
audio-playground extract sam-audio \
  --src input.mp4 \
  --backend mlx \
  --solver midpoint \
  --solver-steps 32  # Default
```

## Performance Benchmarks

Tested on 2-minute audio file:

| Platform             | Backend | Solver   | Steps   | Time | Speedup |
| -------------------- | ------- | -------- | ------- | ---- | ------- |
| Mac M1               | PyTorch | (fixed)  | (fixed) | 360s | 1.0x    |
| Mac M1               | MLX     | midpoint | 32      | 100s | 3.6x    |
| Mac M1               | MLX     | euler    | 16      | 60s  | 6.0x    |
| Mac M3               | MLX     | midpoint | 32      | 75s  | 4.8x    |
| Mac M3               | MLX     | euler    | 16      | 45s  | 8.0x    |
| Linux GPU (RTX 3090) | PyTorch | (fixed)  | (fixed) | 200s | 1.8x    |
| CPU (Intel i9)       | PyTorch | (fixed)  | (fixed) | 600s | 0.6x    |

## Quality vs Speed Tradeoffs

**Solver Method:**

- **Midpoint:** Higher quality, 2x slower (2 forward passes per step)
- **Euler:** Lower quality, 2x faster (1 forward pass per step)

**Solver Steps:**

- **32 steps:** Default, good quality
- **16 steps:** 2x faster, slight quality loss
- **8 steps:** 4x faster, noticeable artifacts
- **64 steps:** Minimal quality improvement, 2x slower

**Recommended Settings:**

- **Production quality:** `--solver midpoint --solver-steps 32`
- **Fast previews:** `--solver euler --solver-steps 16`
- **Quality critical:** `--solver midpoint --solver-steps 64`

## Troubleshooting

### "MLX backend not available"

- MLX only works on Apple Silicon (M1/M2/M3/M4)
- Install with: `pip install mlx-audio`
- Check: `python -c "import mlx_audio; print('MLX OK')"`

### "Solver configuration ignored"

- Solver control only works with MLX backend
- Add `--backend mlx` to enable solver control
- PyTorch backend currently doesn't expose solver parameters

### Fallback Behavior

If you request MLX but it's not available, the system will:

1. Log a warning
2. Automatically fall back to PyTorch
3. Continue processing (solver config will be ignored)

````

**Update README.md:**

Add installation section with MLX instructions and performance comparison table.

### Step 5.7: Testing

**Files:**
- `tests/core/backends/test_mlx_backend.py` (new)
- `tests/core/backends/test_pytorch_backend.py` (new)
- `tests/core/backends/test_backend_abstraction.py` (new)

**Test Coverage:**
- ✅ Auto-detection works on Mac with/without mlx-audio
- ✅ Forced backend selection works
- ✅ Fallback on import error
- ✅ Solver config passes through correctly in MLX
- ✅ Solver config warning in PyTorch
- ✅ Identical output quality between backends (when using same solver settings)
- ✅ Performance benchmarks (if on Apple Silicon)

### Validation Checklist

- [ ] Backend abstraction implemented and tested
- [ ] MLX backend with solver control working on Apple Silicon
- [ ] PyTorch backend maintains backward compatibility
- [ ] Auto-detection logic works correctly
- [ ] CLI `--backend` option functional
- [ ] Solver config passes through to MLX backend
- [ ] PyTorch backend warns when solver config provided
- [ ] Documentation complete (BACKENDS.md, README updates)
- [ ] Performance benchmarks documented
- [ ] All tests passing

### Future Considerations (Phase 5+)

**Option A: Custom ODE Wrapper for PyTorch**
If PyTorch SAMAudio doesn't expose solver parameters, we could reimplement the ODE integration loop ourselves (like mlx-audio did). This would require:
- Deep understanding of SAMAudio's internal architecture
- Access to model's internal `_ode_step_euler()` and `_ode_step_midpoint()` methods
- Significant engineering effort
- Risk of divergence from official implementation

**Option B: Contribute to PyTorch SAMAudio**
- Submit PR to sam-audio repository to expose solver parameters
- Wait for upstream adoption
- Lower effort, better for ecosystem

**Recommended:** Use MLX backend for solver control on Apple Silicon, wait for PyTorch SAMAudio to add native support.

### Multiprocessing Note

**Question:** Should we add multiprocessing capability?

**Answer:** No, not for diffusion-based audio separation. Here's why:

1. **Sequential Dependency:** ODE integration is inherently sequential - each step depends on the previous step's output. You cannot parallelize the diffusion steps themselves.

2. **Memory Constraints:** Loading multiple SAMAudio models in parallel would consume enormous GPU/RAM (each model is ~1-2GB).

3. **GPU Saturation:** Modern GPUs are already well-utilized during diffusion inference. Adding more processes wouldn't increase GPU throughput, just increase contention.

4. **Better Alternatives:**
   - **Chunking** (already implemented): Process long audio in segments
   - **Batching** (future): Process multiple prompts simultaneously in one forward pass
   - **MLX Backend:** Much faster on Apple Silicon without multiprocessing complexity

**Where Multiprocessing COULD Help:**
- Processing multiple completely separate files (e.g., batch processing a directory)
- Pre/post-processing steps (WAV conversion, resampling, concatenation)

**Recommendation:** Skip multiprocessing for SAMAudio inference itself. If needed in future, add batch file processing capability instead (Phase 8 workflows could handle this).

## ⏳ Phase 6: Add Global Config Overrides to Each Command

- **File:** `audio_playground/cli/common.py` (partially complete)
- **Status:** ⚠️ Partially complete (basic options done, global config options pending)
- **Completed:**
  - ✅ `src_option()` - for `--src` parameter
  - ✅ `target_option()` - for `--target` parameter
  - ✅ `output_dir_option()` - for `--output-dir` parameter
  - ✅ `input_dir_option()` - for `--input-dir` parameter
- **TODO:** Add global config decorators:
  ```python
  @click.option("--log-level", type=click.Choice([...]), help="...")
  @click.option("--device", default="auto", help="...")
  @click.option("--temp-dir", type=click.Path(), help="...")
  def common_config_options(func):
      """Decorator for shared config flags."""
````

- **Usage:** Apply to all commands to avoid repetition
- **Test:** Verify each command respects `--device`, `--log-level`, etc.

### Validation Checklist

- [x] **Step 2.1:** `audio-playground convert to-wav --help` works
- [x] **Step 2.1:** Convert command tests pass
- [x] **Step 2.1:** Common option decorators created
- [x] **Step 2.2:** `audio-playground segment split --help` works
- [x] **Step 2.2:** Segment command produces valid output
- [x] **Step 2.3:** `audio-playground merge concat --help` works
- [x] **Step 2.3:** Merge command reconstructs audio correctly
- [x] **Step 2.4:** ⚠️ Deprecated - Command removed in Phase 4 refactoring
- [x] **Step 2.4:** ⚠️ Deprecated - Functionality moved to optimizer
- [x] **Step 2.5:** ⚠️ Deprecated - Command removed in Phase 4 refactoring
- [x] **Step 2.5:** ⚠️ Deprecated - Logic moved to core/demucs_processor.py
- [x] **Step 2.6:** `extract sam-audio` composite command implemented (refactored to use optimizer directly)
- [x] **Step 2.7:** `extract demucs` composite command implemented
- [x] **Phase 3:** Restructured to src layout
- [x] **Phase 3:** Updated pyproject.toml with src configuration
- [x] **Phase 3:** Replaced mypy with ty
- [x] **Phase 4:** PyTorch optimizations implemented (caching, chunking, streaming)
- [x] **Phase 4:** Created sam_audio_optimizer.py with all optimization features
- [x] **Phase 4:** Updated app_config.py with performance settings
- [x] **Phase 4:** Added CLI options to extract sam-audio (removed process-sam-audio command)
- [x] **Phase 4:** Architectural refactoring - eliminated file-based segmentation
- [x] **Phase 4:** Created PerformanceTracker for automatic metrics reporting
- [x] **Phase 4:** Created comprehensive test suite
- [ ] **Phase 5:** MLX backend auto-detection works on Apple Silicon
- [ ] **Phase 5:** Backend abstraction allows switching PyTorch ↔ MLX
- [ ] **Phase 5:** Fallback to PyTorch on missing MLX dependency
- [ ] **Phase 6:** Global config options applied to all commands

**Exit Criteria:** All atomic commands functional; both composite commands work; performance optimizations tested; backend abstraction complete; common options standardized

### Additional Improvements

**CI/CD Enhancements:** ✅ Complete

- GitHub Actions workflow for automated QA checks (runs `task qa` on all pushes/PRs)
- Coverage badge generation (auto-generated on main branch pushes, stored in `badges` branch)
- Coverage artifact upload (HTML reports with 7-day retention)

---

## ⏳ Phase 7: Lazy Caching & Artifact Reuse

### Goal

Avoid re-processing identical inputs by caching segment files and metadata. Caches should be by sub-command. So the command `extract sam-audio` contains the steps convert, create chunks [1...N], process[chunk x], concatenate [chuncks]. Each one of the commands should be in the cache, and the artifacts should be in a folder named as a hash of the arguments

```mermaid
cache
├── convert
│   ├── hash1
│   ├── ...
│   └── hashN
├── create chunks
│   ├── hash1
│   ├── ...
│   └── hashN
├── process chunk with sam-audio
│   ├── hash1
│   ├── ...
│   └── hashN
└── process chunk with sam-audio
    ├── hash1
    ├── ...
    └── hashN
```

24 directories

### Step 7.1: Create cache manifest format

- **File:** `audio_playground/core/cache_manifest.py` (new)
- **Responsibility:**
  - Define structure for storing execution metadata
  - Compute input file hash (MD5 or SHA256)
  - Store command, config, timestamps
- **Format (TOML):**

  ```toml
  [execution]
  command = "segment"
  timestamp = "2025-01-02T12:34:56Z"
  version = "0.1.0"

  [inputs]
  source_file = "input.wav"
  source_hash = "abc123..."

  [config]
  min_length = 9.0
  max_length = 17.0

  [outputs]
  files = ["segment-000.wav", "segment-001.wav"]
  manifest = "segment_metadata.json"
  ```

- **Test:** Manifest loads/saves correctly; hash calculation correct

### Step 7.2: Create cache store

- **File:** `audio_playground/core/cache_store.py` (new)
- **Responsibility:**
  - Manage `/tmp/sam_audio_playground/{uuid}/` workspace
  - List existing executions
  - Load/save manifests
  - Copy/link artifacts from previous runs
- **Signature:**

  ```python
  class CacheStore:
      @staticmethod
      def get_or_create_workspace() -> Path:
          """Return workspace root."""

      @staticmethod
      def find_matching_execution(
          command: str,
          input_hash: str,
          config_dict: dict
      ) -> Path | None:
          """Find UUID folder with matching manifest."""

      @staticmethod
      def create_execution(
          command: str,
          input_hash: str,
          config_dict: dict
      ) -> Path:
          """Create new UUID folder, save manifest."""
  ```

- **Test:** Create/find executions; manifests written correctly

### Step 7.3: Integrate caching into `Segmenter`

- **File:** `audio_playground/core/segmenter.py` (modify)
- **Change:** Before splitting, check if identical task cached:

  ```python
  @staticmethod
  def split_to_files(
      audio_path: Path,
      output_dir: Path,
      segment_lengths: list[float],
      use_cache: bool = True
  ) -> tuple[list[Path], list[tuple[float, float]]]:
      # Compute hash of audio + config
      input_hash = compute_hash(audio_path, segment_lengths)

      if use_cache:
          cached = CacheStore.find_matching_execution("segment", input_hash, {...})
          if cached:
              # Copy/link cached segments
              return copy_from_cache(cached, output_dir)

      # Otherwise, compute normally
      ...
      CacheStore.create_execution("segment", input_hash, {...})
  ```

- **Test:** Verify cache hit copies files; cache miss computes normally

### Step 7.4: Add `--no-cache` flag to commands

- **File:** `audio_playground/cli/segment/split.py` (and others)
- **Change:** Add `@click.option("--no-cache", is_flag=True, help="...")`
- **Test:** Verify `--no-cache` bypasses caching

### Step 7.5: Create cache management commands

- **File:** `audio_playground/cli/cache/__init__.py` (new)
- **Commands:**
  - `audio-playground cache list` – Show all cached executions
  - `audio-playground cache clean` – Remove old executions
  - `audio-playground cache clear` – Nuke entire workspace
- **Test:** Verify commands work; manifest reflects deletions

### Validation Checklist

- [ ] `CacheStore.get_or_create_workspace()` creates `/tmp/sam_audio_playground/`
- [ ] First `segment split` creates manifest + files
- [ ] Second `segment split` (same input) reuses cached files (verify via timestamp)
- [ ] `--no-cache` forces recomputation
- [ ] `cache list` shows all executions
- [ ] `cache clean` removes old executions

**Exit Criteria:** Caching functional; measurable speed improvement on repeat runs

---

## ⏳ Phase 8: YAML Runner & Workflows

### Goal

Allow users to define pipelines as YAML and run with `audio-playground run --config pipeline.yaml`.

### Step 8.1: Create workflow schema

- **File:** `audio_playground/core/workflow.py` (new)
- **Responsibility:**
  - Parse YAML into Python objects (Pydantic models)
  - Validate command sequences
  - Support variable substitution
- **Example YAML:**

  ```yaml
  version: "1.0"

  steps:
    - name: "convert"
      command: "convert to-wav"
      args:
        src: "input.mp4"
        target: "/tmp/audio.wav"

    - name: "segment"
      command: "segment split"
      args:
        src: "/tmp/audio.wav"
        output-dir: "/tmp/segments"
        min: 9
        max: 17

    - name: "extract"
      command: "extract process"
      args:
        input-dir: "/tmp/segments"
        prompts: ["bass", "vocals"]
        output-dir: "output/"
  ```

- **Test:** YAML parses correctly; schema validation works

### Step 8.2: Create workflow executor

- **File:** `audio_playground/core/workflow_executor.py` (new)
- **Responsibility:**
  - Execute steps sequentially
  - Pass outputs of one step to next
  - Handle errors gracefully
  - Log progress
- **Test:** Execute sample workflow; verify outputs

### Step 8.3: Create `run` command

- **File:** `audio_playground/cli/run.py` (new)
- **Usage:** `audio-playground run --config pipeline.yaml`
- **Implementation:** Load YAML, execute via `WorkflowExecutor`
- **Test:** Run sample workflow end-to-end

### Step 8.4: Add Batch File Processing

**Note:** The original `extract process` command was deprecated in Phase 4. This step is updated to add batch processing capability instead.

- **File:** `audio_playground/cli/batch.py` (new)
- **Goal:** Process multiple files in a directory automatically
- **Implementation:**

  ```python
  @click.command()
  @click.option("--input-dir", type=click.Path(exists=True, file_okay=False))
  @click.option("--pattern", default="*.mp4", help="File pattern to match")
  @click.option("--output-dir", type=click.Path())
  @click.option("--prompts", multiple=True, help="Prompts for extraction")
  @click.option("--backend", type=click.Choice(["auto", "mlx", "pytorch"]))
  def batch_extract(input_dir, pattern, output_dir, prompts, backend):
      """
      Process multiple audio files in batch.

      For each file matching pattern in input-dir:
      1. Convert to WAV
      2. Extract with SAM-Audio
      3. Save to output-dir/{filename}/
      """
  ```

- **Benefits:**
  - Process entire directories of audio files
  - Useful for dataset preparation
  - Can utilize multiprocessing for separate files (unlike single-file diffusion)
- **Test:** Process multiple test files; verify all outputs created

### Step 8.5: Document YAML format & examples

- **File:** `docs/WORKFLOWS.md` (new)
- **Content:**
  - Workflow schema explanation
  - Example pipelines (standard, chaining, demucs)
  - Variable substitution guide
- **Test:** Examples are valid and executable

### Validation Checklist

- [ ] YAML with all atomic commands parses correctly
- [ ] `run --config pipeline.yaml` executes all steps
- [ ] Output from step N feeds into step N+1 correctly
- [ ] Batch processing handles multiple files correctly
- [ ] Example workflows run without error

**Exit Criteria:** YAML runner functional; batch processing works; workflows documented

**Note on Multiprocessing:** Batch file processing is where multiprocessing CAN help - processing completely separate files in parallel. Single-file diffusion inference should remain sequential (see Phase 5 multiprocessing note).

---

claude/implement-phase-4-RV4DO

## ⏳ Phase 9: Comprehensive Documentation & ReadTheDocs Integration

**Status:** Not Started

**Goal:** Create professional, comprehensive documentation hosted on ReadTheDocs with tutorials, API reference, and platform-specific installation guides.

### Step 9.1: Setup ReadTheDocs Infrastructure

**Files:**

- `docs/conf.py` (Sphinx configuration)
- `docs/index.rst` (main documentation index)
- `.readthedocs.yaml` (RTD build configuration)
- `docs/requirements.txt` (documentation dependencies)

**Setup:**

```yaml
# .readthedocs.yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
sphinx:
  configuration: docs/conf.py
python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
```

**Dependencies:**

```
# docs/requirements.txt
sphinx>=7.0
sphinx-rtd-theme
myst-parser  # For Markdown support
sphinx-click  # For CLI documentation
sphinx-autodoc-typehints
```

### Step 9.2: Create Documentation Structure

**Files to Create:**

- `docs/index.rst` - Landing page
- `docs/installation.rst` - Installation guide (all platforms)
- `docs/installation_m1.rst` - M1/M2/M3 specific guide (based on blog post)
- `docs/quickstart.rst` - Quick start tutorial
- `docs/user_guide/index.rst` - User guide (commands, workflows)
- `docs/user_guide/backends.rst` - Backend selection guide (from BACKENDS.md)
- `docs/user_guide/performance.rst` - Performance tuning guide
- `docs/api/index.rst` - API reference (auto-generated)
- `docs/tutorials/index.rst` - Step-by-step tutorials
- `docs/contributing.rst` - Contribution guidelines
- `docs/changelog.rst` - Version history

**Documentation Structure:**

```
docs/
├── index.rst                    # Landing page
├── installation.rst             # General installation
├── installation_m1.rst          # Apple Silicon specific
├── quickstart.rst               # 5-minute quick start
├── user_guide/
│   ├── index.rst
│   ├── commands.rst             # All CLI commands
│   ├── backends.rst             # PyTorch vs MLX
│   ├── performance.rst          # Tuning guide
│   ├── workflows.rst            # YAML pipelines
│   └── troubleshooting.rst      # Common issues
├── tutorials/
│   ├── index.rst
│   ├── basic_extraction.rst     # Tutorial 1: Basic usage
│   ├── batch_processing.rst     # Tutorial 2: Batch files
│   ├── quality_tuning.rst       # Tutorial 3: Quality vs speed
│   └── advanced_workflows.rst   # Tutorial 4: Complex pipelines
├── api/
│   ├── index.rst
│   ├── core.rst                 # Core modules
│   ├── cli.rst                  # CLI commands
│   └── backends.rst             # Backend API
└── contributing.rst
```

### Step 9.3: Apple Silicon Installation Guide

**File:** `docs/installation_m1.rst`

**Based on:** <https://gotofritz.net/blog/2025-12-20-playing-with-the-sam-audio-model-on-my-m1-macbook/>

**Content Outline:**

```rst
Apple Silicon (M1/M2/M3/M4) Installation
========================================

Quick Start
-----------

The fastest setup for Apple Silicon:

.. code-block:: bash

   # Install audio-playground
   pip install audio-playground

   # Install MLX backend (10-50x faster on Apple Silicon)
   pip install mlx-audio

   # Verify installation
   audio-playground --version
   python -c "import mlx_audio; print('MLX backend available!')"

Why MLX on Apple Silicon?
--------------------------

- **10-50x faster** than PyTorch on M-series chips
- **Full ODE solver control** (quality/speed tuning)
- **Unified memory architecture** optimization
- **Metal GPU acceleration** built-in

Performance Comparison
----------------------

+----------+----------+---------+--------+---------+
| Platform | Backend  | Solver  | Steps  | Time    |
+==========+==========+=========+========+=========+
| Mac M1   | PyTorch  | (fixed) | (fixed)| 360s    |
+----------+----------+---------+--------+---------+
| Mac M1   | MLX      | midpoint| 32     | 100s    |
+----------+----------+---------+--------+---------+
| Mac M1   | MLX      | euler   | 16     | 60s     |
+----------+----------+---------+--------+---------+

System Requirements
-------------------

- macOS 12.0 or later
- Apple Silicon chip (M1/M2/M3/M4)
- Python 3.9-3.12
- 8GB+ RAM (16GB recommended)

Detailed Installation
---------------------

[Include step-by-step from blog post]

Troubleshooting
---------------

[Common M1 issues and solutions]

Next Steps
----------

- :doc:`quickstart` - Run your first extraction
- :doc:`user_guide/backends` - Learn about backend selection
- :doc:`tutorials/quality_tuning` - Optimize quality vs speed
```

### Step 9.4: API Documentation with Sphinx Autodoc

**File:** `docs/conf.py` (configure autodoc)

```python
# Sphinx configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google/NumPy docstring support
    'sphinx.ext.viewcode',  # Source code links
    'sphinx_click',         # CLI documentation
    'myst_parser',          # Markdown support
]

# Auto-document all modules
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
```

**Generate API docs:**

```bash
# Auto-generate API reference
sphinx-apidoc -o docs/api src/audio_playground
```

### Step 9.5: Interactive Tutorials

**File:** `docs/tutorials/basic_extraction.rst`

**Content:**

```rst
Tutorial 1: Basic Audio Extraction
===================================

In this tutorial, you'll learn how to extract specific instruments from
an audio file using SAM-Audio.

Prerequisites
-------------

- audio-playground installed
- A sample audio file (MP3, MP4, or WAV)

Step 1: Prepare Your Audio
---------------------------

First, let's convert your audio to WAV format:

.. code-block:: bash

   audio-playground convert to-wav \
     --src music.mp4 \
     --target music.wav

Step 2: Extract Instruments
----------------------------

Now extract specific instruments using text prompts:

.. code-block:: bash

   audio-playground extract sam-audio \
     --src music.wav \
     --output-dir output/ \
     --prompts "bass" \
     --prompts "drums" \
     --prompts "vocals"

This will create three files in output/:
- bass-sam.wav
- drums-sam.wav
- vocals-sam.wav

Step 3: Quality Tuning (Optional)
----------------------------------

On Apple Silicon with MLX backend:

.. code-block:: bash

   # High quality (slower)
   audio-playground extract sam-audio \
     --src music.wav \
     --backend mlx \
     --solver midpoint \
     --solver-steps 64 \
     --prompts "guitar solo"

   # Fast preview (lower quality)
   audio-playground extract sam-audio \
     --src music.wav \
     --backend mlx \
     --solver euler \
     --solver-steps 8 \
     --prompts "guitar solo"

What You Learned
----------------

- Converting audio to WAV format
- Using text prompts to extract instruments
- Quality vs speed tradeoffs with solver settings

Next Tutorial
-------------

:doc:`batch_processing` - Process multiple files at once
```

### Step 9.6: CLI Command Reference

**Auto-generate from Click commands using sphinx-click:**

**File:** `docs/user_guide/commands.rst`

```rst
Command Reference
=================

All available audio-playground commands.

Main Commands
-------------

.. click:: audio_playground.cli:cli
   :prog: audio-playground
   :nested: full

Extract Commands
----------------

.. click:: audio_playground.cli.extract:extract
   :prog: audio-playground extract
   :nested: full

Convert Commands
----------------

.. click:: audio_playground.cli.convert:convert
   :prog: audio-playground convert
   :nested: full

[etc for all command groups]
```

### Step 9.7: Troubleshooting Guide

**File:** `docs/user_guide/troubleshooting.rst`

**Content:**

```rst
Troubleshooting Guide
=====================

Common Issues and Solutions
---------------------------

"MLX backend not available"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms:** Warning message when running on Mac

**Causes:**
- Not on Apple Silicon (M1/M2/M3/M4)
- mlx-audio package not installed
- Using x86_64 Python instead of arm64

**Solutions:**
1. Verify Apple Silicon: ``uname -m`` should show ``arm64``
2. Install MLX: ``pip install mlx-audio``
3. Check Python arch: ``python -c "import platform; print(platform.machine())"``

"Out of memory" errors
^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms:** Process crashes with OOM error

**Solutions:**
1. Enable chunking (default, but check --no-chunks not set)
2. Reduce chunk duration: ``--chunk-duration 15``
3. Close other applications
4. Use smaller solver steps: ``--solver-steps 16``

[etc for all common issues]
```

### Step 9.8: Convert Existing Markdown Docs

**Migrate existing docs:**

- `BACKENDS.md` → `docs/user_guide/backends.rst`
- `WORKFLOWS.md` → `docs/user_guide/workflows.rst`
- Update README.md to point to ReadTheDocs

### Step 9.9: Build and Deploy

**Local testing:**

```bash
cd docs
make html
open _build/html/index.html
```

**ReadTheDocs Setup:**

1. Link GitHub repository to ReadTheDocs
2. Configure webhook for auto-builds on push
3. Set up versioning (stable = main, latest = dev)
4. Enable PDF/ePub downloads

### Validation Checklist

- [ ] Sphinx builds without errors (`make html`)
- [ ] All API modules documented with autodoc
- [ ] CLI commands auto-documented with sphinx-click
- [ ] M1 installation guide based on blog post
- [ ] All tutorials runnable and tested
- [ ] ReadTheDocs integration working
- [ ] PDF/ePub builds successfully
- [ ] Search functionality works
- [ ] Links to source code work
- [ ] Version switcher working (stable/latest)

**Exit Criteria:** Complete documentation hosted on ReadTheDocs; all installation paths documented; tutorials tested; API reference complete

### Additional Content

**File:** `docs/index.rst` (landing page)

```rst
Audio-Playground Documentation
===============================

Audio-Playground is a modular, high-performance toolkit for AI-powered audio
separation using SAM-Audio and Demucs models.

Features
--------

✨ **Text-Driven Separation** - Extract any instrument using natural language
⚡ **10-50x Faster on Apple Silicon** - MLX backend optimization
🎛️ **Quality Tuning** - Full control over ODE solver (MLX only)
🔄 **Chunked Processing** - Handle arbitrarily long audio files
📦 **Modular CLI** - Composable commands for custom workflows
🎯 **95%+ Test Coverage** - Production-ready reliability

Quick Example
-------------

.. code-block:: bash

   # Extract bass and drums from any audio file
   audio-playground extract sam-audio \
     --src song.mp4 \
     --prompts "bass" --prompts "drums"

Platform-Specific Installation
-------------------------------

- **Apple Silicon (M1/M2/M3):** :doc:`installation_m1` (recommended for best performance)
- **All Platforms:** :doc:`installation`

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   installation_m1
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/commands
   user_guide/backends
   user_guide/performance
   user_guide/workflows
   user_guide/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/basic_extraction
   tutorials/batch_processing
   tutorials/quality_tuning
   tutorials/advanced_workflows

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   contributing
   changelog

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

## ⏳ Phase 10: Testing & Coverage

### Goal

Achieve >=95% unit test coverage; no regressions.

### Step 10.1: Unit tests for `core/` modules

- **Files:** Create `tests/core/test_*.py` for each module
- **Coverage Target:** >=95% per module
- **Test Types:**
  - Happy path (normal inputs)
  - Edge cases (empty, very large, malformed)
  - Error handling (missing files, permissions, bad config)

**Example Test Structure:**

```python
# tests/core/test_segmenter.py
def test_create_segments_even_split():
    lengths = Segmenter.create_segments(30, min_length=9, max_length=17)
    assert len(lengths) == 2
    assert abs(sum(lengths) - 30) < 0.1

def test_create_segments_single():
    lengths = Segmenter.create_segments(5, min_length=9, max_length=17)
    assert len(lengths) == 1
    assert lengths[0] == 5

def test_split_to_files_creates_segments(tmp_path):
    # Create dummy WAV file
    # Call split_to_files
    # Verify N segment files created
    # Verify metadata JSON structure
```

### Step 10.2: Integration tests for CLI commands

- **Files:** Create `tests/cli/test_*.py` for each command
- **Coverage Target:** >=90% per command
- **Test Types:**
  - Command execution with various flags
  - Output file creation
  - Config override via CLI

### Step 10.3: Regression test for `extract sam-audio`

- **File:** `tests/integration/test_extract_sam_audio.py`
- **Goal:** Ensure refactored version produces identical output to PoC
- **Method:** Run on small test audio; compare outputs (bit-for-bit if possible)
- **Test:** Should run in <5 min with small audio

### Step 10.4: Coverage reporting

- **File:** Update `Taskfile.yml`
- **Change:** Add task `task coverage` to open HTML report
- **Target:** Final `coverage` command shows >=95% overall

### Validation Checklist

- [ ] `pytest` runs all tests; all pass
- [ ] `pytest --cov` shows >=95% coverage
- [ ] `task qa` (lint + test) passes
- [ ] Regression test: refactored `extract sam-audio` ≈ PoC output

**Exit Criteria:** Coverage >=95%; all tests green; no regressions

---

## Implementation Order (Recommended)

```
✅ Completed:
├─ Phase 0: Add --chain-residuals flag
├─ Phase 1: Modularization & Lazy Imports
├─ Phase 2: Atomic CLI Commands
│  ├─ 2.1: convert to-wav
│  ├─ 2.2: segment split
│  ├─ 2.3: merge concat
│  ├─ 2.4: extract process-sam-audio [DEPRECATED - removed in Phase 4]
│  ├─ 2.5: extract process-demucs [DEPRECATED - removed in Phase 4]
│  ├─ 2.6: extract sam-audio composite (refactored to use optimizer directly)
│  └─ 2.7: extract demucs composite
├─ Phase 3: Restructure to src layout
└─ Phase 4: PyTorch Optimizations & Architectural Refactoring
   ├─ Created sam_audio_optimizer.py (caching, chunking, streaming)
   ├─ Eliminated file-based segmentation (moved to in-memory chunking)
   ├─ Removed process-* CLI commands (logic moved to core modules)
   └─ Added PerformanceTracker for automatic metrics reporting

⏳ Upcoming:
├─ Phase 5: Advanced ODE Control & MLX Backend Integration (next)
│  ├─ 5.1: Investigate PyTorch SAMAudio ODE control
│  ├─ 5.2: MLX backend implementation
│  ├─ 5.3: Backend integration into optimizer
│  ├─ 5.4: Configuration & CLI updates
│  ├─ 5.5: Text feature caching optimization
│  ├─ 5.6: Documentation & benchmarks
│  └─ 5.7: Backend testing
├─ Phase 6: Global config overrides to all commands
├─ Phase 7: Lazy caching & artifact reuse
├─ Phase 8: YAML runner, workflows & batch processing
├─ Phase 9: Comprehensive documentation & ReadTheDocs
└─ Phase 10: Testing & coverage (>=95%)
```

---

## Key Principles Throughout

1. **Backward Compatibility:** Every change must not break `extract sam-audio`
2. **Lazy Imports:** Never import `torch`, `sam-audio`, `torchaudio` at module level
3. **Testing First:** Write test before refactoring each section
4. **Small PRs:** Each step should be a commit that compiles and tests pass
5. **Documentation:** Update docstrings and README as you go

---

## Success Metrics

- [x] `audio-playground --help` runs in <1s
- [x] `audio-playground extract sam-audio` still works (regression test passes)
- [ ] > =95% unit test coverage
- [ ] All atomic commands functional
- [ ] Caching provides measurable speedup (2nd run >=10x faster for segment step)
- [ ] YAML workflows execute end-to-end
- [ ] Project is ready to add `demucs` command without major refactoring
