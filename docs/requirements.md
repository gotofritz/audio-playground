# Audio-Playground Refactoring Plan

## Overview

Transform the monolithic `extract sam-audio` command into a modular, testable, cacheable toolkit while maintaining backward compatibility.

**Total Phases:** 8
**Estimated Effort:** ~40-50 incremental changes
**Testing Target:** >=95% coverage by end

---

## Implementation Status

**Current Phase:** Phase 3 Complete | Ready for Phase 4

- ✅ Phase 0: Complete
- ✅ Phase 1: Complete
- ✅ Phase 2: Complete (All atomic and composite commands implemented)
- ✅ Phase 3: Complete (Restructured to src layout, replaced mypy with ty)
- ⏳ Phase 4: Not Started
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

## 🚧 Phase 2: Atomic CLI Commands

### Goal

Break `extract sam-audio` into reusable commands: `convert`, `segment`, `process-sam-audio`, `process-demucs`, `merge`.
Support both SAM-Audio and Demucs models with model-specific processing commands.

### Step 2.1: Create `convert` command ✅

**Status:** ✅ Complete - Created `cli/convert/to_wav.py` wrapping `convert_to_wav()` from `core/wav_converter.py`. Common option decorators created for consistency.

### Step 2.2: Create `segment` command ✅

**Status:** ✅ Complete - Created `cli/segment/split.py` wrapping `create_segments()` and `split_to_files()` from `core/segmenter.py`. Outputs segment files and `segment_metadata.json`.

### Step 2.3: Create `merge` command ✅

**Status:** ✅ Complete - Created `cli/merge/concat.py` wrapping `concatenate_segments()` from `core/merger.py`. Supports glob patterns. Auto-detects sample rate from first file.

### Step 2.4: Create `extract process-sam-audio` command ✅

**Status:** ✅ Complete - Created `cli/extract/process_sam_audio.py` for processing audio segments with SAM-Audio model. Supports multiple segments, glob patterns, and batch processing. Outputs `{segment}-{prompt}.wav` files (suffix applied at merge step).

### Step 2.5: Create `extract process-demucs` command ✅

**Status:** ✅ Complete - Created `cli/extract/process_demucs.py` for Demucs model processing. Supports single audio file input with stem separation (no segmentation needed). Outputs separated stems (drums, bass, other, vocals). Includes progress bar support and configurable parameters via app_config (model, shifts, workers).

### Step 2.6: Make `extract sam-audio` a Composite Command ✅

**Status:** ✅ Complete - Refactored `extract sam-audio` as a composite command orchestrating atomic steps: convert to-wav → segment split → process-sam-audio → merge concat. Users can now run individual steps manually if desired.

### Step 2.7: Create `extract demucs` Composite Command ✅

**Status:** ✅ Complete - Created `cli/extract/demucs.py` composite command for full Demucs pipeline: convert to-wav → process-demucs. Simplified workflow (no segmentation needed) for stem separation.

## ✅ Phase 3: Restructure to src Layout

**Status:** ✅ Complete - Moved `audio_playground/` to `src/audio_playground/`. Updated `pyproject.toml` with `where = ["src"]`. Replaced mypy with ty for type checking.

## ⏳ Phase 4: PyTorch Performance Optimizations (Platform-Agnostic)

- **File:** `audio_playground/core/sam_audio_optimizer.py` (new)
- **Responsibility:** Performance optimizations that work on all platforms (Windows, Linux, Mac, CUDA, CPU)
- **Rationale:** Improve processing speed and memory efficiency without platform-specific dependencies
- **Implementation:**

  **Text Feature Caching:**

  ```python
  class PromptCache:
      """Cache text embeddings to avoid re-encoding same prompts"""
      def get_or_encode(self, prompts: list[str], encoder) -> Tensor:
          # Hash prompts, return cached embeddings if available
          # Huge win for multi-segment processing with same prompts
  ```

  **Chunked Processing with Crossfade:**

  ```python
  def process_long_audio(
      audio_path: Path,
      prompts: list[str],
      chunk_duration: float = 30.0,
      overlap_duration: float = 2.0,
      crossfade_type: str = "cosine"  # or "linear"
  ) -> dict[str, Tensor]:
      """
      Process long audio files in overlapping chunks to reduce peak memory.
      Blends chunks with cosine/linear crossfade to avoid artifacts.
      """
  ```

  **Streaming/Generator Mode:**

  ```python
  def process_streaming(
      audio_path: Path,
      prompts: list[str],
      chunk_duration: float = 15.0
  ) -> Generator[tuple[str, Tensor], None, None]:
      """
      Yield results chunk-by-chunk as they're ready.
      First audio available in ~10-15s instead of waiting for full file.
      Enables interactive applications and progress monitoring.
      """
  ```

  **Configurable ODE Solvers:**

  ```python
  class SolverConfig:
      method: Literal["euler", "midpoint"] = "midpoint"  # euler=faster, midpoint=quality
      steps: int = 32  # Lower=faster but lower quality

  # Allow users to trade quality for speed:
  # - Euler + 16 steps: ~2x faster, slight quality loss
  # - Midpoint + 64 steps: Maximum quality, slower
  ```

  **Memory Management:**

  ```python
  def clear_caches(device: str) -> None:
      """Explicit cache clearing between chunks/batches"""
      import torch
      if device.startswith("cuda"):
          torch.cuda.empty_cache()
          torch.cuda.synchronize()
      # Add memory monitoring and warnings
  ```

- **Configuration Options:** Add to `app_config.py`:

  ```python
  # Performance optimization settings
  enable_prompt_caching: bool = True
  chunk_duration: float = 30.0  # For long-form processing
  chunk_overlap: float = 2.0
  crossfade_type: Literal["cosine", "linear"] = "cosine"
  ode_solver: Literal["euler", "midpoint"] = "midpoint"
  ode_steps: int = 32
  streaming_mode: bool = False  # Yield chunks as ready
  ```

- **CLI Integration:** Add options to `extract process-sam-audio`:

  ```python
  @click.option("--streaming", is_flag=True, help="Stream results chunk-by-chunk")
  @click.option("--solver", type=click.Choice(["euler", "midpoint"]), help="ODE solver method")
  @click.option("--solver-steps", type=int, help="Number of solver steps (lower=faster)")
  @click.option("--chunk-duration", type=float, help="Chunk size for long audio")
  ```

- **Expected Performance Gains:**

  - Text caching: 20-30% speedup for multi-segment processing
  - Chunked processing: Enables arbitrarily long audio (previously limited by memory)
  - Streaming: First results in ~10-15s vs full processing time
  - Euler solver: ~2x faster with minimal quality loss
  - Memory management: Reduces OOM errors on large files

- **Test:** Benchmark before/after on 2-minute audio file; verify crossfade smoothness; test streaming mode

---

## ⏳ Phase 5: MLX Backend Integration (Apple Silicon Fast Path)

- **Files:**
  - `audio_playground/core/backends/mlx_backend.py` (new)
  - `audio_playground/core/backends/pytorch_backend.py` (new)
  - `audio_playground/core/backends/__init__.py` (new)
- **Responsibility:** Optional MLX backend for Mac M1/M2/M3 users (10-50x faster than PyTorch on Apple Silicon)
- **Rationale:** MLX is optimized for Apple's unified memory architecture; massive speedups on M-series chips
- **Implementation:**

  **Backend Abstraction:**

  ```python
  # audio_playground/core/backends/__init__.py
  from abc import ABC, abstractmethod

  class AudioBackend(ABC):
      @abstractmethod
      def load_model(self, model_name: str, device: str):
          pass

      @abstractmethod
      def separate(self, audio_path: Path, prompts: list[str]) -> dict[str, Tensor]:
          pass

  def get_backend(backend: str = "auto") -> AudioBackend:
      """
      Auto-detect best backend:
      - MLX if on Mac with M1/M2/M3 and mlx-audio installed
      - PyTorch otherwise
      """
      if backend == "auto":
          if platform.system() == "Darwin" and _has_mlx() and _has_apple_silicon():
              return MLXBackend()
          return PyTorchBackend()
      elif backend == "mlx":
          return MLXBackend()
      elif backend == "pytorch":
          return PyTorchBackend()
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
      def __init__(self):
          if not HAS_MLX:
              raise ImportError("mlx-audio not installed. Install with: pip install mlx-audio")

      def load_model(self, model_name: str, device: str):
          # MLX handles device automatically (uses Metal)
          self.model = SAMAudioMLX.from_pretrained(model_name)

      def separate(self, audio_path: Path, prompts: list[str]) -> dict[str, Tensor]:
          # Use MLX's optimized separate_long() for chunked processing
          return self.model.separate_long(
              audio_path.as_posix(),  # MLX accepts file paths directly
              prompts,
              chunk_duration=30.0,
              overlap_duration=2.0,
          )

      def separate_streaming(self, audio_path: Path, prompts: list[str]):
          # Use MLX's generator-based streaming
          yield from self.model.separate_streaming(audio_path.as_posix(), prompts)
  ```

  **PyTorch Backend:**

  ```python
  # audio_playground/core/backends/pytorch_backend.py
  class PyTorchBackend(AudioBackend):
      def __init__(self):
          self.model = None
          self.processor = None

      def load_model(self, model_name: str, device: str):
          from sam_audio import SAMAudio, SAMAudioProcessor
          self.model = SAMAudio.from_pretrained(model_name, map_location=device).to(device).eval()
          self.processor = SAMAudioProcessor.from_pretrained(model_name)

      def separate(self, audio_path: Path, prompts: list[str]) -> dict[str, Tensor]:
          # Use optimizations from Step 2.4c
          if hasattr(self, 'optimizer'):
              return self.optimizer.process_long_audio(audio_path, prompts)
          # Fallback to standard processing
          inputs = self.processor(audios=[audio_path.as_posix()], descriptions=prompts)
          return self.model.separate(inputs)
  ```

- **Configuration:** Add to `app_config.py`:

  ```python
  backend: Literal["auto", "mlx", "pytorch"] = "auto"
  # auto: Use MLX on Apple Silicon if available, PyTorch otherwise
  # mlx: Force MLX (will error if not available)
  # pytorch: Force PyTorch (e.g., for testing consistency)
  ```

- **CLI Integration:**

  ```python
  @click.option("--backend", type=click.Choice(["auto", "mlx", "pytorch"]), default="auto",
                help="Processing backend (auto=detect best, mlx=Apple Silicon only)")
  ```

- **Installation Instructions:** Update `README.md`:

  ```markdown
  ## Installation

  ### Standard (All Platforms)

  pip install -e .

  ### Apple Silicon (M1/M2/M3) - Faster Performance

  pip install -e .
  pip install mlx-audio # Optional: 10-50x speedup on Mac
  ```

- **Performance Comparison Table:**

  ```
  Platform          | Backend  | 2min Audio | Speedup
  ------------------|----------|------------|--------
  Mac M1/M2/M3      | PyTorch  | ~360s      | 1x
  Mac M1/M2/M3      | MLX      | ~100s      | 3.6x
  Mac M1/M2/M3 Fast | MLX+Euler| ~60s       | 6x
  Linux/Windows GPU | PyTorch  | ~200s      | 1.8x
  CPU               | PyTorch  | ~600s      | 0.6x
  ```

- **Fallback Behavior:** If MLX fails (e.g., older Mac, missing dependency), automatically fall back to PyTorch with warning
- **Test:**

  - Verify auto-detection on Mac with/without mlx-audio
  - Verify forced backend selection works
  - Verify fallback on import error
  - Benchmark MLX vs PyTorch on Apple Silicon (if available)
  - Ensure identical output quality between backends

- **Documentation:** Add `docs/BACKENDS.md` explaining:
  - When to use each backend
  - Installation instructions
  - Performance characteristics
  - Troubleshooting common issues

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
  ```
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
- [x] **Step 2.4:** `audio-playground extract process-sam-audio --help` works
- [x] **Step 2.4:** Process command handles single/multiple/glob segments
- [x] **Step 2.5:** `audio-playground extract process-demucs --help` works
- [x] **Step 2.5:** Demucs integration produces separated stems
- [x] **Step 2.6:** `extract sam-audio` composite command implemented
- [x] **Step 2.7:** `extract demucs` composite command implemented
- [x] **Phase 3:** Restructured to src layout
- [x] **Phase 3:** Updated pyproject.toml with src configuration
- [x] **Phase 3:** Replaced mypy with ty
- [ ] **Phase 4:** PyTorch optimizations implemented (caching, chunking, streaming)
- [ ] **Phase 4:** Benchmark shows expected performance gains
- [ ] **Phase 4:** Crossfade blending produces smooth audio (no artifacts)
- [ ] **Phase 5:** MLX backend auto-detection works on Apple Silicon
- [ ] **Phase 5:** Backend abstraction allows switching PyTorch ↔ MLX
- [ ] **Phase 5:** Fallback to PyTorch on missing MLX dependency
- [ ] **Phase 6:** Global config options applied to all commands

**Exit Criteria:** All atomic commands functional; both composite commands work; performance optimizations tested; backend abstraction complete; common options standardized

**Next Step:** Implement Phase 4 (PyTorch Performance Optimizations)

### Additional Improvements

**CI/CD Enhancements:** ✅ Complete

- GitHub Actions workflow for automated QA checks (runs `task qa` on all pushes/PRs)
- Coverage badge generation (auto-generated on main branch pushes, stored in `badges` branch)
- Coverage artifact upload (HTML reports with 7-day retention)

---

## ⏳ Phase 7: Lazy Caching & Artifact Reuse

### Goal

Avoid re-processing identical inputs by caching segment files and metadata.

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

### Step 8.4: Add input windowing to `extract process`

- **File:** `audio_playground/cli/extract/process.py` (modify)
- **Change:** Add `--start-time` and `--end-time` options
  ```python
  @click.option("--start-time", type=float, help="Start offset in seconds")
  @click.option("--end-time", type=float, help="End offset in seconds")
  ```
- **Implementation:** Trim input audio before processing
- **Test:** Process specific regions; verify output duration

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
- [ ] `--start-time` and `--end-time` trim correctly
- [ ] Example workflows run without error

**Exit Criteria:** YAML runner functional; workflows documented

---

## ⏳ Phase 9: Testing & Coverage

### Goal

Achieve >=95% unit test coverage; no regressions.

### Step 9.1: Unit tests for `core/` modules

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

### Step 9.2: Integration tests for CLI commands

- **Files:** Create `tests/cli/test_*.py` for each command
- **Coverage Target:** >=90% per command
- **Test Types:**
  - Command execution with various flags
  - Output file creation
  - Config override via CLI

### Step 9.3: Regression test for `extract sam-audio`

- **File:** `tests/integration/test_extract_sam_audio.py`
- **Goal:** Ensure refactored version produces identical output to PoC
- **Method:** Run on small test audio; compare outputs (bit-for-bit if possible)
- **Test:** Should run in <5 min with small audio

### Step 9.4: Coverage reporting

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
├─ Phase 2: Atomic CLI Commands (All steps complete)
│  ├─ 2.1: convert to-wav
│  ├─ 2.2: segment split
│  ├─ 2.3: merge concat
│  ├─ 2.4: extract process-sam-audio
│  ├─ 2.5: extract process-demucs
│  ├─ 2.6: extract sam-audio composite
│  └─ 2.7: extract demucs composite
└─ Phase 3: Restructure to src layout

⏳ Upcoming:
├─ Phase 4: PyTorch Performance Optimizations (next)
├─ Phase 5: MLX Backend Integration
├─ Phase 6: Global config overrides
├─ Phase 7: Caching implementation
├─ Phase 8: YAML runner + workflows
└─ Phase 9: Testing & coverage (>=95%)
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
