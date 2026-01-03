# Audio-Playground Refactoring Plan

## Overview

Transform the monolithic `extract sam-audio` command into a modular, testable, cacheable toolkit while maintaining backward compatibility.

**Total Phases:** 5 (including current PoC)
**Estimated Effort:** ~40-50 incremental changes
**Testing Target:** >=95% coverage by end

---

## Implementation Status

**Current Phase:** Phase 1 Complete | Next: Phase 2

- ✅ Phase 0: Complete
- ✅ Phase 1: Complete
- ⏳ Phase 2: Not Started
- ⏳ Phase 3: Not Started
- ⏳ Phase 4: Not Started
- ⏳ Phase 5: Not Started

---

## ✅ Phase 0: Add `--chain-residuals` Flag (Immediate)

### Goal

Enable the existing residual-chaining logic conditionally before refactoring.

### Step 0.1: Add config option

- **File:** `audio_playground/config/app_config.py`
- **Change:** Add field `chain_residuals: bool = True` to `AudioPlaygroundConfig`
- **Rationale:** Defaults to `True` (current behavior) for backward compatibility
- **Test:** Verify field loads from `.env` and CLI override works

### Step 0.2: Update `sam_audio.py` to respect flag

- **File:** `audio_playground/cli/extract/sam_audio.py`
- **Change:** Wrap Phase 1's "residual chaining" block with condition:
  ```python
  if config.chain_residuals and len(prompts_list) > 1:
      # existing chaining logic
  ```
- **Rationale:** Allows users to skip cumulative residual computation
- **Test:** Run with flag on/off, verify output files match expectations

### Step 0.3: Add CLI flag to command

- **File:** `audio_playground/cli/extract/sam_audio.py`
- **Change:** Add `@click.option("--chain-residuals/--no-chain-residuals", default=True, ...)`
- **Test:** Verify `audio-playground extract sam-audio --help` shows the flag

### Validation Checklist

- [x] `audio-playground extract sam-audio --help` shows `--chain-residuals` flag
- [x] Running with `--no-chain-residuals` produces only `sam-{prompt}.wav` files (no `sam-other.wav`)
- [x] Running without flag produces current output (backward compatible)
- [x] `.env` can override via `CHAIN_RESIDUALS=false`

**Exit Criteria:** Existing tests pass + new flag functional

---

## ✅ Phase 1: Modularization & Lazy Imports

### Goal

Extract reusable components into a `core/` package with lazy imports for speed.

### Implementation Summary

**Completed Changes:**
- Created `core/` package with plain functions (not classes):
  - `wav_converter.py` - audio format conversion
  - `segmenter.py` - audio segmentation with fixed window size
  - `merger.py` - segment concatenation and merging
- Implemented lazy imports (torch/torchaudio/sam_audio only)
- Refactored `sam_audio.py` to use core modules
- Logger passed from app_context (dependency injection pattern)
- Restored prompt batching functionality
- Fixed duplicate logging issue (logger.propagate = False)
- All type checking passes with `--strict`

**Additional Features:**
- `--sample-rate` CLI option to resample outputs
- `--max-segments` to cap number of segments (for testing)
- `--segment-window-size` to set fixed segment length (default: 10.0s)
  - Eliminates most padding from SAM-Audio model
  - Only last (remainder) segment gets padded
- `audio-playground doctor check-durations` diagnostic command
- Consolidated defaults to single source of truth (`app_config.py`)

### Step 1.1: Create `core/` package structure

- **Create:**
  ```
  audio_playground/core/
    ├── __init__.py
    ├── wav_converter.py
    ├── segmenter.py
    └── merger.py
  ```
- **Rationale:** Separate concerns; enable independent testing
- **Test:** `from audio_playground.core import WavConverter` works

### Step 1.2: Extract `WavConverter` class

- **File:** `audio_playground/core/wav_converter.py`
- **Responsibility:**
  - Convert MP4/other → WAV
  - Load/save audio files via `pydub`
  - Detect format and choose appropriate conversion method
- **Key Detail:** **Lazy imports** – import `pydub`, `subprocess`, `shutil` **inside methods**, not at module level
- **Signature:**

  ```python
  class WavConverter:
      @staticmethod
      def convert_to_wav(src_path: Path, dst_path: Path) -> None:
          """Convert any audio format to WAV."""

      @staticmethod
      def load_audio_duration(path: Path) -> float:
          """Return duration in seconds."""
  ```

- **Test:** Create unit test for each conversion type (MP4 → WAV, WAV → WAV)

### Step 1.3: Extract `Segmenter` class

- **File:** `audio_playground/core/segmenter.py`
- **Responsibility:**
  - Create even-length segments from total duration
  - Split WAV file into segment files
  - Track segment metadata (start time, duration)
  - Save metadata to JSON
- **Key Detail:** **Lazy imports** – import `pydub` inside methods
- **Signature:**

  ```python
  class Segmenter:
      @staticmethod
      def create_segments(
          total_length: float,
          min_length: float = 9.0,
          max_length: float = 17.0
      ) -> list[float]:
          """Return list of segment lengths."""

      @staticmethod
      def split_to_files(
          audio_path: Path,
          output_dir: Path,
          segment_lengths: list[float]
      ) -> tuple[list[Path], list[tuple[float, float]]]:
          """Split WAV, return segment files and metadata."""
  ```

- **Test:** Create unit tests for segment calculation (edge cases: very short, very long)

### Step 1.4: Extract `Merger` class

- **File:** `audio_playground/core/merger.py`
- **Responsibility:**
  - Load segment files by pattern
  - Concatenate via numpy/torch
  - Save to output directory
  - Extract prompt from filename patterns
- **Key Detail:** **Lazy imports** – import `torch`, `torchaudio` inside methods
- **Signature:**

  ```python
  class Merger:
      @staticmethod
      def concatenate_segments(segment_files: list[Path]) -> Tensor:
          """Load and concatenate audio segments."""

      @staticmethod
      def find_prompts_from_files(tmp_dir: Path) -> dict[str, list[Path]]:
          """Scan dir for {segment}-target-{prompt}.wav patterns."""

      @staticmethod
      def merge_and_save(
          tmp_dir: Path,
          output_dir: Path,
          chain_residuals: bool = True
      ) -> None:
          """Merge all segments and save outputs."""
  ```

- **Test:** Create unit tests for concatenation, pattern matching

### Step 1.5: Refactor Phase 1 to use `WavConverter` + `Segmenter`

- **File:** `audio_playground/cli/extract/sam_audio.py`
- **Change:** Replace inline logic in `phase_1_segment_and_process()` with:

  ```python
  # Convert
  wav_file = tmp_path / "audio.wav"
  WavConverter.convert_to_wav(src_path, wav_file)

  # Segment
  total_length = WavConverter.load_audio_duration(wav_file)
  segment_lengths = Segmenter.create_segments(total_length, ...)
  segment_files, metadata = Segmenter.split_to_files(wav_file, tmp_path, segment_lengths)
  ```

- **Test:** Existing `extract sam-audio` tests still pass

### Step 1.6: Refactor Phase 2 to use `Merger`

- **File:** `audio_playground/cli/extract/sam_audio.py`
- **Change:** Replace inline logic in `phase_2_blend_and_save()` with:
  ```python
  Merger.merge_and_save(tmp_path, target_path, config.chain_residuals)
  ```
- **Test:** Existing output matches (bit-for-bit if possible, or at least audio quality)

### Step 1.7: Move heavy imports to Phase 1 & 2

- **File:** `audio_playground/cli/extract/sam_audio.py`
- **Change:** Move `import torch`, `import torchaudio`, `from sam_audio import ...` to **inside** `phase_1_segment_and_process()`
- **Rationale:** `--help` and `--version` become instant (no torch compile)
- **Test:** Run `audio-playground --help` and time it (should be <1s)

### Validation Checklist

- [x] `from audio_playground.core import WavConverter, Segmenter, Merger` all work
- [x] Each class has >=80% unit test coverage
- [x] `audio-playground extract sam-audio` still produces identical output
- [x] `--help` runs in <1s (lazy imports verified)
- [x] `.env` and CLI flags still override correctly

**Exit Criteria:** Phase 1 code passes tests; `--help` is fast

---

## ⏳ Phase 2: Atomic CLI Commands

### Goal

Break `extract sam-audio` into reusable commands: `convert`, `segment`, `process`, `merge`.

### Step 2.1: Create `convert` command

- **File:** `audio_playground/cli/convert/__init__.py` + `to_wav.py`
- **Responsibility:** Convert any audio → WAV
- **Usage:** `audio-playground convert to-wav --src input.mp4 --target output.wav`
- **Implementation:** Wrap `WavConverter.convert_to_wav()`
- **Test:** Verify output is valid WAV

### Step 2.2: Create `segment` command

- **File:** `audio_playground/cli/segment/__init__.py` + `split.py`
- **Responsibility:** Split WAV into chunks
- **Usage:** `audio-playground segment split --src input.wav --output-dir ./segments --min 9 --max 17`
- **Implementation:** Wrap `Segmenter.split_to_files()`, save manifest
- **Output:** `./segments/segment-000.wav`, `./segments/segment-001.wav`, `./segments/manifest.json`
- **Test:** Verify segments sum to original length (within tolerance)

### Step 2.3: Create `merge` command

- **File:** `audio_playground/cli/merge/__init__.py` + `concat.py`
- **Responsibility:** Concatenate segment files
- **Usage:** `audio-playground merge concat --input-dir ./segments --pattern "segment-*target*.wav" --output result.wav`
- **Implementation:** Wrap `Merger.concatenate_segments()`
- **Test:** Verify output matches original (if only converting)

### Step 2.4: Create `extract` → `process` command (renamed internally)

- **File:** `audio_playground/cli/extract/process.py` (new)
- **Responsibility:** Run SAM-Audio on a single segment or set of segments
- **Usage:** `audio-playground extract process --segment segment-000.wav --prompts "bass,vocals" --output-dir ./out`
- **Implementation:** Refactor existing model inference logic from Phase 1
- **Test:** Verify produces `{segment}-target-{prompt}.wav` files

### Step 2.5: Make `extract sam-audio` a composite command

- **File:** `audio_playground/cli/extract/sam_audio.py`
- **Change:** Simplify to call the atomic commands in sequence:
  ```python
  # Phase 1: Convert → Segment → Process (on all segments)
  # Phase 2: Merge
  ```
- **Benefit:** Users can now manually `convert` → `segment` → skip processing → `merge` if desired
- **Test:** Output identical to current behavior

### Step 2.6: Add global config overrides to each command

- **File:** `audio_playground/cli/base.py` (new)
- **Responsibility:** Create a shared decorator/mixin for common options:
  ```python
  @click.option("--log-level", type=click.Choice([...]), help="...")
  @click.option("--device", default="auto", help="...")
  @click.option("--temp-dir", type=click.Path(), help="...")
  def common_options(func):
      """Decorator for shared CLI flags."""
  ```
- **Usage:** Apply to all commands to avoid repetition
- **Test:** Verify each command respects `--device`, `--log-level`, etc.

### Validation Checklist

- [ ] `audio-playground convert --help` works
- [ ] `audio-playground segment --help` works
- [ ] `audio-playground merge --help` works
- [ ] Running atomic commands in sequence produces same output as `extract sam-audio`
- [ ] Each command saves manifest/metadata for potential caching later

**Exit Criteria:** All atomic commands functional; `extract sam-audio` is composite

---

## ⏳ Phase 3: Lazy Caching & Artifact Reuse

### Goal

Avoid re-processing identical inputs by caching segment files and metadata.

### Step 3.1: Create cache manifest format

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

### Step 3.2: Create cache store

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

### Step 3.3: Integrate caching into `Segmenter`

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

### Step 3.4: Add `--no-cache` flag to commands

- **File:** `audio_playground/cli/segment/split.py` (and others)
- **Change:** Add `@click.option("--no-cache", is_flag=True, help="...")`
- **Test:** Verify `--no-cache` bypasses caching

### Step 3.5: Create cache management commands

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

## ⏳ Phase 4: YAML Runner & Workflows

### Goal

Allow users to define pipelines as YAML and run with `audio-playground run --config pipeline.yaml`.

### Step 4.1: Create workflow schema

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

### Step 4.2: Create workflow executor

- **File:** `audio_playground/core/workflow_executor.py` (new)
- **Responsibility:**
  - Execute steps sequentially
  - Pass outputs of one step to next
  - Handle errors gracefully
  - Log progress
- **Test:** Execute sample workflow; verify outputs

### Step 4.3: Create `run` command

- **File:** `audio_playground/cli/run.py` (new)
- **Usage:** `audio-playground run --config pipeline.yaml`
- **Implementation:** Load YAML, execute via `WorkflowExecutor`
- **Test:** Run sample workflow end-to-end

### Step 4.4: Add input windowing to `extract process`

- **File:** `audio_playground/cli/extract/process.py` (modify)
- **Change:** Add `--start-time` and `--end-time` options
  ```python
  @click.option("--start-time", type=float, help="Start offset in seconds")
  @click.option("--end-time", type=float, help="End offset in seconds")
  ```
- **Implementation:** Trim input audio before processing
- **Test:** Process specific regions; verify output duration

### Step 4.5: Document YAML format & examples

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

## ⏳ Phase 5: Testing & Coverage

### Goal

Achieve >=95% unit test coverage; no regressions.

### Step 5.1: Unit tests for `core/` modules

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

### Step 5.2: Integration tests for CLI commands

- **Files:** Create `tests/cli/test_*.py` for each command
- **Coverage Target:** >=90% per command
- **Test Types:**
  - Command execution with various flags
  - Output file creation
  - Config override via CLI

### Step 5.3: Regression test for `extract sam-audio`

- **File:** `tests/integration/test_extract_sam_audio.py`
- **Goal:** Ensure refactored version produces identical output to PoC
- **Method:** Run on small test audio; compare outputs (bit-for-bit if possible)
- **Test:** Should run in <5 min with small audio

### Step 5.4: Coverage reporting

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
Week 1:
├─ Phase 0: Add --chain-residuals flag
└─ Phase 1.1-1.4: Extract core modules (WavConverter, Segmenter, Merger)

Week 2:
├─ Phase 1.5-1.7: Refactor sam_audio.py + lazy imports
└─ Phase 5.1: Write unit tests for core modules

Week 3:
├─ Phase 2.1-2.3: Create atomic commands (convert, segment, merge)
└─ Phase 2.4-2.6: Refactor extract as composite

Week 4:
├─ Phase 3.1-3.3: Implement caching
├─ Phase 3.4-3.5: Cache management commands
└─ Phase 5.2: Integration tests

Week 5:
├─ Phase 4.1-4.5: YAML runner + workflows
└─ Phase 5.3-5.4: Regression + coverage
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
- [ ] >=95% unit test coverage
- [ ] All atomic commands functional
- [ ] Caching provides measurable speedup (2nd run >=10x faster for segment step)
- [ ] YAML workflows execute end-to-end
- [ ] Project is ready to add `demucs` command without major refactoring
