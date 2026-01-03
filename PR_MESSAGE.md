## Phase 1: Modularization & Lazy Imports

Implements Phase 1 of the refactoring plan: extract reusable components into a `core/` package with lazy imports for fast CLI startup.

### Core Changes

**Modularization:**
- Created `core/` package with plain functions (not classes):
  - `wav_converter.py` - audio format conversion
  - `segmenter.py` - audio segmentation with fixed window size
  - `merger.py` - segment concatenation and output merging
- Refactored `sam_audio.py` to use core modules (178 lines removed)

**Lazy Imports:**
- Moved heavy imports (torch/torchaudio/sam_audio) inside functions
- `--help` now runs in <1s (previously ~10s)
- Module-level imports only for lightweight libraries

**Dependency Injection:**
- Logger passed from `app_context` instead of creating new instances
- Fixed duplicate logging issue (`logger.propagate = False`)

### Additional Features

**Segmentation:**
- Replaced variable-length segments (min/max) with fixed window size
- Default: 10.0s windows (eliminates most SAM-Audio padding)
- CLI: `--segment-window-size` to override
- CLI: `--max-segments` to cap segments for testing

**Audio Processing:**
- `--sample-rate` to resample final outputs
- Restored prompt batching (was missing, config existed but unused)
- Fixed batching bug (duplicate audio paths for SAM-Audio processor)

**Diagnostics:**
- New `doctor check-durations` command to analyze segment padding

**Configuration:**
- Consolidated all defaults to `app_config.py` (single source of truth)
- CLI args only override when explicitly provided
- Config respects `.env` and CLI overrides correctly

### Technical Details

- All type checking passes with `mypy --strict`
- Fixed float precision in segment timing (no accumulating rounding errors)
- Phase 0 default changed: `chain_residuals` now defaults to `False`

### Testing

- ✅ Backward compatible (existing functionality preserved)
- ✅ `--help` runs in <1s
- ✅ All CLI flags functional
- ✅ Config overrides working (`.env` and CLI)
- ✅ Type checking passes

Closes Phase 0 and Phase 1 of the refactoring plan.
