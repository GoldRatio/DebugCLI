# Prompt 06 — Outcome-calibrated confidence scoring

Implements: `model_agreement` stops being a fixed 0.5 and becomes measured —
per LLM identity, per subsystem, binned recalibration from verified outcomes.
Confidence then honestly reflects observed fleet fix-rates, and degrades
gracefully on novel evidence.

## Mission

Replace the static `model_agreement` default in the scorer with a calibration
store keyed by LLM identity; add a `harness calibrate` command that recomputes
it; never let calibration exaggerate when data is thin.

## Read first

- `src/harness/diagnosis/scorer.py` (`Scorer`, `score_diagnosis`, `apply_to`,
  `WEIGHTS`, `ConfidenceBreakdown` in schema)
- `src/harness/diagnosis/schema.py`
- `src/harness/diagnosis/case_store.py`, `CaseOutcome` (Prompt 03)
- `src/harness/operator/cli.py` (subcommand registration style)
- `tests/unit/test_audit_scorer.py` for style

## Changes

### 1. `diagnosis/calibration.py` (new module)

- `@dataclass Calibration`:
  - `llm_ident: str`
  - `subsystem_bins: dict[str, list[tuple[float, float, int]]]` —
    subsystem → [(bin_max, observed_fix_rate, n_samples)] with bin edges
    [0.0,0.2,0.4,0.6,0.8,1.0]
  - `aggregate: list[tuple[float, float, int]]` — same bins over all cases
  - `created_at: str`
- `build_calibration(cases: list[CaseOutcome], llm_ident: str,
  min_per_bin: int = 3) -> Calibration | None`:
  - Requires `outcome in ("fixed","partial","not_fixed")` (resolution counted
    as partial=0.5, fixed=1.0, not_fixed=0.0 for rate computation).
  - Only build when the LLM ident matches the case `llm_ident` (each model
    calibrates itself — this is what makes model swaps safe).
  - Bins with fewer than `min_per_bin` samples keep observed rate from their
    parent (aggregate) bin instead.
- `CalibrationStore(root: Path)` (default `<out-dir>/calibration`):
  `save(cal)`, `load(llm_ident) -> Calibration | None`, `all()`.
- `agreement_for(cal: Calibration | None, predicted: float,
  subsystem: str | None) -> float`:
  - None → 0.5 (unchanged behavior today).
  - Else: bin lookup (subsystem bins, falling back to aggregate) →
    `1 - abs(predicted - observed_rate)`, clamped [0,1].
  - Zero total samples in every bin → 0.5.

### 2. `scorer.py`

- `score_diagnosis(..., model_agreement: float | None = None)`:
  - `None` → try `CalibrationStore` (path from a new optional arg
    `calibration_root: Path | None`); resolve via `agreement_for` using the
    diagnosis's primary subsystem (from `subsystems_considered[0]` when
    present); fall back to 0.5 on any failure (no exceptions to callers).
- `ConfidenceBreakdown` gains `calibration_llm: str | None` (populated with the
  resolved ident or None) — schema evolution: optional field only.

### 3. CLI — `harness calibrate`

- `harness calibrate --cases <dir> [--llm <ident>] [--out <dir>]`:
  - Loads all case records, groups by `llm_ident` (or the given one), builds
    and saves a `Calibration` per ident, prints per-subsystem bin tables
    (bin, rate, n).
  - Non-zero exit suggestion (stderr note) when a subsystem bin has zero
    samples: "insufficient data — calibration falls back to aggregate".
- `diagnose`/`console`/`session`: pass `calibration_root` through so scoring
  uses it; missing store → 0.5, never an error.

## Acceptance criteria

- `tests/unit/test_calibration.py`:
  - `build_calibration` gate: returns None before `min_per_bin`; fixes rates
    computed with partial=0.5/not_fixed=0.0; sparse subsystem bins fall back
    to aggregate; empty calibration → `agreement_for`=0.5.
  - `agreement_for`: overconfident diagnosis (predicted 0.9, observed 0.4)
    gives agreement ≈ 0.5; well-calibrated gives ≈ 1.0; clamp at [0,1].
  - scorer integration: `score_diagnosis` with `calibration_root` uses the
    calibration (assert via `calibration_llm` on the breakdown + changed
    `model_agreement` for a skewed calibration), and equals 0.5 behavior when
    the root is empty.
- CLI test: `harness calibrate` writes per-ident files and prints bin tables.
- `pytest tests/unit -q`, `ruff check src tests`.