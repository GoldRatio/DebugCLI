# Prompt 07 — `harness eval`: offline regression against the case library

Implements: the safety net for every change to prompts, catalog, retrieval, or
LLM model. Replays held-out verified cases through the CURRENT pipeline, reports
verdict accuracy, calibration error, and retrieval quality; diffs against a
stored baseline and flags regressions.

## Mission

A deterministic-ish, auditable `harness eval` command with an explicit holdout
that is excluded from case retrieval, priors, and calibration during the eval.
The runbook (Prompt 08) and every future prompt/catalog/LLM change gate on this.

## Read first

- `prompts/03-outcome-capture.md`, `prompts/04-case-retrieval.md`,
  `prompts/06-calibration.md` (contracts below)
- `src/harness/diagnosis/engine.py` (how a diagnosis is assembled; what to
  reuse vs. what to replay)
- `src/harness/diagnosis/prompt.py`, `src/harness/diagnosis/llm.py`
  (adapter shape — known `stub` exists; a real adapter is optional)
- `src/harness/operator/cli.py` (subcommand registration style)
- `tests/unit/test_llm.py`, `tests/unit/test_diagnosis_flow.py` for style

## Changes

### 1. Holdout partition

- Deterministic split: `hashlib.sha256(run_id.encode()).hexdigest()`, first two
  hex digits → `int(d, 16)`; `n < int(225 * frac)` → holdout (≈ frac of 256,
  stable across runs). `--holdout-frac` flag, default 0.25.
- The holdout id set is passed as `exclude_holdout` to `CaseLibrary.similar`
  (Prompt 04) and excluded from `build_priors`/`build_calibration` inputs in
  eval mode.

### 2. Replay semantics

For each holdout case:
- Rebuild the evidence block the LLM saw, deterministically, from the case
  record: symptom, model_key, and the stored decoded-register lines
  (`evidence` field — see Prompt 08 runbook note; store what's needed in
  `CaseOutcome.evidence_summary: list[str]` in this prompt if missing, as an
  optional field).
- Re-run CURRENT retrieval with `rag_platform_filter=model_key` over the
  current doc library; take the top-3 snippet titles.
- Assertions recorded per case:
  - `verdict_hit`: replay LLM's structured `state` ∈ {fault,degraded,healthy,
    unknown} matches the case's stale verdict direction: fault/degraded vs
    outcome=fixed (was a fault, is now fixed). Treat "unknown" replay verdicts
    as misses.
  - `citation_support`: fraction of the replay's references found in retrieved
    snippets (reuse `score_diagnosis` internals if convenient).
  - `retrieval_recall`: overlap of replay-retrieved snippet titles with the
    original run's cited snippet titles (stored in `CaseOutcome.cited_titles`).
- Report (JSON + human table):
  - per-subsystem verdict accuracy, ECE on replayed confidences vs outcomes
    (reuse the binning from `calibration.build_calibration` for bin edges),
    mean citation support, mean retrieval recall, case counts, holdout ids.
- Baseline: write `eval/baseline.json` on first run; on later runs diff and
  exit code 1 when verdict accuracy or ECE regresses beyond `--tolerance`
  (default 0.05 absolute) — with a human-readable regression list. `--update-baseline`
  rewrites it deliberately.

### 3. CLI

- `harness eval --cases <dir> [--lib <dir>] [--llm {stub,openai,gemini}]
  [--holdout-frac 0.25] [--out eval_report.json] [--update-baseline]`
- The current pipeline code paths (retrieval, prompt build, scorer) are used
  directly — an eval that cannot construct them reports a hard error, not a
  silent pass.

## Acceptance criteria

- `tests/unit/test_eval.py`:
  - Holdout split is deterministic and stable; `exclude_holdout` propagates to
    case retrieval (assert via a spy `CaseLibrary`).
  - Replay with the stub LLM: produces a report with the right shape; a
    fixture case with an intentionally degraded retrieval yields a visible
    `retrieval_recall` drop.
  - Baseline diff: first run writes `baseline.json`; a forced regression
    (tampered baseline) exits 1 with `--tolerance 0.01`.
- `pytest tests/unit -q`, `ruff check src tests`.