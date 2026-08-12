# Prompt 08 — LLM swap runbook (`docs/MODEL_SWAP.md`)

Implements: a documented procedure so swapping the diagnosis LLM is a routine,
calibrated, regression-gated operation — not a trust leap. Everything the model
must transfer is data, and everything model-specific is explicitly re-derived.

## Mission

Write `docs/MODEL_SWAP.md` (new file; see `docs/` for doc conventions) that an
operator follows to migrate diagnostics from one LLM to another. It must
reference the real commands this series built (running actual `harness` CLI
syntax), and it must be a *procedure document*, not a design essay.

## Read first

- `prompts/README.md` and prompts 03, 06, 07 (commands and contracts)
- `REVISED_SPEC.md` section 4 (confidence scoring intent)
- `docs/` for the existing documentation voice and level of detail

## Content requirements

1. **What transfers and what doesn't** — table:
   - Transfers as-is: case records, doc library + platform tags, register
     catalog, collector profiles, `harness eval` holdout set, audit chain.
   - Must be re-derived per model: calibration (`model_agreement` bins, Prompt
     06), prompt formatting quirks (citation contract), `model_agreement`
     behavior on sloppy citation.
2. **Pre-flight on the new model** — run the existing envs: `harness eval
   --cases <dir> --lib <lib> --llm <new>` and compare against the
   `--update-baseline` commit of the OLD model; acceptance thresholds:
   verdict accuracy and ECE within `--tolerance` (0.05 default); do not
   proceed on regression without an owner sign-off recorded in the audit.
3. **Recalibration** — `harness calibrate --cases <dir>` for the new ident;
   note that calibration is per-`llm_ident` and old-model calibration files
   are retained (rollback path), never overwritten.
4. **Citation contract check** — sample N holdout diagnoses for the fraction
   of actions carrying a real source+page Reference; drop below ~0.9 triggers
   a prompt-formatting rework, because `retrieval_citation_support` (and thus
   confidence) structurally depends on it.
5. **Rollback** — restore the previous `--llm` + calibration file + baseline;
   nothing in the case store or audit is touched by the swap (WORM).
6. **Record keeping** — every swap: date, old/new idents, eval diff, sign-off,
   appended to the audit log via the existing `harness report` (or an
   `audit.record` equivalent — choose the real one that exists at
   implementation time).

## Acceptance criteria

- File exists at `docs/MODEL_SWAP.md`, follows `docs/` style, every command it
  shows matches the actual CLI flags implemented in prompts 03/06/07.
- No new runtime code in this prompt (documentation only) — but if you add the
  audit-record step mentioned in 6 because it doesn't exist, keep it behind
  the real `auditlog` API.
- `pytest tests/unit -q` and `ruff check src tests` remain green (no code
  changes expected; if any, they must carry tests in the same change).