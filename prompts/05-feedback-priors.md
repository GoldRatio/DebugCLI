# Prompt 05 — Outcome-fed subsystem priors and ticket outcome flow

Implements: `plan/subsystem.py` learns symptom→subsystem likelihood from
*verified* outcomes (never from unverified runs), and the ticket layer gains an
outcome record step so the case store is populated from the approval flow too.

## Mission

The collector plan gets smarter with fleet history — but only from outcome-
verified cases, Laplace-smoothed, with sane caps so the static heuristic table
is never drowned out by small samples.

## Read first

- `src/harness/plan/subsystem.py` (`classify`, `_SYMPTOM_TO_SUBSYSTEM`)
- `src/harness/plan/profile.py`
- `src/harness/diagnosis/case_store.py`, `src/harness/diagnosis/schema.py`
  (`CaseOutcome`)
- `src/harness/operator/tickets.py`, `src/harness/operator/cli.py`
  (subcommand registration style)
- `tests/unit/test_summarize.py` for test style

## Changes

### 1. `plan/subsystem.py` — priors

- `@dataclass class PriorModel`:
  - `defaults: dict[str, list[str]]` — the existing static table
  - `keyword_multipliers: dict[str, dict[str, float]]` — keyword →
    per-subsystem multiplier (unknown keywords default 1.0)
  - property `covers(keyword)` / `empty`
- `classify(symptom_text: str, priors: PriorModel | None = None)`:
  - Compute static scores as today; when `priors` is given, multiply each
    keyword hit's contribution by its multiplier (clamped to [0.5, 2.0]).
  - Behavior identical to today when `priors` is None or empty.

### 2. `plan/priors_build.py` (new module) — derive priors from verified outcomes

- `build_priors(cases: list[CaseOutcome], min_verified: int = 10) -> PriorModel | None`:
  - Count, per symptom keyword (reuse tokenization from `classify`), the
    outcome-weighted prevalence per subsystem: fixed=1.0, partial=0.5,
    not_fixed=-0.75 (negative evidence), inconclusive/unknown=0.25.
  - Laplace smoothing: `mult = (count_s + 1) / (count_all + 1) / (1 / n_subsystems)`.
  - Returns None (callers keep static behavior) until at least `min_verified`
    outcome-recorded cases exist fleet-wide.
  - Clamp multipliers into [0.5, 2.0] at build time.

### 3. Ticket outcome flow

- `Ticketing` gains `record_outcome(ticket_id: str, outcome: str,
  actions_taken: list[str]) -> str` (status line). `NoOpTicketing` implements
  it by delegating to the same `harness report` path (print the case id);
  semantic default `raise NotImplementedError` for the ABC is fine — but the
  harness must not crash when a backing system lacks it: wrap in try/except in
  the caller.
- CLI: after the approval gate completes, print the recorded
  `pending_case.json` id and the one-liner
  `harness report --run <id> --outcome ...` so the operator (or automation)
  closes the loop.

### 4. CLI — `harness priors update`

- `harness priors update --cases <dir> [--out priors.json]` — builds and
  writes the prior file; prints the `min_verified` gate state (e.g. "5/10
  verified cases, priors inactive").
- `diagnose`/`console`/`session` load `priors.json` from the out-dir when
  present and pass it to `plan_collection`/`classify`; failures to load are
  warnings, never errors.

## Acceptance criteria

- `tests/unit/test_priors.py`:
  - `classify` unchanged when priors None; multiplier applied and clamped with
    priors present.
  - `build_priors` returns None under `min_verified`; with enough fixed memory
    cases, "mce"/"ecc" boost "memory" above static ranking; a keyword with
    mostly `not_fixed` storage outcomes dampens "storage" below 1.0.
  - Priors file round-trip through the CLI path.
- Ticket test: `record_outcome` on NoOp produces a case record via the report
  path; a ticketing backend without the method doesn't crash the flow.
- `pytest tests/unit -q`, `ruff check src tests`.