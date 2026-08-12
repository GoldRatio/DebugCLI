# Prompt 03 — Case outcome capture (the learning loop's ground truth)

Implements: every diagnosis can become a structured, append-only case record
with the human action taken and the verified outcome — the signal everything
else in this series consumes.

## Mission

A `harness report` CLI that turns a diagnosis run into a `CaseOutcome` record,
a `CaseStore` for them, verification linkage, and audit hashing. Outcome is
stored once; nothing is ever overwritten.

## Read first

- `src/harness/diagnosis/schema.py` (pydantic models, `schema_version` pattern)
- `src/harness/diagnosis/verifier.py` (VerifyResult)
- `src/harness/diagnosis/llm.py` (how the LLM adapter is identified)
- `src/harness/audit/auditlog.py` (append API to reuse; read to learn the call
  signature — do not change its internals)
- `src/harness/operator/cli.py` (subcommand registration; mirror `verify`)
- `src/harness/operator/tickets.py`
- `tests/unit/test_verifier_parts.py`, `tests/unit/test_audit_scorer.py` for style

## Changes

### 1. `diagnosis/schema.py` — add `CaseOutcome`

Pydantic model with `schema_version` (int, start at 1):
- `run_id: str` — the harness run directory id.
- `target_id: str` — alias / rack-cable / address used
- `model_key: str | None`, `model_source: str | None`
- `symptom: str`
- `subsystem_primary: str | None`
- `actions_recommended: list[str]` (step + action text)
- `actions_taken: list[str]`
- `outcome: str` — enum: `"fixed" | "partial" | "not_fixed" | "inconclusive" | "unknown"`
- `verification_verdict: str | None` (VerifyResult.verdict), `verification_delta: dict`
- `llm_ident: str` — e.g. `"openai/gpt-4o"` or `"stub"`; must record WHICH model
  produced the diagnosis (swap-safe).
- `evidence_hash: str` — sha256 over a canonical JSON serialization of the
  collected RegisterDumps (deterministic key order).
- `created_at: str` (ISO UTC)

### 2. `diagnosis/case_store.py` (new module)

- `CaseStore(root: Path)` where `root = <out-dir>/cases` (default under
  `harness_runs/` — git-ignored).
- `save(record: CaseOutcome)`: write `{run_id}.json` atomically
  (write tmp, `replace`); refuse to overwrite an existing file (append-only).
- `get(run_id) -> CaseOutcome | None`; `all() -> list[CaseOutcome]`.
- `index.json`: rebuilt manifest `{run_id: {model_key, outcome, created_at}}`
  written after each save; tolerate a missing/stale index by rescanning files.
- No secrets anywhere: reject fields containing `secret_dir`/key file paths if
  passed (defense in depth).

### 3. `diagnosis/verifier.py` — record path

- `Verifier.record(case: CaseOutcome, store: CaseStore)` → audits
  (hash + append via `auditlog`) and persists. Honest hash: include the
  record's own serialization; on mismatch with stored bytes, raise (WORM
  invariant) rather than repair.

### 4. LLM identity

- `EngineContext` gains `llm_ident: Callable[[], str] | None` (default
  `"stub"`); CLI provides it from the configured `--llm` + model name.

### 5. CLI — `harness report`

Mirror the `verify` subcommand style:
- `harness report --run <run_id> --outcome {fixed,partial,not_fixed,inconclusive}
  [--taken "reseated DIMM 3"] [--cases <dir>]`
- Loads the run's stored evidence (from the run directory under
  `harness_runs/`) to compute `evidence_hash`; fills the record from stored
  run metadata (symptom, model, actions); writes via `CaseStore`.
- `harness report --run <id> --status` prints the record if present.
- Also auto-seed: at the end of each diagnosis, CLI writes a
  `pending_case.json` (outcome="unknown") inside the run dir so `report` has a
  base to fill — implement this seeding in `cli.py` where the run directory is
  finalized.

## Acceptance criteria

- `tests/unit/test_case_store.py`: save/load round-trip; overwrite refusal;
  index rebuild after deleting index.json; `evidence_hash` is deterministic
  for identical dumps and differs for changed dumps.
- `tests/unit/test_verifier_parts.py` additions: `record()` persists and
  appends an audit entry whose hash covers the record.
- CLI test: `report` on a seeded run dir fills `actions_taken`/`outcome`;
  `report --status` prints; a second report with a different outcome is
  rejected (WORM).
- `pytest tests/unit -q`, `ruff check src tests`.