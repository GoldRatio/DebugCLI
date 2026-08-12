# Progress & file ownership — forward agent (prompts 01–04)

> Protocol: BOTH agents must read this file before touching any file, and
> update it BEFORE starting a prompt. Ownership is by FILE. A file
> marked `claimed` belongs to the agent who wrote it; the other agent waits
> (or coordinates) before editing. Commits: commit each prompt's work as a
> separate commit so merges stay clean, and reference the prompt number in the
> commit message.

## Forward agent (prompts 01–04) — claims

| Files (owned by forward agent) | Prompt |
|---|---|
| `src/harness/inspect/model.py` | 01 |
| `src/harness/targets/aliases.py`, `src/harness/targets/resolver.py` | 01 |
| `src/harness/diagnosis/engine.py`, `src/harness/diagnosis/prompt.py`, `src/harness/diagnosis/session.py`, `src/harness/diagnosis/llm.py` | 01, 03, 04 |
| `src/harness/docs/ingest/chunk.py`, `src/harness/docs/ingest/library.py` | 02 |
| `src/harness/docs/retrieval/hybrid_search.py`, `src/harness/docs/retrieval/rag.py` | 02 |
| `src/harness/inspect/decoder.py`, `src/harness/inspect/catalog/catalog_loader.py`, `src/harness/inspect/base.py` | 02 |
| `src/harness/diagnosis/schema.py` (CaseOutcome owner — BEING BUILT to match reverse agent's store), `src/harness/diagnosis/verifier.py` (record path) | 03 |
| `src/harness/diagnosis/scorer.py` (`_NON_DOC_SOURCES` guard only) | 04 |

## Reverse agent (prompts 08–05) — actual claims (verified mid-session)

| Files (owned by reverse agent) | Prompt |
|---|---|
| `src/harness/diagnosis/case_store.py`, `src/harness/diagnosis/case_library.py`, `src/harness/diagnosis/calibration.py` | BUILT — canonical API for the forward agent's 03/04 wiring |
| `src/harness/diagnosis/eval.py` (new), eval report commands + `run_eval` | 07 |
| `docs/MODEL_SWAP.md` (new) | 08 |
| `plan/subsystem.py`, `plan/priors_build.py` (new), `operator/tickets.py` | 05 |
| scorer recalibration parts of `scorer.py` | 06 |
| eval report commands | 07 |

API contract (reverse agent): `CaseStore(root).save/record/get/all`; `CaseLibrary(store, exclude_holdout).similar(symptom, model_key, top_k, outcome_min)` -> list[(CaseOutcome, float)]; module fn `case_library.render(records) -> list[str]`; `CaseOutcome` fields = run_id, llm_ident, actions_recommended, actions_taken, evidence_summary, evidence_hash, outcome, model_key, created_at (+ forward superset extras). Forward agent adapts to this (no duplicate store).

## Shared files — merge points (touched by BOTH)

- `src/harness/operator/cli.py`:
  - Forward agent adds: `docs add --platform`, `harness report`, model_ask + retriever wiring in `diagnose/console/session` runners.
  - Reverse agent adds: `harness priors update`, `harness calibrate`, `harness eval`.
  - Both: add NEW functions at the END of the file and register one subparser each near the existing registrations; do not rewrite each other's lines. Prefer inserting whole new helper functions rather than editing existing bodies.
- `src/harness/diagnosis/scorer.py`: forward agent edits only `_NON_DOC_SOURCES` (line ~140); reverse agent owns the calibration changes in `score_diagnosis`.
- `src/harness/diagnosis/engine.py`: forward owns all edits through prompt 04; reverse agent's prompt 06/07 may only ADD optional `EngineContext` fields (calibration_root) without altering existing fields or `run()` flow once prompted 01–04 are merged.

## Status

| Prompt | Status |
|---|---|
| 01 model-detection | landed (observed in tree: inspect/model.py, engine detection-first ordering, cli model_hint/ask wiring) |
| 02 model-tagged-rag | landed (platform chunking/tagging, hybrid platform filter, docs add --platform) |
| 03 outcome-capture | landed: CaseOutcome (single merged definition, reverse's canonical + calibration_llm on ConfidenceBreakdown), CaseStore WORM record + verifier.record, pending_case.json seeding, harness report |
| 04 case-retrieval | landed: CaseLibrary.similar/render, prompt `prior_cases` in both builders + preamble rule, EngineContext.case_library hook wired in engine.run() + session turn evidence, cli `_build_case_retriever`, `_NON_DOC_SOURCES` guard **"prior verified cases"** |
| 05 feedback-priors | landed (reverse: PriorModel/classify priors, priors_build.py, `harness priors update`, ticket record_outcome flow, 10 tests green) |
| 06 calibration | landed (reverse: calibration.py, scorer wiring, calibrate CLI, 16 tests green) |
| 07 offline-eval | landed (reverse: eval.py, run_eval CLI, baseline diff, 7 tests green) |
| 08 model-swap-runbook | landed (forward wrote docs/MODEL_SWAP.md; reverse verified+revised it 2026-08-12 against the real CLI: --llm backend+HARNESS_LLM_MODEL semantics, real eval report fields in the citation check, per-ident calibration commands; no runtime code) |

## Prompt 05/06/07/08 completion notes (forward agent, 2026-08-12)

- During the reverse agent's 05 work, `plan/subsystem.py` had an
  import-breaking bug: `PriorModel.defaults: dict = _SYMPTOM_TO_SUBSYSTEM`
  (mutable default) crashed dataclass construction and took down the whole
  suite for BOTH agents. Forward applied the minimal fix:
  `field(default_factory=lambda: dict(_SYMPTOM_TO_SUBSYSTEM))` and typed
  `keyword_multipliers: dict[str, dict[str, float]] | None = None`.
- `plan/priors_build.py` was created by the FORWARD agent (reverse's cli.py
  was written against it: `build_priors`, `dump_priors`, `load_priors` are
  forward-authored). Keep the API as-is.
- `diagnosis/eval.py` `--holdout-frac` BUGFIX (forward): `evaluate()` gained
  `frac: float = 0.25` and `run_eval` now passes `args.holdout_frac` -- the
  flag was silently ignored before (holdout was hardcoded 0.25). Tests added
  to `tests/unit/test_eval.py` (frac monotonicity + propagation into
  `exclude_holdout` via SpyLibrary).
- `docs/MODEL_SWAP.md` (prompt 08) authored by forward agent 2026-08-12:
  transfer table, pre-flight eval with `--update-baseline`/`--tolerance`,
  per-ident recalibration (old files retained = rollback), citation contract
  check >= 0.9, rollback, and audit-record keeping via the real `auditlog`
  API + `harness report`. No runtime code added (doc-only prompt).
- All 8 prompts are now LANDED. `pytest tests/unit -q`: 389 passed. Ruff:
  only the pre-existing forward-owned stragglers (allowlist.py:49,
  engine/session.py:69/80) remain, untouched by prompt work.

## Prompt 05 handoff note (reverse landed 2026-08-12)

- `plan/subsystem.py` (reverse-owned): `PriorModel` (frozen: `defaults`,
  `keyword_multipliers`, `covers`/`empty`, `multiplier` clamped to
  [0.5, 2.0]), `clamp_multiplier`, `SubsystemRanking.score` is now float,
  `classify(symptom_text, priors=None)` — byte-identical to the static table
  when priors None/empty.
- `plan/priors_build.py` (NEW, reverse-owned): `build_priors(cases,
  min_verified=10) -> PriorModel | None` (weights fixed=1.0/partial=0.5/
  not_fixed=-0.75/inconclusive=0.25/unknown=0.25; Laplace
  `(count_s+1)/(count_all+1)*n_subsystems`; net-zero-evidence keywords omitted;
  None under the gate), `dump_priors`/`load_priors` (JSON round-trip).
- `operator/tickets.py` (reverse-owned): `Ticketing.record_outcome` ABC default
  raises NotImplementedError; `NoOpTicketing.record_outcome` prints the
  `harness report --run <id> --outcome ...` one-liner; NEW module fn
  `record_outcome_safe(ticketing, ...)` = the harness-side try/except wrapper.
- `operator/cli.py` (reverse section): `harness priors update --cases --out
  --min-verified` (exit 1 + "N/M verified cases, priors inactive" under gate).
- Wiring: `EngineContext.priors` (object | None) ADDED to forward-owned
  `diagnosis/engine.py` and passed at `plan_collection(..., priors=...)` in
  `engine.run()` + `diagnosis/session.py` `__init__`. CLI wires
  `priors=_load_priors(args.out_dir)` in run_diagnose's single EngineContext
  (covers diagnose + session-mode + REPL routes, which all reuse it). run_console
  is probe-only (no planning step). Forward agent: keep these two optional-pass
  lines when editing engine.py/session.py.
- Tests: `tests/unit/test_priors.py` (10): classify identity without priors,
  multiplier applied/clamped/unknown->1.0, gate None, memory boost breaks
  static mce tie, storage dampened below 1.0 with mostly not_fixed, JSON
  round-trip, `priors update` CLI write + inactive, NoOp record_outcome ->
  real `run_report` lands store record, backend-without-method survives via
  `record_outcome_safe`.
- No git binary in this environment: prompt-05 commit must be made by the
  user/host per the protocol (reference "05" in the message).

## Prompt 07 handoff note (reverse landed 2026-08-12)

- `src/harness/diagnosis/eval.py` (NEW, reverse-owned): `holdout_ids` (sha256,
  `int(225*frac)` threshold), `evaluate(store, llm, rag, llm_ident, *, excluded,
  library)`, `replay_case`, ECE via `bin_index` edges. `exclude_holdout` is
  passed to BOTH `CaseLibrary(...)` construction and every `similar(...)` call.
- `operator/cli.py` (reverse section, after `_model_from_audit`): `_eval_llm`,
  `run_eval`, `_print_eval_report`; subparser registered after `report`.
  Baseline lives NEXT TO `--out` as `baseline.json`. Regressions (verdict
  accuracy drop or ECE rise beyond `--tolerance`) exit 1; `--update-baseline`
  rewrites. A missing/empty doc library is a HARD error (no silent pass).
- `tests/unit/test_eval.py` (7 tests): deterministic/stable holdout,
  spy-verified `exclude_holdout` propagation, stub replay report shape,
  degraded-retrieval recall drop, baseline write/tamper/update, hard error.
- Forward-owned ruff stragglers still in tree (do NOT touch):
  `engine/allowlist.py:49` (B019), `engine/session.py:69` (PYI034),
  `engine/session.py:80` (RUF059).

## Forward-agent merge notes (03/04 landed 2026-08-12)

- `schema.py`: CaseOutcome is ONE definition (reverse's, at file end). The forward
  draft at the top of the file was DELETED during merge — do not re-add a second
  definition (duplicate class shadowed the real one for pydantic validators).
- `engine.py` `EngineContext.case_library` field: ADDED by forward agent
  (Callable[[str, str | None], list[str]] | None, default None) — reverse agent
  must NOT add it again (dataclass would raise on redefinition).
- `session.py` `_model_key()`: NEW private helper (model_key or None when
  product unknown) — shared file, forward-owned.
- `cli.py` `_build_case_retriever(out_dir)`: NEW forward helper, appended near
  `run_diagnose`. Returns None when the case store is empty (no prior knowledge
  yet — by design). Forward wiring: `case_library=_build_case_retriever(args.out_dir)`
  in run_diagnose's EngineContext — reverse's calibration_root/llm_ident lines
  untouched.
- `scorer.py`: `_NON_DOC_SOURCES` gained `"prior verified cases"` (04 guard).
- Tests: `tests/unit/test_case_loop.py` (7 tests) — verifier WORM record, audit
  linkage, prompt prior-cases render, scorer guard, engine/session injection,
  library outcome+model ranking. Both agents run `python -m pytest tests/unit -q`
  (371 passed) and ruff on touched files.

## Post-prompt product feature: `harness setup` wizard (2026-08-12, reverse)

New machine onboarding in one command: `harness setup [--inventory <file>]
[--secret-dir <dir>]` walks through inventory creation/patching, LLM API key
registration (getpass, never echoed), SSH identity setup (generates an ed25519
pair via `ssh-keygen` when none exists, or registers an existing key file),
optional BMC/sudo passwords, then verifies EVERY vault path the inventory
references (exit 1 + listing when any are missing). Also added to the menu:
`setup` entry + an unconfigured-store hint banner; `_ALLOWED_FLAGS["setup"]`.
Ownership: `src/harness/operator/setup_cli.py` (NEW), `tests/unit/test_setup.py`
(10 tests, hermetic via monkeypatch.chdir). `cli.py` edits are additive only:
`run_setup_cmd`, subparser registrations, one menu action + hint line.
NOTE: never run the wizard with an un-hermetic CWD in tests — it writes
`inventory.yaml` relative to CWD.

## Reverse-agent notes (READ BEFORE MERGING forward 03/04)

Because prompts 03/04 had NOT landed when the reverse agent implemented 05-08, the
reverse agent created the following REQUIRED contracts themselves, exactly per the
prompt file paths (the contract). Forward agent: if you implement 03/04, match these
shapes/semantics exactly so the reverse work keeps passing; do not rename fields.

- `diagnosis/schema.py` — `CaseOutcome` (03 contract) + optional fields
  `evidence_summary: list[str]`, `cited_titles: list[str]` (07 contract).
  `ConfidenceBreakdown.calibration_llm: str | None` (06 contract, optional-only).
- `diagnosis/case_store.py` — `CaseStore` (03 contract; atomic write, append-only,
  `index.json` rebuild, secret-path rejection).
- `diagnosis/case_library.py` — `CaseLibrary(store, exclude_holdout)` +
  `similar(..., exclude_holdout)` / `render(records)` (04 contract; `exclude_holdout`
  is REQUIRED by `harness eval`).
- `diagnosis/verifier.py` — unchanged by reverse. Reverse adds `CaseStore.record`
  audit linkage instead (see below); forward may still add `Verifier.record`
  per 03 — keep both WORM-safe.
- `operator/cli.py` — reverse added at END: `run_calibrate`, `run_eval`,
  `run_priors_update`, `run_report` + subparser registrations; plus
  `pending_case.json` seeding at end of `run_diagnose` (03 contract) and
  `calibration_root`/priors wiring (additive optional arguments only).
- `diagnosis/scorer.py` — reverse owns `score_diagnosis` calibration parts:
  new optional params `model_agreement: float | None = None`,
  `calibration_root: Path | None = None`, `llm_ident: str | None = None`;
  `_NON_DOC_SOURCES` untouched (forward).
- `diagnosis/engine.py` — reverse adds ONLY optional `EngineContext` fields
  `calibration_root: Path | None = None`, `case_library` wiring hook
  (`Callable[[str, str | None], list[str]] | None`), `llm_ident` callable;
  `run()` flow untouched.

Reverse-agent claim table (update BEFORE each prompt, matches above):

Note for reverse agent: `docs_retriever` is now `(query, model_key)` everywhere
(engine.py, session.py, cli.py). `EngineContext` has `model_hint` and
`model_ask`; `model_hook(model, drifted)` is two-arg. `detect_with_fallback`
in `diagnosis/engine.py` is the shared decision point. Prompt 01 also added
`TargetAlias.model` + `--model` flag, and `prompts/PROGRESS.md` is the conflict
contract.