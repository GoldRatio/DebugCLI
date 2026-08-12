# Agent prompts: fleet-learning loop for DebugCLI

One prompt per work item, in implementation order. Each prompt is self-contained:
give it to a coding agent (opencode / Claude Code / similar) with the repo checked
out, and it should be able to implement, test, and verify the item without further
context.

## Order and dependencies

| # | Prompt | Depends on | What it delivers |
|---|--------|-----------|------------------|
| 01 | model-detection | — | Model facts (tri-state, aliases, fallback), detection-prior-to-retrieval ordering fix |
| 02 | model-tagged-rag | 01 | Platform-tagged doc chunks + model-filtered retrieval |
| 03 | outcome-capture | — | Case outcome records, `harness report` CLI, audit linkage (foundation of the learning loop) |
| 04 | case-retrieval | 03 | Past-case retrieval + prompt injection as verified prior cases |
| 05 | feedback-priors | 03, 04 | Verified outcome priors feeding `plan/subsystem.py` and ticket outcome flow |
| 06 | calibration | 03 | Outcome-calibrated confidence scoring, per LLM identity |
| 07 | offline-eval | 03, 04, 06 | `harness eval` regression harness with holdout and baseline diff |
| 08 | model-swap-runbook | 07 | `docs/MODEL_SWAP.md` — procedure for LLM migration |

## How to run each prompt

1. Apply the prompts strictly in order.
2. Read `prompts/PROGRESS.md` FIRST — it records file ownership between the
   forward agent (01–04) and the reverse agent (08–05). Never edit a file the
   other agent owns without updating PROGRESS.md and coordinating. Prefer whole
   new functions over edits to an existing function the other agent also edits.
3. Run `pytest tests/unit -q` and `ruff check src tests` before and after each
   item; the item is only done when both pass and the new tests it mandates
   exist and pass.
4. If an agent reports a conflict with a later prompt, fix to the later prompt's
   contract (the file paths below are the contract).

## Conventions every prompt shares (repeated in each file)

- Python 3.11+, `pydantic` for schemas, `dataclasses` for internal records.
- Follow the style of neighboring modules: module docstring, type hints, no
  comments unless they explain *why*.
- The harness is **read-only by construction** (`force_read_only`). Nothing
  here may write to hardware, recommend writes, or weaken the allowlist.
- Schema evolution rule (REVISED_SPEC section 4): schema changes are
  optional-field additions only; keep `schema_version` and document bumps.
- Secrets never enter prompts, transcripts, case records, or eval output.
- Run artifacts (cases, calibration, priors, eval reports) default under the
  run out-dir (`harness_runs/`), which is git-ignored; never write them to
  `secrets/`, `keys/`, or `src/`.
- Read the file(s) named in "Read first" before editing.
- Write tests in `tests/unit/` following the style of the existing test files;
  tests must be deterministic and never require network or hardware.