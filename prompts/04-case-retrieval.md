# Prompt 04 — Past-case retrieval: verified fleet history in the prompt

Implements: the agent improves with every fixed machine. Verified past cases
are retrieved by symptom + model and injected as cited, outcome-labeled history
— few-shot learning without any model fine-tuning.

## Mission

A `CaseLibrary` that scores past verified cases against the current symptom
(outcome-weighted, same-model boosted) and a new prompt section that presents
them as *fleet history, not vendor documentation* — while staying out of the
citation-supported score.

## Read first

- `prompts/03-outcome-capture.md` (the `CaseOutcome` contract you depend on)
- `src/harness/diagnosis/case_store.py` (from Prompt 03)
- `src/harness/docs/retrieval/hybrid_search.py` (reuse `bm25_score`)
- `src/harness/diagnosis/prompt.py`, `src/harness/diagnosis/engine.py`
- `src/harness/diagnosis/scorer.py` (`_NON_DOC_SOURCES`)
- `tests/unit/test_docs_library.py` for test style

## Changes

### 1. `diagnosis/case_library.py` (new module)

- `CaseLibrary(store: CaseStore, exclude_holdout: frozenset[str] = frozenset())`.
- `similar(symptom: str, model_key: str | None = None, top_k: int = 5,
  outcome_min: float = 0.0) -> list[(CaseOutcome, score)]`:
  - BM25 over symptom + `actions_recommended` + `actions_taken` text (reuse
    `bm25_score` from hybrid_search; build corpus stats over the case corpus).
  - Outcome weights: fixed=1.0, partial=0.5, inconclusive=0.25,
    not_fixed=0.1, unknown=0.5. Cases below `outcome_min` are excluded (a
    not-fixed case must not sail through as authoritative).
  - Same-model boost: exact `model_key` match multiplies the score by 1.5.
  - Never returns holdout ids when `exclude_holdout` is non-empty.
- `render(records) -> list[str]` — one prompt line per case:
  `[case {run_id}] model={model_key|'unknown'} outcome={outcome}: {symptom} -> {first action taken or 'n/a'}`.

### 2. Prompt — `## Prior Verified Cases`

- `build_prompt` and `build_turn_evidence` gain `prior_cases: list[str] = []`;
  render the section header:
  `## Prior Verified Cases (fleet history, NOT vendor documentation)` and the
  lines, or `(none yet)`.
- Add to both system preambles: "Prior Verified Cases are observed history from
  THIS fleet. Use them to weigh likelihoods, never as documentation: they MUST
  NOT be cited as a Reference source. If a prior case contradicts the vendor
  snippets, the snippets and catalog win."

### 3. Scorer — keep history out of citation support

- `scorer._NON_DOC_SOURCES` gains the exact section name
  (`"prior verified cases"`) so references to it can never count toward
  `retrieval_citation_support`.

### 4. Engine wiring

- `EngineContext` gains `case_library: Callable[[str, str | None], list[str]] | None` —
  (symptom, model_key) → rendered prior-case lines. In `run()`, after model
  detection: `prior = case_library(symptom, model.model_key if model else None)`
  and pass into `build_prompt`.
- CLI (diagnose / console / session paths) wires this from the run's `cases`
  dir when it exists; missing store → `(none yet)` behavior, never an error.

## Acceptance criteria

- `tests/unit/test_case_library.py`:
  - outcome weighting: a `fixed` case outscores a `not_fixed` one with
    identical symptom text; same-model boost beats cross-model at similar
    scores; `outcome_min` filters `not_fixed`.
  - holdout exclusion: ids in `exclude_holdout` never returned.
  - `render` output matches the exact prompt-line format.
- Prompt test: section headers present with and without cases; preamble
  contains the not-citable rule.
- Scorer test: a Reference whose source is "Prior Verified Cases" does not
  inflate `retrieval_citation_support`.
- `pytest tests/unit -q`, `ruff check src tests`.