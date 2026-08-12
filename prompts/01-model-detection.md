# Prompt 01 — Deterministic model facts: tri-state, alias canonicalization, operator fallback, detection-before-retrieval

Implements: model detection the agent never has to guess, and a detection
ordering fix so the whole pipeline (RAG, case library, calibration) can be
model-consistent.

## Mission

Make the harness always know *how* it knows the server model, never let the LLM
be the source of the model, and guarantee model detection happens BEFORE
document retrieval.

## Read first

- `src/harness/inspect/model.py`
- `src/harness/diagnosis/engine.py` (the `run()` method and `EngineContext`)
- `src/harness/diagnosis/prompt.py` (`build_prompt`, `build_turn_evidence`)
- `src/harness/targets/aliases.py` and `src/harness/targets/resolver.py`
- `src/harness/operator/cli.py` around the `diagnose`/`console`/`session`
  subcommands and how the doc retriever callback is wired
- One existing test file, e.g. `tests/unit/test_targets.py`, for style

## Changes

### 1. `inspect/model.py` — tri-state detection with provenance

- Add to `DetectedModel`:
  - `source: str` — one of `"dmidecode" | "fru" | "operator" | "alias"`.
  - Keep `model_key` (canonical product name), but canonicalize it through a
    new alias map + normalizer:
    - `MODEL_ALIASES: dict[str, list[str]]` — canonical key -> list of known
      product-name variants (spellings, dash/space forms, short names). Start
      empty-ish with 2-3 illustrative entries.
    - `normalize_product(raw: str) -> str | None` — lowercase, collapse
      whitespace, strip vendor-noise tokens (e.g. "system", "server"), then
      resolve through `MODEL_ALIASES` to the canonical key; returns None when
      no alias matches and the raw string is empty.
    - `model_key` returns `normalize_product(product_name) or <lowercased raw>`.
- `detect_model` keeps its current dmidecode/FRU logic but sets `source`
  accordingly. Detection failure stays `None` (engine handles tri-state).
- Add a factory used by the engine for the two non-probe sources:
  `from_operator(product: str) -> DetectedModel` (source="operator") and
  `from_alias(key: str) -> DetectedModel` (source="alias").

### 2. `diagnosis/engine.py` — ordering fix + fallback chain

In `EngineContext`:
- `docs_retriever` signature changes from `Callable[[str], list[str]]` to
  `Callable[[str, str | None], list[str]]` (query, model_key-or-None).
- Add `model_ask: Callable[[], str | None] | None` — optional operator fallback:
  returns a product name string or None (non-blocking; automated runs leave it
  None).
- Add `model_hook` already exists (audit) — extend its payload semantics only
  if trivial; otherwise keep as-is.

In `run()`:
- Move `model = detect_model(...)` to BEFORE the step-0 `docs_retriever` call.
- Fallback chain:
  1. `detect_model(runner)` → `source="dmidecode|fru"`.
  2. If None and the resolved target alias has a stored model_key →
     `from_alias(key)` (source="alias").
  3. If still None and `model_ask` is set → prompt once; if the operator
     answers, `from_operator(name)`.
  4. If still None → model stays None; emit a progress note "model=unknown".
- Drift check: when the detected model_key differs from the alias-stored one,
  call `model_hook` with a record/flag the CLI can surface as a warning, and
  prefer the freshly DETECTED value (detection beats alias). Do NOT update the
  alias store automatically.
- Pass `model.model_key if model else None` into `docs_retriever`.

### 3. `targets/aliases.py` — cache model per target

- Store an optional `model` (canonical key) per alias; `targets add` gains a
  `--model <key>` flag. Path-only file format stays; keep it lint-clean.

### 4. `diagnosis/prompt.py` — model provenance in the prompt

- `## System` block (both builders) gains:
  `model_source={model.source if model else 'unknown'}`
  and when no model: `model=unknown (detection failed; treat register/bit
  meanings as unverified)`.
- Add one preamble rule to both system preambles: "The System section is
  hardware-detected fact. If you believe it is wrong based on the snippets,
  say so explicitly instead of silently assuming a different model."

### 5. CLI wiring

- The retriever callbacks passed by the CLI (in `operator/cli.py`) must accept
  and ignore-or-use the second argument (model_key) — no other behavior change
  yet (Prompt 02 makes it filter).
- Wire `model_ask` in interactive paths (menu/session/diagnose) so the prompt
  appears only when detection failed; keep one-shot/automated paths unset.

## Acceptance criteria

- New tests in `tests/unit/test_model.py`:
  - `normalize_product` alias resolution and None-on-empty; `model_key` output.
  - Fallback chain order: dmidecode → alias → operator → None, each source
    recorded on the resulting `DetectedModel`.
  - Drift: detected key differs from alias → detected wins and hook fires.
  - `run()` calls the retriever with the detected model_key (assert via a fake
    retriever), and calls it with None when detection fails and no fallback.
- `build_prompt` with no model renders the "unknown ... unverified" wording;
  with a model renders `model_source=`.
- Existing suite stays green: `pytest tests/unit -q` and `ruff check src tests`.