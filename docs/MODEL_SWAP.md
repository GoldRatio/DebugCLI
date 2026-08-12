# LLM Swap Runbook

Procedure for migrating the diagnosis LLM from one model to another (e.g.
`gemini-2.5-flash` -> `claude-...`, or an in-house `openai`-compatible
service). The goal: a swap is a routine, calibrated, regression-gated
operation -- not a trust leap. Everything the model needs is data; everything
model-specific is explicitly re-derived before the new model serves reports.

Audience: an operator with `harness` installed and a case store + doc library
from prior diagnoses.

## 1. What transfers and what doesn't

| Asset | In a swap? | Why |
|---|---|---|
| Case records (`harness_runs/cases/`, WORM) | carries over as-is | ground truth for priors, calibration, eval holdout; writes are append-only so no history is lost or rewritten |
| Doc library + platform tags (`harness_docs/`) | carries over as-is | retrieval input; model-independent |
| Register catalog, collector profiles (`--profile`, subsystem tables) | carries over as-is | read-only probe definitions, unchanged by model choice |
| `harness eval` holdout set (sha256 over run ids) | carries over as-is | deterministic; both models are measured on the same cases |
| Audit chain (`audit.jsonl`) | carries over as-is | every run + verdict is chained; a rollback leaves it untouched |
| Calibration bins (`harness_runs/calibration/<ident>.json`) | MUST be re-derived | `model_agreement` is per-`llm_ident`; the old model's fix-rates do not describe the new one |
| `model_agreement` on sloppy citation | comes along with recalibration | per-ident bins measure agreement between predicted confidence and observed outcomes, so weak citing shows up there (and in `retrieval_citation_support`) instead of going unmeasured |
| Prompt formatting quirks / citation contract | re-verified | the new model must emit `Reference` entries with real titles+pages (see 4) |
| Priors (`harness_runs/priors.json`) | carries over as-is* | learned from verified outcomes, not from model id; recalibrate later if you like |

\* Priors are outcome-weighted, not model-weighted -- see
`harness priors update`; nothing forces a rebuild on swap.

## 2. Pre-flight on the new model

The eval CLI selects a *backend* (`--llm {stub,openai,gemini}`); the concrete
model name comes from the `HARNESS_LLM_MODEL` environment variable (or the
inventory `llm.model`). The report's `llm_ident` records exactly what was
replayed, so old- vs new-model runs are distinguishable.

1. Freeze the OLD model's numbers as the reference baseline:
   ```powershell
   $env:HARNESS_LLM_MODEL = "<old-model>"
   harness eval --cases .\harness_runs\cases --lib .\harness_docs --llm openai --out eval_old.json --update-baseline
   ```
   This writes `baseline.json` next to `eval_old.json`.
2. Run the same eval with the new model:
   ```powershell
   $env:HARNESS_LLM_MODEL = "<new-model>"
   harness eval --cases .\harness_runs\cases --lib .\harness_docs --llm openai --out eval_new.json
   ```
3. Acceptance: `verdict_accuracy` and `ece` must stay within `--tolerance`
   (default 0.05 absolute) of the old baseline. The command exits 1 on
   regression and prints the offending metrics.
4. If the new model regresses, do NOT proceed without an owner sign-off;
   record the decision in the audit (see 6) and either rework the prompt or
   stay on the old model.

## 3. Recalibration

Calibration is per `llm_ident`; the new model must build ITS OWN bins:

```powershell
harness calibrate --cases .\harness_runs\cases --llm <new-ident> --out .\harness_runs\calibration
```

- Old-model files (`.../calibration/<old-ident>.json`) are retained, never
  overwritten -- that is the rollback path.
- Sparse subsystem bins already fall back to the aggregate rate, and an
  empty store degrades scoring to the historical 0.5 default; a brand-new
  model scores conservatively until it accumulates verified outcomes.
- The scorer picks the right calibration automatically from the run's
  `llm_ident`; you only need to make sure scoring runs after the calibration
  store already exists (`default: harness_runs/calibration`).

## 4. Citation contract check

`retrieval_citation_support` (and therefore confidence) structurally depends
on references carrying a real title + page from the retrieved snippets. The
eval report re-measures this during replay, so the check is reading the
`eval_new.json` from the pre-flight:

- Top-level `mean_citation_support` (and `per_subsystem.<s>.mean_citation_support`)
  is the fraction of each replay's references that resolved against the
  retrieved snippets. It must be >= ~0.9.
- Sample the `cases[]` entries below the threshold: an entry's
  `citation_support` is the per-case fraction, `cited_titles` is what the
  replayed model actually cited, and `retrieved_titles` is what retrieval
  returned -- the pair shows whether the model cites real retrieved
  documents or invents titles/pages.

Example (one suspicious case, PowerShell):

```powershell
$r = Get-Content .\eval_new.json | ConvertFrom-Json
$r.cases | Where-Object { $_.citation_support -lt 0.9 } |
    Select-Object run_id, state, citation_support, @{n="cited";e={$_.cited_titles -join ", "}}
```

- Fraction >= ~0.9: citation contract holds, proceed.
- Below ~0.9: rework the prompt formatting (the citation rule in
  `prompts/`), then re-run the pre-flight eval -- do not ship a model that
  cites what it cannot support.

## 5. Rollback

Nothing in the swap is destructive; a rollback is a config change only:

1. Point live runs back at the old model (inventory `llm: model:` / the
   `--llm` backend or `HARNESS_LLM_MODEL`, whichever you changed).
2. The old calibration file was never touched, so `model_agreement` resolves
   against the old ident again automatically.
3. Optionally restore the old baseline:
   ```powershell
   $env:HARNESS_LLM_MODEL = "<old-model>"
   harness eval --cases .\harness_runs\cases --lib .\harness_docs --llm openai --update-baseline
   ```
4. Case store, audit chain, and doc library are untouched by the swap (WORM)
   -- no data repair needed.

## 6. Record keeping

Every swap must leave an audit trace. The audit chain is the run-linked
`audit.jsonl` that `harness report` appends to; use the same path for the
swap event:

1. Run a diagnosis with the new model, then close the loop as usual:
   ```powershell
   harness report --run <run_id> --outcome <verified outcome> --taken "..."
   ```
   (the first verified case from the new model is itself the first
   audit-linked evidence that it works end to end).
2. Keep the signed artifacts together: `eval_old.json` / `eval_new.json` /
   `baseline.json` and the per-ident calibration files carry `created_at` and
   `llm_ident`, so the pair (ident, eval diff, date) is on disk without extra
   files.
3. For sign-off: append a `model_swap` event to the run's audit log with the
   real `auditlog` API (what `harness report` uses):
   ```python
   from harness.audit.auditlog import AuditLog
   log.append("<run_id>", "model_swap",
              {"old": "gemini-2.5-flash", "new": "<new>",
               "baseline": "eval_old.json", "eval": "eval_new.json",
               "sign_off": "<owner>"})
   ```
   Use a generated `<run_id>` when the swap itself is not a diagnosis run.

## Checklist

- [ ] `harness eval --update-baseline` with the old model
- [ ] `harness eval` with the new model passes `--tolerance` (or owner sign-off)
- [ ] `harness calibrate --llm <new-ident>` ran; `<new-ident>.json` exists in the store
- [ ] citation contract sample >= 0.9 from `eval_new.json`
- [ ] rollback path verified: old calibration file still present
- [ ] `model_swap` event appended to the audit chain with sign-off