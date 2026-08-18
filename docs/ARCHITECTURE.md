# Architecture

## Module layout

```
src/harness/
  engine/      secure SSH + serial console (SOL) + command execution + read-only
               allowlist enforcement; bastion hops, retries, sudoers generation
  inspect/     register/sensor collectors (cpu_msr, ipmi, kernel, pcie, storage,
               bmc_console), decoder, curated register catalog (YAML)
  plan/        symptom -> subsystem -> minimal collector set (collector profiles)
  docs/        PDF ingestion (pymupdf), chunking, hybrid BM25+dense retrieval
               (RAG), parts graph (FRU/PN lookup)
  testlog/     operator-supplied test-harness log evidence (--test-log): parser
               for Quanta FAT `.log` files, structured failure model, log-source
               seam (files now, a website fetcher later)
  diagnosis/   pydantic schema, summarizer, LLM orchestrator, scorer,
               verifier (baseline diff), session state
  audit/       append-only hash-chained trace, secret redaction
  operator/    CLI + interactive menu + REPL + human approval gate + supervisor
  targets/     dynamic target resolution (named / console rack-cable / SSH-by-IP /
               alias)
  config/      inventory (vault paths only), secrets client, inventory lint
```

## Pipeline (diagnose)

```
--test-log (optional) ─▶ testlog.parse ─▶ failure codes/test names (redacted)
symptom ─▶ plan.symptom -> subsystem ─▶ minimal collector set
   ─▶ engine.runner (allowlist + force_read_only) ─▶ RegisterDump per probe
   ─▶ inspect.decoder ─▶ evidence
   ─▶ docs.retrieval (RAG over PDFs, seeded by symptom + test-log failures)
        + parts graph ─▶ context
   ─▶ diagnosis.engine (LLM) ─▶ prioritized repair actions (JSON schema)
   ─▶ diagnosis.scorer + diagnosis.verifier (baseline compare)
   ─▶ operator.gate (human approval) ─▶ operator.supervisor (time budget)
   ─▶ audit.trace (append-only, redacted)
   ─▶ pending_case.json (carries test_log_failures) ─▶ harness report ─▶
        case store ─▶ case_library.similar matches future runs by failure code
```

A `--test-log` run renders a `## Factory Test Log Evidence` section into every
prompt (single-shot and session), seeds doc retrieval with its failure queries
(capped), records its failure signatures on the pending case, and — after the
operator confirms the outcome with `harness report` — the verified case is
retrieved for future runs whose log shows the same failure code.

## Execution domains

- **SSH (OS path)** — `engine/runner.py` runs each argv as a parameterized
  `subprocess` (no shell interpolation) after validation against the exact
  template allowlist. Keys are materialized to temp files (mode 0o600) from the
  secret store; host keys are pinned.
- **Console (SOL path)** — `engine/sol.py` renders an `expect` script:
  `jumpin q<rack>-1 rm` → `start serial session -i <cable> [-p <port>]` →
  wait for node prompt → send probe → optional `sudo -S` password handshake.
  Only constructible at `trust_level` lab/qa; destructive or injection-y
  probes are rejected by `validate_serial_probe`.
- **BMC (IPMI path)** — `engine/bmc.py` for IPMI/BMC access with separate
  rotated credentials from SSH.

## Credential flow

1. Operator registers material with `harness secrets` (non-agent CLI) **or** on
   demand: `operator/credential_gate.py` wraps the store, and the first time a
   vault path is actually needed (LLM call / SSH open / console probe) the
   operator is prompted in the terminal — generate + install an SSH key onto the
   target, or `getpass` the LLM/BMC/sudo credentials. Non-tty runs skip prompting.
2. Inventory YAML stores **vault paths only** — `inventory_lint` rejects inline
   secrets at load.
3. Prompts and transcripts carry identifiers + vault paths; the LLM never sees
   key material.

## Run artifacts

Every run writes under `--out-dir` (default `harness_runs/<run-id>/`):

| File | Contents |
|---|---|
| `audit.jsonl` | Append-only hash-chained audit events, secrets redacted |
| `trace.json` | Pipeline trace (plan → collect → decode → diagnosis) |
| `diagnosis.json` | Final scored recommendations, all pending approval |
| `dumps.json` | Probe manifest; baseline for `harness verify` |
| `dumps/*.txt` | Raw collector output per subsystem |

Sessions persist under `--session-dir` as `session.json`.
