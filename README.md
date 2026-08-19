# DebugCLI — AI-Driven Server Debugging Companion

An AI-driven debugging companion for servers. It connects to a target over SSH
or a rack-manager serial console (SOL), performs **read-only** inspection of
hardware and software registers, cross-references vendor architecture
documentation (PDFs) with a hybrid retrieval/RAG pipeline, and returns a safe,
prioritized list of **human-approved** repair actions.

> **Absolute rule: the harness performs zero raw hardware writes.**
> No MSR write, no `setpci -w`, no `/dev/mem` write, no firmware flash.
> It only *recommends* — a human (or a separate, independently-audited layer)
> is the only writer.

---

## Features

- **Read-only by construction** — every command runs through a single
  `engine.runner` choke point validated against an **exact argv-template
  allowlist**; anything else is denied (`force_read_only`).
- **Two zero-YAML targeting modes** — target any server by rack/cable
  (`--rack Q61 --cable 8`) or by IP (`--address 10.0.0.50`); named inventory
  hosts keep working.
- **Serial-console path (lab/QA)** — `jumpin` → BMC serial session with an
  expect-script engine: automatic prompt waits, `sudo -S` password handshake
  (password fetched from the secret store, never embedded), host-key pinning.
- **Credential safety** — credentials are registered via a non-agent
  `harness secrets` CLI (or on demand: `harness menu` / `harness session` prompt
  the OPERATOR for each credential the moment it is needed — generate + install
  an SSH key onto the target, capture the LLM key and BMC/sudo passwords via
  `getpass`) and never pass through the LLM prompt, transcripts, or audit logs.
  Vault paths only; `inventory_lint` rejects inline secrets.
- **RAG over vendor PDFs** — hybrid BM25 + dense retrieval over chunked
  hardware docs, plus a parts-lookup graph for FRU/PN validation.
- **Audit trail** — append-only, hash-chained `audit.jsonl` per run with
  secret redaction; every diagnosis is reproducible from its own `harness_runs/<id>/`
  directory.
- **Interactive CLI** — a menu (`harness`), a Claude-Code-style chat session
  (`harness session`) with type-ahead and slash commands, and one-shot flags.

---

## Installation

Requires **Python 3.11+** (developed on 3.14, Windows PowerShell; the CLI is
cross-platform).

```powershell
pip install -e ".[test,docs]"
```

Dependencies: `pydantic`, `PyYAML`, `paramiko`; optional extras add
`pymupdf` + `tiktoken` (docs/RAG) and `pytest` + `pytest-cov` (tests).

## Quick start

```powershell
# First run with no inventory yet? `harness` auto-launches `harness setup`
# to create inventory.yaml + register credentials -- or do it by hand:

# 1. Point at an inventory (vault paths only — see src/harness/config/inventory.yaml)
harness lint --inventory inventory.yaml

# 2. Register credentials once (non-agent CLI, never echoed)
harness secrets add-ssh 10.0.0.50 --key-file .\secrets\id_ed25519 --secret-dir .\secrets
harness secrets set-password bmc-ro --secret-dir .\secrets

#    ...or skip registration entirely: the menu/session prompts you for each
#    credential the first time it is needed (never through the agent):
harness

# 3. Diagnose — interactive menu (no flags to remember)
harness

#    or one-shot / chat session, optionally seeding from a harness/FAT log
harness diagnose --inventory inventory.yaml --host h1 --test-log quanta_qmf_fat_*.log
harness session --inventory inventory.yaml --rack Q61 --cable 8
#   harness> /testlog quanta_qmf_fat_*.log
#   harness> Diagnose the server in Q61 Cable 8
```

## Targeting a server without per-server YAML

The inventory holds shared defaults; any machine can be targeted at launch
time or in natural language:

```powershell
# Console by rack/cable (needs a console_defaults: block in the inventory)
harness diagnose --inventory inventory.yaml --rack Q61 --cable 8 --symptom "ECC errors"
harness session --inventory inventory.yaml --rack Q61 --cable 8
#   harness> Diagnose the server in Q61 Cable 8

# SSH by IP (identity resolved from the secret store)
harness diagnose --inventory inventory.yaml --address 10.0.0.50 --symptom "ECC errors"
#   harness> Diagnose 10.0.0.50
```

Resolution order: `--host` > `--target <alias>` > `--rack/--cable` > `--address`.

## Command reference

| Command | Purpose |
|---|---|
| `harness` / `harness menu` | Interactive menu: pick target + action (includes a **model** picker) |
| `harness session` | Chat REPL; background runs, slash commands (`/model`, `/quit`) |
| `harness diagnose` | One-shot read-only diagnosis of a target |
| `harness console` | Run read-only probes over the serial console (lab/QA only) |
| `harness verify --baseline <dumps.json>` | Re-run collectors, compare error counters |
| `harness docs add <pdf...> \| ls \| rm \| reindex` | Manage the RAG PDF library |
| `harness lint --inventory <file>` | Validate inventory (rejects inline secrets) |
| `harness secrets add-ssh \| set-password \| list \| rm \| check` | Register credentials (never via the agent) |
| `harness targets add \| ls \| rm` | Short aliases for repeated targets (path-only file) |

## Test-log evidence (`--test-log`)

Before debugging, an operator can hand the pipeline a factory/harness run log
(e.g. a Quanta FAT `.log`) whose failures are parsed for error codes and test
names (`[FAIL][P02002001@PCIe Test Fail]`, `FAIL: pcie_cmp_chk`) and injected
as evidence into every prompt, seed doc (RAG) retrieval, and record on the
learning loop:

```powershell
harness diagnose --inventory inventory.yaml --host h1 --test-log quanta_qmf_fat_*.log
#  (--symptom is optional here; it derives from the log's first failure)
```

- Works in single-shot and session mode; `harness session` queues logs with
  `/testlog <path>`.
- The raw log is copied into `harness_runs/<id>/test_logs/`, and an audit
  `test_log_loaded` event records only metadata + failure codes (cleartext
  passwords in logs are redacted before anything reaches a prompt).
- Learning: `pending_case.json` carries the failure signatures; after the
  operator confirms the outcome (`harness report --run <id> --outcome fixed`),
  a future run whose log shows the same failure code surfaces the verified fix
  as a Prior Verified Case, pre-probe.

Common flags: `--inventory`, `--secret-dir`, `--docs-lib`, `--docs-dir`,
`--parts-csv`, `--out-dir`, `--llm {openai,gemini,stub}`,
`--llm-model <ident>` (e.g. `gemini/gemini-2.5-pro`), `--approval` /
`--approve-all`, `--max-turns`.

## Model selection

The LLM reasoning backend is picked like in opencode/pi:

- **`harness menu` → `model`** — arrow-key picker (type to filter) over the
  catalog: inventory `llm` block + well-known defaults + any custom models you
  have added. `+ add a custom model` registers a provider/model/url/key vault
  path for your own endpoint.
- **`harness session` → `/model`** (or `/model <ident>`) — same picker, or a
  direct ident, right inside the chat; the switch applies to the next run.
- **`--llm-model <ident>`** — pin one model for a one-shot `diagnose`/`session`.

The pick is remembered in `config/models.yaml` (machine-local, git-ignored) so
the next run keeps it. Precedence: `--llm-model` > `--llm` > remembered >
inventory `llm` block > default `openai/harness-diag`. Calibration is keyed per
model ident, so a swap never inherits another model's calibration.

## Documentation

| Document | Contents |
|---|---|
| [README.md](README.md) | This page |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module layout, data flow, pipeline stages |
| [docs/SECURITY.md](docs/SECURITY.md) | Security invariants, threat model, deployment notes |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | End-to-end walkthrough: menu, chat, console path |
| [TESTING.md](TESTING.md) | Unit tests + live console test against rack Q71 cable 8 |
| [REVISED_SPEC.md](REVISED_SPEC.md) | Full design spec and risk mitigations (rev 2.0) |

## Development

```powershell
pip install -e ".[test,docs]"
pytest            # 250 unit tests, no hardware needed
ruff check .
```

## Repository hygiene

- `keys/`, `secrets/`, `harness_runs/`, vendor PDFs, and `known_hosts` files
  are **git-ignored** — never commit credentials or run artifacts.
- The sample inventory in `src/harness/config/inventory.yaml` contains vault
  **paths only**; `harness lint` enforces this invariant.
- Add `docs/*.pdf` locally (they are architecture reference material); the RAG
  library (`harness docs add`) builds its index on your machine.

## License

Proprietary. All rights reserved.
